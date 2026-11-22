"""Tool-call grammar for the HG4 Fusion Gateway.

* `render_tool_system_prompt` renders OpenAI `tools` into model-native system instructions
  (`<tool_call>{json}</tool_call>` Hermes/Qwen JSON or `<function=name><parameter=k>v</parameter></function>` XML).
* `parse_tool_calls` extracts tool calls in every common local-model dialect:
    - `<tool_call>{"name": ..., "arguments": ...}</tool_call>` (Hermes / Qwen / Nemotron)
    - `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` (Qwen3-Coder / LFM3)
    - bare `<function=name>...</function>` and Llama-3.1 `<function=name>{json}</function>`
    - Llama `<|python_tag|>{json}`, Mistral `[TOOL_CALLS][...]`, fenced ```json {"name":..,"arguments":..}``` blocks
* `repair_json` deterministically repairs trailing commas, single quotes, Python literals, unquoted keys,
  raw newlines / unescaped quotes inside code strings, missing commas/colons and missing closing brackets.
* Arguments are coerced against the JSON Schema in `parameters` and validated for `required` keys, so clients
  always receive `function.arguments` as a valid JSON string.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "JsonRepairError",
    "ParsedToolCall",
    "ParseResult",
    "repair_json",
    "coerce_arguments",
    "validate_arguments",
    "render_tool_system_prompt",
    "active_tools",
    "parse_tool_calls",
    "has_tool_call_markup",
    "new_call_id",
]


class JsonRepairError(ValueError):
    pass


# --------------------------------------------------------------------------------------------------------------------
# Deterministic JSON repair
# --------------------------------------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_BARE_RE = re.compile(r"[A-Za-z0-9_.+\-$]+")
_KEY_AHEAD_RE = re.compile(r"\s*[\"'][^\"'\n]{0,200}[\"']\s*:")
_LITERALS = {
    "true": "true", "True": "true", "TRUE": "true",
    "false": "false", "False": "false", "FALSE": "false",
    "null": "null", "None": "null", "NULL": "null", "undefined": "null",
    "NaN": "null", "Infinity": "null", "-Infinity": "null",
}


class _Frame:
    __slots__ = ("kind", "state")

    def __init__(self, kind: str):
        self.kind = kind  # "{" or "["
        # object: key -> colon -> value -> after ; array: value -> after
        self.state = "key" if kind == "{" else "value"


def _fix_number(tok: str) -> str | None:
    t = tok
    if t.startswith("+"):
        t = t[1:]
    if t.startswith("."):
        t = "0" + t
    if t.startswith("-."):
        t = "-0" + t[1:]
    if t.endswith("."):
        t = t[:-1]
    if re.fullmatch(r"-?0\d+", t):
        t = str(int(t))
    return t if _NUM_RE.match(t) else None


_NEXT_KEY_RE = re.compile(r",\s*(?:[\"'][^\"'\n]{0,200}[\"']|[A-Za-z_$][\w$\-]*)\s*:")
_BARE_KEY_AHEAD_RE = re.compile(r"[A-Za-z_$][\w$\-]*\s*:(?!//)")


def _plausible_close(s: str, j: int, frame: _Frame | None, depth: int) -> bool:
    """Decide whether a quote whose next significant char is `s[j]` really terminates the current string."""
    n = len(s)
    if j >= n:
        return True
    nxt = s[j]
    in_key = frame is not None and frame.kind == "{" and frame.state == "key"
    if nxt == ":":
        return in_key
    if in_key:
        return False
    if nxt == ",":
        if frame is None or frame.kind == "[":
            return True
        rest = s[j + 1:].lstrip()
        return bool(_NEXT_KEY_RE.match(s, j)) or not rest or rest[0] in "}]"
    if nxt in "}]":
        k = j + 1
        while k < n and s[k] in " \t\r\n":
            k += 1
        return k >= n or s[k] in ",}]" or depth <= 1
    if nxt in "\"'":
        return bool(_KEY_AHEAD_RE.match(s, j))
    if frame is not None and frame.kind == "{":
        return bool(_BARE_KEY_AHEAD_RE.match(s, j))  # missing comma before an unquoted key
    return False


def _repair_json_text(s: str) -> str:
    """Single-pass tolerant rewriter producing strict JSON text from a glitchy JSON-ish fragment."""
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start < 0:
        raise JsonRepairError("no JSON object or array found")
    s = s[start:]
    n = len(s)
    out: list[str] = []
    stack: list[_Frame] = []
    i = 0

    def before_value() -> None:
        if not stack:
            return
        f = stack[-1]
        if f.kind == "{":
            if f.state == "after":
                out.append(",")
                f.state = "key"
            elif f.state == "colon":
                out.append(":")
                f.state = "value"
        else:
            if f.state == "after":
                out.append(",")
            f.state = "value"

    def after_value() -> None:
        if not stack:
            return
        f = stack[-1]
        if f.kind == "{":
            f.state = "colon" if f.state == "key" else "after"
        else:
            f.state = "after"

    def strip_trailing_comma() -> None:
        j = len(out) - 1
        while j >= 0 and out[j].isspace():
            j -= 1
        if j >= 0 and out[j] == ",":
            del out[j]

    def close_frame() -> None:
        f = stack.pop()
        if f.kind == "{":
            if f.state == "colon":
                out.append(":null")
            elif f.state == "value":
                out.append("null")
            elif f.state == "key":
                strip_trailing_comma()
            out.append("}")
        else:
            if f.state == "value":
                strip_trailing_comma()
            out.append("]")
        after_value()

    while i < n:
        c = s[i]
        if c in "\"'":
            quote = c
            before_value()
            out.append('"')
            i += 1
            closed = False
            while i < n:
                ch = s[i]
                if ch == "\\":
                    nx = s[i + 1] if i + 1 < n else ""
                    if nx == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2:i + 6] or ""):
                        out.append(s[i:i + 6])
                        i += 6
                    elif nx and nx in "\"\\/bfnrt":
                        out.append("\\" + nx)
                        i += 2
                    elif nx == "'":
                        out.append("'")
                        i += 2
                    elif nx == "\n":
                        out.append("\\n")
                        i += 2
                    else:
                        out.append("\\\\")
                        i += 1
                    continue
                if ch == quote:
                    j = i + 1
                    while j < n and s[j] in " \t\r\n":
                        j += 1
                    if _plausible_close(s, j, stack[-1] if stack else None, len(stack)):
                        out.append('"')
                        i += 1
                        closed = True
                        break
                    out.append('\\"' if ch == '"' else ch)  # unescaped quote inside a string (typical in code)
                    i += 1
                    continue
                if ch == '"':
                    out.append('\\"')
                elif ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                elif ord(ch) < 0x20:
                    out.append("\\u%04x" % ord(ch))
                else:
                    out.append(ch)
                i += 1
            if not closed:
                out.append('"')
            after_value()
            continue
        if c in " \t\r\n":
            out.append(c)
            i += 1
            continue
        if c in "{[":
            before_value()
            out.append(c)
            stack.append(_Frame(c))
            i += 1
            continue
        if c in "}]":
            if not stack:
                break
            want = "{" if c == "}" else "["
            if any(f.kind == want for f in stack):
                while stack and stack[-1].kind != want:
                    close_frame()
                close_frame()
            i += 1
            if not stack:
                break
            continue
        if c == ",":
            if stack and stack[-1].kind == "{" and stack[-1].state in ("colon", "value"):
                out.append(":null," if stack[-1].state == "colon" else "null,")
                stack[-1].state = "key"
            elif stack and stack[-1].state == "after":
                out.append(",")
                stack[-1].state = "key" if stack[-1].kind == "{" else "value"
            i += 1
            continue
        if c == ":":
            if stack and stack[-1].kind == "{" and stack[-1].state == "colon":
                out.append(":")
                stack[-1].state = "value"
            i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] in "/*":
            if s[i + 1] == "/":
                j = s.find("\n", i)
                i = n if j < 0 else j
            else:
                j = s.find("*/", i + 2)
                i = n if j < 0 else j + 2
            continue
        if c == "#":
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        is_key = bool(stack) and stack[-1].kind == "{" and stack[-1].state in ("key", "after")
        m = _BARE_RE.match(s, i)
        tok = m.group(0) if m else ""
        end = m.end() if m else i
        if is_key or not (tok in _LITERALS or _fix_number(tok) is not None):
            # unquoted key / string value: capture up to the next structural delimiter
            stop = ":" if is_key else ",}]\n"
            j = i
            while j < n and s[j] not in stop and not (is_key and s[j] in "{}[],\n"):
                j += 1
            tok, end = s[i:j].strip(), j
        if not tok:
            i += 1
            continue
        i = end
        before_value()
        if is_key:
            out.append(json.dumps(tok))
        elif tok in _LITERALS:
            out.append(_LITERALS[tok])
        else:
            num = _fix_number(tok)
            out.append(num if num is not None else json.dumps(tok))
        after_value()
    while stack:
        close_frame()
    return "".join(out)


def repair_json(text: Any) -> Any:
    """Parse `text` as JSON, deterministically repairing common local-model glitches. Raises `JsonRepairError`."""
    if isinstance(text, (dict, list)):
        return text
    if text is None:
        raise JsonRepairError("empty JSON")
    s = str(text).strip()
    fence = re.match(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```\s*$", s, re.S)
    if fence:
        s = fence.group(1).strip()
    if not s:
        raise JsonRepairError("empty JSON")
    try:
        return json.loads(s)
    except ValueError:
        pass
    fixed = _repair_json_text(s)
    try:
        return json.loads(fixed)
    except ValueError as exc:
        raise JsonRepairError(f"unrepairable JSON: {exc}") from exc


# --------------------------------------------------------------------------------------------------------------------
# JSON Schema coercion & validation
# --------------------------------------------------------------------------------------------------------------------

def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _schema_types(schema: dict[str, Any]) -> list[str]:
    t = schema.get("type")
    if isinstance(t, list):
        return [str(x) for x in t]
    if isinstance(t, str):
        return [t]
    for comb in ("anyOf", "oneOf"):
        if isinstance(schema.get(comb), list):
            out: list[str] = []
            for sub in schema[comb]:
                if isinstance(sub, dict):
                    out.extend(_schema_types(sub))
            return out
    if "properties" in schema:
        return ["object"]
    if "items" in schema:
        return ["array"]
    return []


def _coerce_value(v: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict) or not schema:
        return v
    types = _schema_types(schema)
    if v is None and ("null" in types or not types):
        return None
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and v not in enum and isinstance(v, str):
        for e in enum:
            if isinstance(e, str) and e.lower() == v.strip().lower():
                return e
    for t in types or ["any"]:
        try:
            if t == "string":
                if isinstance(v, str):
                    return v
                if isinstance(v, (dict, list, bool)) or v is None:
                    return json.dumps(v, ensure_ascii=False)
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                return str(v)
            if t == "integer":
                if isinstance(v, bool):
                    continue
                if isinstance(v, int):
                    return v
                if isinstance(v, float) and v.is_integer():
                    return int(v)
                if isinstance(v, str) and re.fullmatch(r"\s*[-+]?\d+(\.0*)?\s*", v):
                    return int(float(v))
                continue
            if t == "number":
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    return v
                if isinstance(v, str):
                    f = float(v.strip())
                    return int(f) if re.fullmatch(r"\s*[-+]?\d+\s*", v) else f
                continue
            if t == "boolean":
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    lv = v.strip().lower()
                    if lv in ("true", "yes", "y", "1", "on"):
                        return True
                    if lv in ("false", "no", "n", "0", "off", ""):
                        return False
                continue
            if t == "array":
                vv = v
                if isinstance(vv, str):
                    vv = repair_json(vv.strip()) if vv.strip().startswith("[") else [vv]
                elif not isinstance(vv, list):
                    vv = [vv]
                if not isinstance(vv, list):
                    continue
                items = schema.get("items") if isinstance(schema.get("items"), dict) else {}
                return [_coerce_value(x, items) for x in vv]
            if t == "object":
                vv = repair_json(v) if isinstance(v, str) else v
                if not isinstance(vv, dict):
                    continue
                return coerce_arguments(vv, schema)
            if t == "null":
                if v is None or (isinstance(v, str) and v.strip().lower() in ("null", "none", "")):
                    return None
                continue
            if t == "any":
                return v
        except (ValueError, TypeError, JsonRepairError):
            continue
    return v


def coerce_arguments(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce argument values (and near-miss key spellings like `file_path` -> `filePath`) to the JSON Schema."""
    if not isinstance(args, dict):
        return {}
    if not isinstance(schema, dict):
        return dict(args)
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    norm_map = {_norm_key(k): k for k in props}
    out: dict[str, Any] = {}
    for k, v in args.items():
        key = k
        if key not in props:
            alt = norm_map.get(_norm_key(k))
            if alt is not None and alt not in args:
                key = alt
        if key in props:
            out[key] = _coerce_value(v, props[key] if isinstance(props[key], dict) else {})
        elif schema.get("additionalProperties") is False:
            continue
        else:
            out[key] = v
    for k in schema.get("required") or ():
        sub = props.get(k)
        if k not in out and isinstance(sub, dict) and "default" in sub:
            out[k] = sub["default"]
    return out


def _type_ok(t: str, v: Any) -> bool:
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    if t == "null":
        return v is None
    return True


def validate_arguments(args: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    """Return a list of human-readable schema problems (empty list == valid)."""
    if not isinstance(schema, dict):
        return []
    problems: list[str] = []
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for r in schema.get("required") or ():
        sub = props.get(r) if isinstance(props.get(r), dict) else {}
        if r not in args or (args[r] is None and "null" not in _schema_types(sub)):
            problems.append(f"missing required parameter '{r}'")
    for k, v in args.items():
        sub = props.get(k)
        if not isinstance(sub, dict):
            continue
        types = _schema_types(sub)
        if types and not any(_type_ok(t, v) for t in types):
            problems.append(f"parameter '{k}' should be {'/'.join(types)}")
            continue
        enum = sub.get("enum")
        if isinstance(enum, list) and enum and v not in enum:
            problems.append(f"parameter '{k}' must be one of {enum}")
    return problems


# --------------------------------------------------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------------------------------------------------

def _fn_of(tool: dict[str, Any]) -> dict[str, Any]:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return fn if isinstance(fn, dict) else {}


def active_tools(tools: list[dict[str, Any]] | None, tool_choice: Any = None) -> list[dict[str, Any]]:
    """Return the list of function specs that are callable for this request (honours `tool_choice`)."""
    if not tools or tool_choice == "none":
        return []
    fns = [_fn_of(t) for t in tools if isinstance(t, dict)]
    fns = [f for f in fns if f.get("name") and not f.get("defer_loading")]
    if isinstance(tool_choice, dict):
        want = _fn_of(tool_choice).get("name") or tool_choice.get("name")
        if want:
            fns = [f for f in fns if f.get("name") == want] or fns
    return fns


def render_tool_system_prompt(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
    style: str = "json",
) -> str:
    """Render OpenAI tools as model-native system instructions. `style` is `json` (Hermes/Qwen) or `xml`."""
    fns = active_tools(tools, tool_choice)
    if not fns:
        return ""
    lines = []
    for f in fns:
        spec = {"name": f.get("name"), "description": f.get("description") or "",
                "parameters": f.get("parameters") or {"type": "object", "properties": {}}}
        lines.append(json.dumps({"type": "function", "function": spec}, ensure_ascii=False, sort_keys=True))
    out = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        *lines,
        "</tools>",
        "",
    ]
    if style == "xml":
        out += [
            "To call a function, reply with the following format and nothing else inside the block:",
            "<tool_call>",
            "<function=example_function_name>",
            "<parameter=example_parameter_1>",
            "value_1",
            "</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "Rules: function names must match exactly; include every required parameter; "
            "multi-line values (code, file contents) go verbatim between the parameter tags; "
            "objects and arrays are written as JSON.",
        ]
    else:
        out += [
            "For each function call, return a JSON object with the function name and arguments within "
            "<tool_call></tool_call> XML tags:",
            "<tool_call>",
            '{"name": <function-name>, "arguments": <args-json-object>}',
            "</tool_call>",
            "",
            "Rules: emit strict JSON (double quotes, no trailing commas, escape newlines in strings as \\n); "
            "function names must match exactly; include every required parameter.",
        ]
    if tool_choice == "required":
        out.append("You MUST call at least one function in this turn.")
    elif isinstance(tool_choice, dict) and fns:
        out.append(f"You MUST call the function `{fns[0]['name']}` in this turn.")
    else:
        out.append("If no function is needed, answer the user directly without any <tool_call> block.")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""
    problems: list[str] = field(default_factory=list)
    repaired: bool = False

    @property
    def valid(self) -> bool:
        return not self.problems

    def to_openai(self, index: int | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id or new_call_id(),
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }
        if index is not None:
            d["index"] = index
        return d


@dataclass
class ParseResult:
    content: str
    tool_calls: list[ParsedToolCall]
    errors: list[str]
    had_markup: bool

    @property
    def valid_calls(self) -> list[ParsedToolCall]:
        return [c for c in self.tool_calls if c.valid]

    @property
    def all_valid(self) -> bool:
        return not self.errors and all(c.valid for c in self.tool_calls)

    @property
    def problems(self) -> list[str]:
        out = list(self.errors)
        for c in self.tool_calls:
            out.extend(f"tool '{c.name}': {p}" for p in c.problems)
        return out

    def openai_tool_calls(self) -> list[dict[str, Any]]:
        return [c.to_openai(i) for i, c in enumerate(self.valid_calls)]


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


_MARKUP_RE = re.compile(r"<tool_call>|<function=|<\|python_tag\|>|\[TOOL_CALLS\]", re.I)


def has_tool_call_markup(text: str) -> bool:
    return bool(_MARKUP_RE.search(text or ""))


_XML_FN_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|\Z)", re.S)


def _strip_one_newline(v: str) -> str:
    if v.startswith("\r\n"):
        v = v[2:]
    elif v.startswith("\n"):
        v = v[1:]
    if v.endswith("\r\n"):
        v = v[:-2]
    elif v.endswith("\n"):
        v = v[:-1]
    return v


def _parse_xml_function(name: str, body: str) -> tuple[str, dict[str, Any], bool]:
    name = name.strip().strip("\"'")
    params = list(_XML_PARAM_RE.finditer(body))
    if params:
        args = {m.group(1).strip().strip("\"'"): _strip_one_newline(m.group(2)) for m in params}
        return name, args, False
    st = body.strip()
    if st.startswith("{") or st.startswith("```"):
        obj = repair_json(st)
        if isinstance(obj, dict):
            if set(obj.keys()) <= {"name", "arguments", "parameters"} and ("arguments" in obj or "parameters" in obj):
                obj = obj.get("arguments", obj.get("parameters")) or {}
            return name, obj if isinstance(obj, dict) else {}, True
    return name, {}, False


def _calls_from_json_obj(obj: Any) -> list[tuple[str, Any]]:
    """Extract (name, raw_arguments) pairs from a parsed JSON object/array in Hermes/OpenAI/Llama shapes."""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, list):
        for x in obj:
            out.extend(_calls_from_json_obj(x))
        return out
    if not isinstance(obj, dict):
        return out
    if isinstance(obj.get("tool_calls"), list):
        return _calls_from_json_obj(obj["tool_calls"])
    if isinstance(obj.get("function"), dict) and obj["function"].get("name"):
        f = obj["function"]
        return [(str(f["name"]), f.get("arguments", f.get("parameters", {})))]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, str) and name:
        args = obj.get("arguments", obj.get("parameters", obj.get("args", obj.get("input", {}))))
        out.append((name, args))
    return out


def _balanced_json_spans(text: str) -> list[tuple[int, int]]:
    """Locate top-level `{...}` spans (string-aware) for bare-JSON tool-call detection."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_s = False
    esc = False
    for i, ch in enumerate(text):
        if in_s:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_s = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def parse_tool_calls(text: str, tools: list[dict[str, Any]] | None, tool_choice: Any = None) -> ParseResult:
    """Extract, repair, coerce and validate tool calls from raw model output."""
    raw = text or ""
    fns = {f["name"]: f for f in active_tools(tools, tool_choice)}
    lower_map = {k.lower(): k for k in fns}
    calls: list[ParsedToolCall] = []
    errors: list[str] = []
    had_markup = has_tool_call_markup(raw)
    if not fns:
        return ParseResult(content=raw, tool_calls=[], errors=[], had_markup=had_markup)

    found: list[tuple[str, Any, bool]] = []  # (name, raw_args, repaired)
    consumed: list[tuple[int, int]] = []

    def take_xml(src: str) -> None:
        for m in _XML_FN_RE.finditer(src):
            try:
                found.append(_parse_xml_function(m.group(1), m.group(2)))
            except JsonRepairError as exc:
                errors.append(f"tool call '{m.group(1).strip()}': {exc}")

    def take_json_block(block: str) -> None:
        b = block.strip()
        if not b:
            return
        if "<function=" in b:
            take_xml(b)
            return
        try:
            strict = True
            try:
                obj = json.loads(re.sub(r"^```[A-Za-z0-9_-]*\s*|```$", "", b).strip())
            except ValueError:
                strict = False
                obj = repair_json(b)
            pairs = _calls_from_json_obj(obj)
            if not pairs:
                errors.append("tool call block without a function name")
            for nm, a in pairs:
                found.append((nm, a, not strict))
        except JsonRepairError as exc:
            errors.append(f"tool call: {exc}")

    # 1) <tool_call> ... </tool_call> (closing tag optional for truncated output)
    for m in re.finditer(r"<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|\Z)", raw, re.S):
        take_json_block(m.group(1))
        consumed.append(m.span())
    if not consumed:
        # 2) bare <function=...> blocks
        for m in _XML_FN_RE.finditer(raw):
            take_xml(m.group(0))
            consumed.append(m.span())
    if not consumed:
        # 3) Llama <|python_tag|> / Mistral [TOOL_CALLS]
        m = re.search(r"(<\|python_tag\|>|\[TOOL_CALLS\])(.*)\Z", raw, re.S)
        if m:
            take_json_block(m.group(2).split("<|eom_id|>")[0].split("<|eot_id|>")[0])
            consumed.append(m.span())
    if not consumed:
        # 4) fenced or bare JSON objects naming a known tool
        for fm in re.finditer(r"```(?:json|tool_call|tool)?\s*\n?(.*?)```", raw, re.S):
            try:
                obj = repair_json(fm.group(1))
            except JsonRepairError:
                continue
            pairs = [(nm, a) for nm, a in _calls_from_json_obj(obj) if nm in fns or nm.lower() in lower_map]
            if pairs:
                found.extend((nm, a, False) for nm, a in pairs)
                consumed.append(fm.span())
        if not consumed:
            for a, b in _balanced_json_spans(raw):
                try:
                    obj = json.loads(raw[a:b])
                except ValueError:
                    continue
                pairs = [(nm, x) for nm, x in _calls_from_json_obj(obj) if nm in fns or nm.lower() in lower_map]
                if pairs:
                    found.extend((nm, x, False) for nm, x in pairs)
                    consumed.append((a, b))

    rest = raw
    for a, b in sorted(consumed, reverse=True):
        rest = rest[:a] + rest[b:]
    content = re.sub(r"</?tool_call>|<\|eom_id\|>|<\|eot_id\|>", "", rest).strip()

    for nm, raw_args, repaired in found:
        name = nm.strip()
        if name not in fns:
            alt = lower_map.get(name.lower()) or lower_map.get(name.lower().replace("-", "_"))
            if alt is None:
                errors.append(f"unknown tool '{name}'")
                continue
            name = alt
        args_obj: Any = raw_args
        if isinstance(args_obj, str):
            st = args_obj.strip()
            if not st:
                args_obj = {}
            else:
                try:
                    args_obj = json.loads(st)
                except ValueError:
                    try:
                        args_obj = repair_json(st)
                        repaired = True
                    except JsonRepairError as exc:
                        errors.append(f"tool '{name}': arguments {exc}")
                        continue
        if args_obj is None:
            args_obj = {}
        if not isinstance(args_obj, dict):
            errors.append(f"tool '{name}': arguments must be a JSON object")
            continue
        schema = fns[name].get("parameters") if isinstance(fns[name].get("parameters"), dict) else None
        coerced = coerce_arguments(args_obj, schema)
        problems = validate_arguments(coerced, schema)
        calls.append(ParsedToolCall(name=name, arguments=coerced, id=new_call_id(), problems=problems,
                                    repaired=repaired or coerced != args_obj))
    return ParseResult(content=content, tool_calls=calls, errors=errors, had_markup=had_markup or bool(found))
