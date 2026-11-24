"""Clover Studio Capability S1: Semantics / CSI Declined-Tool Negation Filter & Non-Answer Guard.

Distilled cleanly from `clover-ai-system/studio/clover/semantics.py` (`classify_user_turn_Role`
declined-option pruning & `classify_assistant_output` non-answer promise detection) with zero
external dependencies, zero system-prompt mutation, and strict prompt-only inputs.
"""

from __future__ import annotations

import ast
import re
from typing import Any

# Canonical mapping from natural-language action phrases to OpenCode / OpenAI tool names.
_TOOL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bash": ("bash", "shell", "terminal", "command", "commands", "execute", "executing"),
    "websearch": ("websearch", "web search", "search the web"),
    "webfetch": ("webfetch", "web fetch", "fetch url", "fetch any url", "fetch urls", "fetch any urls"),
    "write": ("write", "writing files", "write any files", "create files"),
    "edit": ("edit", "editing", "edit any files", "modify files"),
    "grep": ("grep",),
    "glob": ("glob",),
    "read": ("read",),
}

_CLAUSE_SPLIT_RE = re.compile(r"(?:[;\n]|\s+—\s+|\s+--\s+|\binstead\b|\bbut\b|\brather\b)", re.IGNORECASE)

_NEGATION_CUE_RE = re.compile(
    r"\b(?:do\s+not|don['’]t|never|without|skip|refrain\s+from|avoid|forbid|forbidden\s+to|no\s+tool)\b",
    re.IGNORECASE,
)

_POSITIVE_TOOL_CUE_RE = re.compile(
    r"\b(?:call|use|invoke|run)\s+(?:the\s+)?[`'\"]?(read|glob|grep|bash|write|edit|webfetch|websearch)[`'\"]?\b",
    re.IGNORECASE,
)

_PURE_TEXT_DELIVERABLE_RE = re.compile(
    r"\b(?:answer\s+from\s+memory|only\s+output|just\s+tell\s+me|just\s+give|in\s+one\s+line|plain\s+fenced\s+code\s+block)\b",
    re.IGNORECASE,
)

_CODE_REQUEST_RE = re.compile(
    r"(?:\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|\bclass\s+[A-Za-z_][A-Za-z0-9_]*\b|\bwrite\s+(?:a\s+)?(?:complete\s+)?python\s+function\b)",
    re.IGNORECASE,
)

_PROMISE_LEADIN_RE = re.compile(
    r"\b(?:i\s+will|i['’]ll|let\s+me|sure,\s+i|certainly,\s+let\s+me|okay,\s+let\s+me|first\s+i['’]ll|to\s+solve\s+this,\s+i\s+need)\b",
    re.IGNORECASE,
)


def detect_declined_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[set[str], set[str], bool]:
    """Inspect the latest user turn and return `(declined_tool_names, explicitly_requested_tools, pure_text_mode)`."""
    if not messages or not tools:
        return set(), set(), False

    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            last_user = content if isinstance(content, str) else str(content or "")
            break
    if not last_user.strip():
        return set(), set(), False

    available = {
        str(((t.get("function") or {}).get("name")) or "")
        for t in tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    available.discard("")
    if not available:
        return set(), set(), False

    declined: set[str] = set()
    positive: set[str] = set()

    for raw_clause in _CLAUSE_SPLIT_RE.split(last_user):
        clause = raw_clause.strip()
        if not clause:
            continue
        low = clause.lower()
        if _NEGATION_CUE_RE.search(clause):
            for canonical, syns in _TOOL_SYNONYMS.items():
                if canonical not in available:
                    continue
                for syn in syns:
                    if re.search(r"\b" + re.escape(syn) + r"\b", low):
                        declined.add(canonical)
                        break
            # "write or edit any files" / "web search and don't fetch"
            if re.search(r"\b(?:write|edit)\s+any\s+files\b", low):
                for c in ("write", "edit"):
                    if c in available:
                        declined.add(c)
            if re.search(r"\bfetch\s+any\s+urls?\b", low) and "webfetch" in available:
                declined.add("webfetch")
        else:
            for m in _POSITIVE_TOOL_CUE_RE.finditer(clause):
                tname = m.group(1).lower()
                if tname in available:
                    positive.add(tname)

    # Never mark a tool as declined if a separate non-negated clause explicitly instructs calling it.
    declined -= positive
    pure_text = bool(declined) and not positive and bool(_PURE_TEXT_DELIVERABLE_RE.search(last_user))
    return declined, positive, pure_text


def apply_pre_semantics_guard(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> tuple[list[dict[str, Any]] | None, Any, dict[str, Any]]:
    """Prune explicitly declined tools from `tools` (without mutating `messages`!)."""
    declined, positive, pure_text = detect_declined_tools(messages, tools)
    if not declined or not tools:
        return tools, tool_choice, {"declined_tools": [], "positive_tools": sorted(positive), "pure_text_mode": False}

    if pure_text:
        return None, "none", {
            "declined_tools": sorted(declined),
            "positive_tools": [],
            "pure_text_mode": True,
            "tools_pruned": len(tools),
        }

    filtered = [
        t for t in tools
        if str(((t.get("function") or {}).get("name")) or "") not in declined
    ]
    new_choice = tool_choice if filtered else "none"
    return (filtered or None), new_choice, {
        "declined_tools": sorted(declined),
        "positive_tools": sorted(positive),
        "pure_text_mode": False,
        "tools_pruned": len(tools) - len(filtered),
    }


def _extract_fenced_or_def_python(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    m = re.search(r"(^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+).*)", cleaned, flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def check_non_answer_or_truncated_promise(
    messages: list[dict[str, Any]],
    content: str,
    tool_calls: list[dict[str, Any]],
    finish_reason: str,
) -> str | None:
    """Return a concise continuation prompt if the assistant emitted a meta-promise or truncated code block."""
    if tool_calls:
        return None
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            last_user = c if isinstance(c, str) else str(c or "")
            break
    if not last_user.strip():
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    wants_code = bool(_CODE_REQUEST_RE.search(last_user))

    if not cleaned:
        return "Your previous reply was empty. Provide the complete requested answer directly now."

    # Check unclosed markdown code fence (e.g., hit max_tokens mid-function after a conversational lead-in)
    fence_count = len(re.findall(r"```", cleaned))
    if fence_count % 2 == 1:
        return (
            "Your previous response stopped mid-code-block before closing ```. "
            "Output the COMPLETE, self-contained code block from start to finish with no prose."
        )

    if wants_code:
        code = _extract_fenced_or_def_python(cleaned)
        if not code and _PROMISE_LEADIN_RE.search(cleaned):
            return (
                "You stated an intention to write the code without including the implementation. "
                "Output the COMPLETE Python implementation inside ```python ... ``` now."
            )
        if code and finish_reason == "length":
            try:
                ast.parse(code)
            except SyntaxError:
                return (
                    "Your Python implementation was truncated before completion. "
                    "Output ONLY the complete, syntactically valid ```python ... ``` implementation."
                )
    return None
