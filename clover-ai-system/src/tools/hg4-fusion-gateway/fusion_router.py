"""Copilot Fusion (`Single / Cascade / Critique`) and Elephant-Goldfish (EGM) virtual-model router.

Direct models: `lfm3-omni-2b`, `clover-35b-a3b`, `nemotron-3.5-super-vl`, `kimi-k3` (+ alias `kimi-k3.1-pro`).
Virtual models:
  * `clover-fusion-standard` – trivial triage / formatting -> `lfm3-omni-2b`; coding / tool turns -> `clover-35b-a3b`.
  * `clover-fusion-high`     – Cascade: `clover-35b-a3b`, escalating to `nemotron-3.5-super-vl` ONLY on objective
                               failure signals (trailing `tool` output with compiler / Lean / pytest errors, or a tool
                               call that is still schema-invalid after 1 retry). Never on self-reported confidence.
  * `clover-fusion-max`      – Critique: draft on Nemotron, one bounded stateless auditor pass (<= 350 tokens) on
                               Clover-35B (non-veto, concrete flaws only), revise on Nemotron only if a flaw is flagged.
  * `clover-egm-elephant`    – Nemotron-120B for spec writing / theorem decomposition.
  * `clover-egm-goldfish`    – Clover-35B, hard `max_tokens <= 400`, structured JSON findings (non-veto reviewer).
Each response carries `x_clover_telemetry` (usage, prefill cache-hit %, latency, escalations).
"""
from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import skill_loader as sl
import token_economizer as te
import tool_grammar as tg
from backends import MODEL_ALIASES, BackendBase, GenParams, GenResult

VIRTUAL_MODELS: dict[str, str] = {
    "clover-fusion-standard": "Copilot Fusion Single: Omni-2B triage/formatting, Clover-35B coding & tools",
    "clover-fusion-high": "Copilot Fusion Cascade: Clover-35B -> Nemotron-120B on objective failure signals",
    "clover-fusion-max": "Copilot Fusion Critique: Nemotron-120B draft + Clover-35B bounded audit + revise",
    "clover-egm-elephant": "Elephant (EGM): Nemotron-120B spec writing / theorem decomposition",
    "clover-egm-goldfish": "Goldfish (EGM): Clover-35B non-veto reviewer, <= 400 tokens, structured findings",
}

GOLDFISH_MAX_TOKENS = 400
AUDITOR_MAX_TOKENS = 350
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_CAP = 16384

# Objective failure markers (compilers, Lean, pytest, cargo, tsc, shells). Deliberately conservative.
FAILURE_RE = re.compile(
    r"(?im)("
    r"\berror(?:\[[A-Z]?\d+\])?:"            # gcc/clang/rustc/lean/lake `error:` / `error[E0308]:`
    r"|unsolved goals"                        # Lean 4
    r"|Traceback \(most recent call last\)"   # Python
    r"|^FAILED\b|\bFAILED\s*\(|\s FAILED$"    # pytest / unittest
    r"|\b[1-9]\d* failed\b"                   # `2 failed, 10 passed`
    r"|^E\s{3}\w"                             # pytest assertion detail lines
    r"|\b(?:Assertion|Syntax|Type|Name|Attribute|Index|Key|Value|Import|ModuleNotFound)Error\b"
    r"|\berror TS\d+"                         # TypeScript
    r"|\bpanicked at\b"                       # Rust
    r"|compilation (?:failed|terminated)"
    r"|BUILD FAILED|\bnpm ERR!"
    r"|exit(?:ed with)? (?:code|status) [1-9]\d*"
    r")"
)

TRIVIAL_HINT_RE = re.compile(
    r"(?i)\b(format|reformat|rename|title|summari[sz]e|tl;?dr|classify|triage|label|translate|spell|grammar|"
    r"rephrase|bullet|capitali[sz]e|sort|commit message|yes or no|one word)\b"
)
CODE_HINT_RE = re.compile(
    r"(?i)(```|\bdef |\bclass |\bfn |\bimport |#include|\btheorem\b|\blemma\b|\bfix\b|\bbug\b|\brefactor\b|"
    r"\bimplement\b|\bcompile\b|\btest\b|\berror\b|Traceback|\.py\b|\.rs\b|\.ts\b|\.lean\b|\.c\b)"
)

GOLDFISH_SYSTEM = (
    "You are a Goldfish reviewer in an Elephant-Goldfish pair. You are NON-VETO: report only concrete, "
    "verifiable findings (logic errors, type errors, missed edge cases, spec violations). Be brief.\n"
    "Respond ONLY with JSON: {\"findings\": [{\"severity\": \"high|medium|low\", \"location\": \"...\", "
    "\"issue\": \"...\", \"suggestion\": \"...\"}], \"verdict\": \"non-veto\"}. Use an empty findings list "
    "when nothing concrete is wrong."
)
ELEPHANT_SYSTEM = (
    "You are the Elephant in an Elephant-Goldfish pair: own the long-horizon context, write precise specs, "
    "decompose theorems/tasks into verifiable sub-goals, and integrate reviewer findings."
)
AUDITOR_SYSTEM = (
    "CLOVER_AUDITOR: You are a strict, stateless code and reasoning auditor. Inspect the DRAFT for concrete "
    "logic, type, off-by-one and edge-case flaws only. If you find one, reply with lines starting with "
    "'FLAW:' naming the exact location and fix. If nothing concrete is wrong reply exactly 'NO_ISSUES'. "
    "Do not restate the draft. Do not comment on style."
)


def canonical_model(model: str) -> str:
    m = (model or "").strip()
    if "/" in m:
        m = m.split("/", 1)[1]
    return MODEL_ALIASES.get(m, m)


def parse_params(body: dict[str, Any], cap: int | None = None) -> GenParams:
    mt = body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
    try:
        mt = int(mt)
    except (TypeError, ValueError):
        mt = DEFAULT_MAX_TOKENS
    mt = max(1, min(mt, MAX_TOKENS_CAP))
    if cap is not None:
        mt = min(mt, cap)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = []
    temp = body.get("temperature")
    top_p = body.get("top_p")
    return GenParams(
        max_tokens=mt,
        temperature=float(temp) if isinstance(temp, (int, float)) else 0.0,
        top_p=float(top_p) if isinstance(top_p, (int, float)) else 1.0,
        stop=[str(s) for s in stop if s][:4],
        seed=int(body.get("seed") or 0),
    )


def trailing_tool_outputs(messages: list[dict[str, Any]]) -> list[str]:
    """Contents of the `tool` messages at the end of the conversation (results of the latest tool calls)."""
    out: list[str] = []
    for m in reversed(messages or []):
        if m.get("role") in ("tool", "function"):
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
            out.append(str(c or ""))
        else:
            break
    return list(reversed(out))


def objective_failure(messages: list[dict[str, Any]]) -> str | None:
    """Return the matched failure marker when the latest tool output shows an objective failure."""
    for txt in trailing_tool_outputs(messages):
        m = FAILURE_RE.search(txt)
        if m:
            return m.group(0).strip()
    return None


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def classify_standard(messages: list[dict[str, Any]], tools: Any, tool_choice: Any) -> tuple[str, str]:
    """Route for `clover-fusion-standard` -> (model_id, reason)."""
    if tool_choice not in (None, "none", "auto") and tools:
        return "clover-35b-a3b", "tool_choice forces a tool call"
    if trailing_tool_outputs(messages):
        return "clover-35b-a3b", "tool-result turn"
    user = _last_user_text(messages)
    total = sum(len(str(m.get("content") or "")) for m in messages or [])
    if CODE_HINT_RE.search(user):
        return "clover-35b-a3b", "coding turn"
    if len(user) <= 400 and total <= 6000 and (TRIVIAL_HINT_RE.search(user) or not tools):
        return "lfm3-omni-2b", "trivial triage/formatting"
    return "clover-35b-a3b", "standard turn"


# --------------------------------------------------------------------------------------------------------------------
# Streaming content filter (holds back possible tool-call markup)
# --------------------------------------------------------------------------------------------------------------------

_MARKUP_STARTS = ("<tool_call>", "<function=", "<|python_tag|>", "[TOOL_CALLS]", "```json", "```tool")


class ContentStreamer:
    def __init__(self, emit: Callable[[str], None] | None, tools_active: bool):
        self.emit = emit
        self.tools_active = tools_active
        self.buf = ""
        self.sent = 0
        self.blocked = False
        self.enabled = emit is not None

    def feed(self, delta: str) -> None:
        if not self.enabled or not delta:
            return
        self.buf += delta
        if not self.tools_active:
            self._send(len(self.buf))
            return
        if self.blocked:
            return
        idx = min((i for i in (self.buf.find(m) for m in _MARKUP_STARTS) if i >= 0), default=-1)
        if idx < 0 and self.buf.lstrip().startswith("{"):
            idx = len(self.buf) - len(self.buf.lstrip())
        if idx >= 0:
            self.blocked = True
            self._send(idx)
            return
        safe = len(self.buf)
        for m in _MARKUP_STARTS:
            for k in range(min(len(m) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(m[:k]):
                    safe = min(safe, len(self.buf) - k)
                    break
        if self.buf.strip() == "":
            safe = self.sent
        self._send(safe)

    def _send(self, upto: int) -> None:
        # never emit trailing whitespace until more text follows (final content is stripped)
        upto = min(upto, len(self.buf[:upto].rstrip()))
        if upto > self.sent:
            piece = self.buf[self.sent:upto]
            if self.sent == 0:
                piece = piece.lstrip()
            if piece:
                self.emit(piece)
            self.sent = upto

    def finish(self, final_content: str) -> None:
        """Flush whatever part of the final (markup-stripped) content was not streamed yet."""
        if not self.enabled:
            return
        sent_text = self.buf[:self.sent].strip()
        fc = final_content or ""
        if not sent_text:
            if fc:
                self.emit(fc)
        elif fc.startswith(sent_text) and len(fc) > len(sent_text):
            self.emit(fc[len(sent_text):])
        self.enabled = False


# --------------------------------------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------------------------------------

class Telemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.c: collections.Counter = collections.Counter()
        self.per_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.escalation_reasons: collections.Counter = collections.Counter()
        self.started = time.time()

    def record(self, virtual: str, t: dict[str, Any]) -> None:
        with self._lock:
            self.c["requests"] += 1
            self.c["escalations"] += t.get("escalations", 0)
            self.c["tool_call_repairs"] += t.get("tool_call_repairs", 0)
            self.c["tool_call_retries"] += t.get("tool_call_retries", 0)
            self.c["audits"] += 1 if t.get("audit") else 0
            self.c["audit_flaws"] += 1 if (t.get("audit") or {}).get("flaw") else 0
            self.c["wall_ms"] += t.get("wall_ms", 0.0)
            if t.get("escalation_reason"):
                self.escalation_reasons[t["escalation_reason"]] += 1
            vm = self.per_model[virtual]
            vm["requests"] += 1
            vm["escalations"] += t.get("escalations", 0)
            for st in t.get("steps", []):
                pm = self.per_model[st["model"]]
                for k in ("prompt_tokens", "cached_tokens", "prefill_tokens", "completion_tokens"):
                    pm[k] += st[k]
                    self.c[k] += st[k]
                pm["prefill_ms"] += st["prefill_ms"]
                pm["decode_ms"] += st["decode_ms"]
                pm["calls"] += 1
                self.c["prefill_ms"] += st["prefill_ms"]
                self.c["decode_ms"] += st["decode_ms"]

    @staticmethod
    def _derive(c: collections.Counter) -> dict[str, Any]:
        d = dict(c)
        pt = c.get("prompt_tokens", 0)
        d["prefill_cache_hit_pct"] = round(100.0 * c.get("cached_tokens", 0) / pt, 2) if pt else 0.0
        dm = c.get("decode_ms", 0.0)
        d["decode_tokens_per_sec"] = round(1000.0 * c.get("completion_tokens", 0) / dm, 2) if dm > 0 else 0.0
        pm = c.get("prefill_ms", 0.0)
        d["prefill_tokens_per_sec"] = round(1000.0 * c.get("prefill_tokens", 0) / pm, 2) if pm > 0 else 0.0
        for k in ("prefill_ms", "decode_ms", "wall_ms"):
            if k in d:
                d[k] = round(d[k], 2)
        return d

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tot = self._derive(self.c)
            req = self.c.get("requests", 0)
            tot["escalation_rate_pct"] = round(100.0 * self.c.get("escalations", 0) / req, 2) if req else 0.0
            return {
                "object": "clover.telemetry",
                "uptime_s": round(time.time() - self.started, 1),
                "totals": tot,
                "cascade": {"escalations": self.c.get("escalations", 0),
                            "reasons": dict(self.escalation_reasons)},
                "models": {k: self._derive(v) for k, v in sorted(self.per_model.items())},
            }


# --------------------------------------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class StepOutcome:
    model: str
    role: str
    result: GenResult
    parsed: tg.ParseResult
    retried: bool = False


@dataclass
class Completion:
    model: str
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    usage: dict[str, Any]
    telemetry: dict[str, Any] = field(default_factory=dict)


class RouterError(Exception):
    def __init__(self, status: int, message: str, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class FusionRouter:
    def __init__(self, backends: dict[str, BackendBase], telemetry: Telemetry | None = None):
        self.backends = backends
        self.telemetry = telemetry or Telemetry()

    def reset_caches(self) -> dict[str, Any]:
        reset_models: list[str] = []
        for mid, be in self.backends.items():
            if hasattr(be, "reset_cache") and callable(be.reset_cache):
                be.reset_cache()
                reset_models.append(mid)
        return {"reset": reset_models, "count": len(reset_models)}

    # -- discovery ---------------------------------------------------------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        now = int(time.time())
        out = []
        for mid, be in self.backends.items():
            out.append({"id": mid, "object": "model", "created": now, "owned_by": "clover-hg4",
                        "x_clover": {"kind": be.kind, "family": be.family, "virtual": False}})
        if "kimi-k3" in self.backends:
            out.append({"id": "kimi-k3.1-pro", "object": "model", "created": now, "owned_by": "clover-hg4",
                        "x_clover": {"kind": self.backends["kimi-k3"].kind, "alias_of": "kimi-k3", "virtual": False}})
        for vid, desc in VIRTUAL_MODELS.items():
            if all(m in self.backends for m in self._virtual_deps(vid)):
                out.append({"id": vid, "object": "model", "created": now, "owned_by": "clover-fusion",
                            "x_clover": {"virtual": True, "description": desc, "uses": self._virtual_deps(vid)}})
        return out

    @staticmethod
    def _virtual_deps(vid: str) -> list[str]:
        return {
            "clover-fusion-standard": ["lfm3-omni-2b", "clover-35b-a3b"],
            "clover-fusion-high": ["clover-35b-a3b", "nemotron-3.5-super-vl"],
            "clover-fusion-max": ["nemotron-3.5-super-vl", "clover-35b-a3b"],
            "clover-egm-elephant": ["nemotron-3.5-super-vl"],
            "clover-egm-goldfish": ["clover-35b-a3b"],
        }[vid]

    def _backend(self, mid: str) -> BackendBase:
        be = self.backends.get(mid)
        if be is None:
            raise RouterError(503, f"model backend '{mid}' is not available", "model_not_available")
        return be

    # -- single model step with tool-call validation + 1 retry -------------------------------------------------------
    def _run(self, mid: str, role: str, messages, tools, tool_choice, params: GenParams,
             streamer: ContentStreamer | None = None, retry_invalid: bool = True) -> StepOutcome:
        be = self._backend(mid)
        tools_active = bool(tg.active_tools(tools, tool_choice))
        res = None
        for ev, payload in be.stream(messages, tools, tool_choice, params):
            if ev == "delta":
                if streamer is not None:
                    streamer.feed(payload)
            else:
                res = payload
        assert res is not None
        parsed = tg.parse_tool_calls(res.text, tools, tool_choice) if tools_active else \
            tg.ParseResult(content=res.text.strip(), tool_calls=[], errors=[], had_markup=False)
        outcome = StepOutcome(mid, role, res, parsed)
        need_call = tools_active and (tool_choice == "required" or isinstance(tool_choice, dict))
        invalid = tools_active and ((parsed.had_markup and not parsed.all_valid) or (need_call and not parsed.valid_calls))
        if invalid and retry_invalid:
            problems = parsed.problems or ["no tool call was emitted although one is required"]
            fix_msgs = list(messages) + [
                {"role": "assistant", "content": res.text},
                {"role": "user", "content": "Your previous tool call was invalid: " + "; ".join(problems)
                 + ". Re-emit ONLY the corrected tool call(s) with every required parameter, as strict JSON."},
            ]
            retry = self._run(mid, role, fix_msgs, tools, tool_choice, params, streamer=None, retry_invalid=False)
            retry.retried = True
            retry.result.extra["first_attempt"] = {"prompt_tokens": res.prompt_tokens,
                                                   "cached_tokens": res.cached_tokens,
                                                   "prefill_tokens": res.prefill_tokens,
                                                   "completion_tokens": res.completion_tokens,
                                                   "prefill_ms": res.prefill_ms, "decode_ms": res.decode_ms}
            if not retry.parsed.content and parsed.content:
                retry.parsed.content = parsed.content
            return retry
        return outcome

    @staticmethod
    def _invalid(o: StepOutcome) -> bool:
        p = o.parsed
        return p.had_markup and not p.all_valid and not p.valid_calls

    # -- orchestration -----------------------------------------------------------------------------------------------
    def complete(self, body: dict[str, Any], on_delta: Callable[[str], None] | None = None) -> Completion:
        t0 = time.perf_counter()
        requested = str(body.get("model") or "")
        model = canonical_model(requested)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RouterError(400, "'messages' must be a non-empty array")
        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        tool_choice = body.get("tool_choice")
        telem: dict[str, Any] = {"virtual_model": model if model in VIRTUAL_MODELS else None, "escalations": 0}

        # Shipped production defaults: TokenEconomizer, OutputBudgeter, and SKILL.md index are ON by default
        # unless explicitly disabled via request body (`x_clover_* = false`) or env (`CLOVER_ENABLE_* = 0`).
        if "x_clover_economizer" in body:
            use_econ = bool(body["x_clover_economizer"])
        else:
            use_econ = os.environ.get("CLOVER_ENABLE_ECONOMIZER", "1") != "0"
        if use_econ:
            messages, tools, econ_meta = te.apply_economizer(model, messages, tools)
            telem["economizer"] = econ_meta

        if "x_clover_output_budget" in body:
            use_budget = bool(body["x_clover_output_budget"])
        else:
            use_budget = os.environ.get("CLOVER_ENABLE_OUTPUT_BUDGET", "1") != "0"
        if use_budget:
            messages, trimmed_n = te.budget_tool_outputs(messages)
            telem["output_budget_trimmed"] = trimmed_n

        if "x_clover_skills_index" in body:
            use_skills_idx = bool(body["x_clover_skills_index"])
        else:
            use_skills_idx = os.environ.get("CLOVER_ENABLE_SKILLS_INDEX", "1") != "0"

        if use_skills_idx:
            default_skills_dirs = [
                Path(__file__).resolve().parents[3] / "skills" if len(Path(__file__).resolve().parents) > 3 else Path("/nonexistent"),
                Path(__file__).resolve().parents[1] / "skills",
                Path("/opt/clover-k3/skills"),
            ]
            skills_root = body.get("x_clover_skills_dir") or os.environ.get("CLOVER_SKILLS_DIR") or next(
                (str(p) for p in default_skills_dirs if p.is_dir()), None
            )
            if skills_root:
                entries = sl.discover_skills(Path(str(skills_root)))
                idx_block = sl.format_compact_skill_index(entries)
                if idx_block:
                    messages = self._with_system(messages, idx_block)
                    telem["skills_indexed"] = [e.name for e in entries]

        tools_active = bool(tg.active_tools(tools, tool_choice))
        streamer = ContentStreamer(on_delta, tools_active) if on_delta else None
        steps: list[StepOutcome] = []
        final: StepOutcome

        if model in self.backends:
            final = self._run(model, "direct", messages, tools, tool_choice, parse_params(body), streamer)
            steps.append(final)
        elif model == "clover-fusion-standard":
            target, reason = classify_standard(messages, tools, tool_choice)
            telem["route_reason"] = reason
            final = self._run(target, "single", messages, tools, tool_choice, parse_params(body), streamer)
            steps.append(final)
        elif model == "clover-fusion-high":
            final = self._cascade(body, messages, tools, tool_choice, streamer, steps, telem)
        elif model == "clover-fusion-max":
            final = self._critique(body, messages, tools, tool_choice, steps, telem)
        elif model == "clover-egm-elephant":
            msgs = self._with_system(messages, ELEPHANT_SYSTEM)
            final = self._run("nemotron-3.5-super-vl", "elephant", msgs, tools, tool_choice, parse_params(body),
                              streamer)
            steps.append(final)
        elif model == "clover-egm-goldfish":
            final = self._goldfish(body, messages, tools, tool_choice, streamer, steps, telem)
        else:
            raise RouterError(404, f"model '{requested}' not found", "model_not_found")

        content = final.parsed.content
        tool_calls = final.parsed.openai_tool_calls() if tools_active else []
        if final.parsed.tool_calls and not tool_calls:
            telem["dropped_invalid_tool_calls"] = final.parsed.problems
        if final.parsed.errors and not tool_calls:
            telem["tool_call_errors"] = final.parsed.errors
        if final.result.finish_reason == "length" and not tool_calls:
            finish = "length"
        else:
            finish = "tool_calls" if tool_calls else "stop"

        if final.result.gen_ids and not final.retried and final.role not in ("revise", "goldfish-structured"):
            self.backends[final.model].remember_assistant(content, tool_calls, final.result.gen_ids)

        if streamer is not None:
            streamer.finish(content)

        step_rows = []
        for s in steps:
            for r in ([s.result.extra["first_attempt"]] if "first_attempt" in s.result.extra else []) + [s.result]:
                g = r if isinstance(r, dict) else {
                    "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
                    "prefill_tokens": r.prefill_tokens, "completion_tokens": r.completion_tokens,
                    "prefill_ms": r.prefill_ms, "decode_ms": r.decode_ms}
                step_rows.append({
                    "model": s.model, "role": s.role, **{k: g[k] for k in ("prompt_tokens", "cached_tokens",
                                                                          "prefill_tokens", "completion_tokens")},
                    "prefill_ms": round(g["prefill_ms"], 2), "decode_ms": round(g["decode_ms"], 2),
                    "cache_hit_pct": round(100.0 * g["cached_tokens"] / g["prompt_tokens"], 2)
                    if g["prompt_tokens"] else 0.0,
                    "backend_kind": self.backends[s.model].kind,
                })
        pt = sum(r["prompt_tokens"] for r in step_rows)
        ct = sum(r["completion_tokens"] for r in step_rows)
        cached = sum(r["cached_tokens"] for r in step_rows)
        dec_ms = sum(r["decode_ms"] for r in step_rows)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        telem.update({
            "requested_model": requested,
            "served_by": final.model,
            "steps": step_rows,
            "tool_call_repairs": sum(1 for c in final.parsed.tool_calls if c.repaired),
            "tool_call_retries": sum(1 for s in steps if s.retried),
            "prefill_cache_hit_pct": round(100.0 * cached / pt, 2) if pt else 0.0,
            "prefill_tokens": sum(r["prefill_tokens"] for r in step_rows),
            "wall_ms": round(wall_ms, 2),
            "decode_tokens_per_sec": round(1000.0 * ct / dec_ms, 2) if dec_ms > 0 else 0.0,
        })
        self.telemetry.record(model, telem)
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        return Completion(model=requested or model, content=content, tool_calls=tool_calls, finish_reason=finish,
                          usage=usage, telemetry=telem)

    @staticmethod
    def _with_system(messages: list[dict[str, Any]], extra: str) -> list[dict[str, Any]]:
        msgs = list(messages)
        n = 0
        while n < len(msgs) and msgs[n].get("role") in ("system", "developer"):
            n += 1
        return msgs[:n] + [{"role": "system", "content": extra}] + msgs[n:]

    def _cascade(self, body, messages, tools, tool_choice, streamer, steps, telem) -> StepOutcome:
        params = parse_params(body)
        marker = objective_failure(messages)
        if marker:
            telem["escalations"] = 1
            telem["escalation_reason"] = "tool_failure"
            telem["escalation_marker"] = marker
            final = self._run("nemotron-3.5-super-vl", "cascade-escalated", messages, tools, tool_choice, params,
                              streamer)
            steps.append(final)
            return final
        first = self._run("clover-35b-a3b", "cascade-primary", messages, tools, tool_choice, params,
                          None if bool(tg.active_tools(tools, tool_choice)) else streamer)
        steps.append(first)
        if first.retried and self._invalid(first):
            telem["escalations"] = 1
            telem["escalation_reason"] = "invalid_tool_call_after_retry"
            final = self._run("nemotron-3.5-super-vl", "cascade-escalated", messages, tools, tool_choice, params,
                              streamer)
            steps.append(final)
            return final
        if streamer is not None and tg.active_tools(tools, tool_choice):
            streamer.feed(first.parsed.content)
        return first

    def _critique(self, body, messages, tools, tool_choice, steps, telem) -> StepOutcome:
        params = parse_params(body)
        draft = self._run("nemotron-3.5-super-vl", "draft", messages, tools, tool_choice, params)
        steps.append(draft)
        if draft.parsed.valid_calls:
            telem["audit"] = {"skipped": "draft is a tool call"}
            return draft
        task = _last_user_text(messages)[-6000:]
        audit_msgs = [
            {"role": "system", "content": AUDITOR_SYSTEM},
            {"role": "user", "content": f"TASK:\n{task}\n\nDRAFT:\n{draft.parsed.content[-12000:]}"},
        ]
        audit = self._run("clover-35b-a3b", "auditor", audit_msgs, None, None,
                          GenParams(max_tokens=AUDITOR_MAX_TOKENS, temperature=0.0))
        steps.append(audit)
        verdict = audit.parsed.content.strip()
        flaws = [ln.strip() for ln in verdict.splitlines() if ln.strip().upper().startswith("FLAW:")]
        telem["audit"] = {"flaw": bool(flaws), "findings": flaws[:8], "tokens": audit.result.completion_tokens}
        if not flaws:
            return draft
        revise_msgs = list(messages) + [
            {"role": "assistant", "content": draft.parsed.content},
            {"role": "user", "content": "CLOVER_REVISE: An independent auditor flagged these concrete issues:\n"
             + "\n".join(flaws) + "\nIf they are valid, produce the corrected complete final answer; "
             "otherwise restate your answer unchanged."},
        ]
        revised = self._run("nemotron-3.5-super-vl", "revise", revise_msgs, tools, tool_choice, params)
        steps.append(revised)
        return revised

    def _goldfish(self, body, messages, tools, tool_choice, streamer, steps, telem) -> StepOutcome:
        params = parse_params(body, cap=GOLDFISH_MAX_TOKENS)
        telem["max_tokens_cap"] = GOLDFISH_MAX_TOKENS
        msgs = self._with_system(messages, GOLDFISH_SYSTEM)
        out = self._run("clover-35b-a3b", "goldfish", msgs, tools, tool_choice, params, None)
        steps.append(out)
        if not out.parsed.valid_calls:
            structured = _structure_findings(out.parsed.content)
            if structured != out.parsed.content:
                out.role = "goldfish-structured"
            out.parsed.content = structured
        if streamer is not None:
            streamer.tools_active = bool(out.parsed.valid_calls)
            streamer.feed(out.parsed.content if not out.parsed.valid_calls else "")
        return out


def _structure_findings(text: str) -> str:
    """Enforce the Goldfish structured JSON findings contract (non-veto) while preserving explicit PROVABLE/UNPROVABLE verdicts."""
    t = (text or "").strip()
    up_head = (t.splitlines()[0].upper() if t else "")
    explicit_subgoal = "UNPROVABLE" if "UNPROVABLE" in up_head else ("PROVABLE" if "VERDICT: PROVABLE" in up_head else None)
    try:
        obj = tg.repair_json(t) if ("{" in t or "[" in t) else None
    except tg.JsonRepairError:
        obj = None
    if isinstance(obj, list):
        obj = {"findings": obj}
    if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
        norm = []
        for f in obj["findings"]:
            if isinstance(f, dict):
                norm.append({"severity": str(f.get("severity") or "medium"), "location": str(f.get("location") or ""),
                             "issue": str(f.get("issue") or f.get("description") or ""),
                             "suggestion": str(f.get("suggestion") or f.get("fix") or "")})
            elif str(f).strip():
                norm.append({"severity": "medium", "location": "", "issue": str(f).strip(), "suggestion": ""})
        out_d: dict[str, Any] = {"findings": norm, "verdict": "non-veto"}
        if obj.get("subgoal_verdict") in ("PROVABLE", "UNPROVABLE"):
            out_d["subgoal_verdict"] = obj["subgoal_verdict"]
        elif explicit_subgoal:
            out_d["subgoal_verdict"] = explicit_subgoal
        return json.dumps(out_d, ensure_ascii=False)
    lines = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip() for ln in t.splitlines()]
    if explicit_subgoal == "PROVABLE":
        return json.dumps({"findings": [], "verdict": "non-veto", "subgoal_verdict": "PROVABLE"}, ensure_ascii=False)
    sev_default = "high" if explicit_subgoal == "UNPROVABLE" else "medium"
    issues = [
        ln for ln in lines
        if ln and not ln.lower().startswith(("no issues", "no_issues", "lgtm", "verdict: provable"))
    ]
    findings = [{"severity": sev_default, "location": "", "issue": ln[:500], "suggestion": ""} for ln in issues[:12]]
    out_plain: dict[str, Any] = {"findings": findings, "verdict": "non-veto"}
    if explicit_subgoal:
        out_plain["subgoal_verdict"] = explicit_subgoal
    return json.dumps(out_plain, ensure_ascii=False)
