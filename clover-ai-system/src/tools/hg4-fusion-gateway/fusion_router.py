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
import studio_epistemic_memory as sem_mem
import studio_jepa_router as sjepa
import studio_ontology_verifier as sonto
import studio_outcome_verifier as sout
import studio_semantics_guard as ssem
import studio_xcruncher_verifier as sxc
import token_economizer as te
import tool_grammar as tg
from backends import MODEL_ALIASES, BackendBase, GenParams, GenResult

VIRTUAL_MODELS: dict[str, str] = {
    "clover-fusion-standard": "Copilot Fusion Single: Omni-2B triage/formatting, Clover-35B coding & tools",
    "clover-fusion-high": "Copilot Fusion Cascade: Clover-35B -> Nemotron-120B on objective failure signals",
    "clover-fusion-cluster": "3-Layer Multi-Server Cluster: Deterministic Specialist Gate + 35B -> 120B -> Kimi-2.8T Cascade",
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

# Layer-1 Deterministic Specialist Input-Type Patterns (0 ms routing, zero LLM guesswork)
LEAN4_SPECIALIST_RE = re.compile(
    r"(?m)(\bimport\s+Mathlib\b|\b(?:theorem|lemma)\s+\w+.*?:=\s*by\b|```lean\b|\blake\s+(?:env|build)\b|\.lean\b)"
)
BIO_FASTA_RE = re.compile(
    r"(?m)(^>[A-Za-z0-9_|.\-]{3,}.*\n[ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy\s]{30,}"
    r"|\b[ACDEFGHIKLMNPQRSTVWY]{45,}\b|\b(?:boltz|fold\s+protein|binding\s+affinity|ligand\s+docking|\.pdb\b|\.cif\b))"
)
VIDEO_PHYSICS_RE = re.compile(
    r"(?i)(\.(?:mp4|webm|mov|avi)\b|\binput_video\b|\bvideo\s+(?:clip|frames|trajectory)\b)"
    r".*(\b(?:predict|collision|trajectory|fall|spill|motion|next\s+frame|what\s+happens\s+if|physics)\b)"
    r"|(\b(?:predict|collision|trajectory|motion|what\s+happens\s+when)\b).*(\.(?:mp4|webm|mov|avi)\b|\binput_video\b)"
)
TIMESERIES_RE = re.compile(
    r"(?is)\b(?:forecast|predict\s+next\s+\d+|time[\s\-]?series|horizon|quantile\s+forecast)\b"
    r".*(?:\d{4}-\d{2}-\d{2}[T\s,]\s*-?\d+(?:\.\d+)?|(?:[-+]?\d+(?:\.\d+)?\s*,\s*){8,}[-+]?\d+(?:\.\d+)?)"
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


def consecutive_objective_failures(messages: list[dict[str, Any]]) -> tuple[int, str | None]:
    """Count consecutive trailing `[assistant, tool(error)+]` rounds at the end of `messages`."""
    msgs = list(messages or [])
    idx = len(msgs) - 1
    rounds = 0
    latest_marker: str | None = None
    while idx >= 0 and msgs[idx].get("role") in ("tool", "function"):
        round_marker: str | None = None
        while idx >= 0 and msgs[idx].get("role") in ("tool", "function"):
            c = msgs[idx].get("content")
            if isinstance(c, list):
                c = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
            m = FAILURE_RE.search(str(c or ""))
            if m and round_marker is None:
                round_marker = m.group(0).strip()
            idx -= 1
        if round_marker is None:
            break
        if latest_marker is None:
            latest_marker = round_marker
        rounds += 1
        if idx >= 0 and msgs[idx].get("role") == "assistant":
            idx -= 1
        else:
            break
    return rounds, latest_marker


def classify_specialist_domain(
    messages: list[dict[str, Any]],
    available_backends: set[str] | dict[str, Any],
) -> tuple[str | None, str | None]:
    """Layer-1 deterministic input-type specialist gate (0 ms, never guesses across domains)."""
    user = _last_user_text(messages)
    raw_last = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            raw_last = json.dumps(m.get("content"), ensure_ascii=False) if not isinstance(m.get("content"), str) else m["content"]
            break
    if "goedel-prover-v2-32b" in available_backends and LEAN4_SPECIALIST_RE.search(user):
        return "goedel-prover-v2-32b", "specialist:lean4_formal_proof"
    if "boltz-2" in available_backends and BIO_FASTA_RE.search(user):
        return "boltz-2", "specialist:biomolecular_structure"
    if "vjepa-2.1" in available_backends and (
        VIDEO_PHYSICS_RE.search(raw_last) or ("video_url" in raw_last or "input_video" in raw_last)
    ):
        return "vjepa-2.1", "specialist:physical_world_video"
    if "chronos-2" in available_backends and TIMESERIES_RE.search(user):
        return "chronos-2", "specialist:timeseries_forecast"
    return None, None


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
        self.epistemic_memory = sem_mem.EpistemicMemoryStore()

    def reset_caches(self) -> dict[str, Any]:
        reset_models: list[str] = []
        for mid, be in self.backends.items():
            if hasattr(be, "reset_cache") and callable(be.reset_cache):
                be.reset_cache()
                reset_models.append(mid)
        self.epistemic_memory.reset()
        return {"reset": reset_models, "count": len(reset_models), "epistemic_memory_reset": True}

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
            "clover-fusion-cluster": ["lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"],
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

        # Security: filesystem paths (`skills_dir`, `s2b_lean_project`) and host subprocess execution (`s2_mode`)
        # are controlled strictly by server-side environment variables unless local harness explicitly opts in via
        # `CLOVER_ALLOW_UNSAFE_REQUEST_OVERRIDES=1`.
        allow_unsafe_req = os.environ.get("CLOVER_ALLOW_UNSAFE_REQUEST_OVERRIDES", "0") == "1"

        if use_skills_idx:
            default_skills_dirs = [
                Path(__file__).resolve().parents[3] / "skills" if len(Path(__file__).resolve().parents) > 3 else Path("/nonexistent"),
                Path(__file__).resolve().parents[1] / "skills",
                Path("/opt/clover-k3/skills"),
            ]
            req_skills_dir = body.get("x_clover_skills_dir") if allow_unsafe_req else None
            skills_root = req_skills_dir or os.environ.get("CLOVER_SKILLS_DIR") or next(
                (str(p) for p in default_skills_dirs if p.is_dir()), None
            )
            if skills_root:
                entries = sl.discover_skills(Path(str(skills_root)))
                idx_block = sl.format_compact_skill_index(entries)
                if idx_block:
                    messages = self._with_system(messages, idx_block)
                    telem["skills_indexed"] = [e.name for e in entries]

        tools_active = bool(tg.active_tools(tools, tool_choice))
        studio_telem: dict[str, Any] = {}

        # -- Clover Studio Capability Extensions (Phases S1 -> S6, independently gated) -------------------------------
        use_s1 = (
            bool(body["x_clover_studio_s1"])
            if "x_clover_studio_s1" in body
            else os.environ.get("CLOVER_STUDIO_S1_SEMANTICS", "0") != "0"
        )
        s2_mode = str(
            (body.get("x_clover_studio_s2_mode") if allow_unsafe_req else None)
            or os.environ.get("CLOVER_STUDIO_S2_MODE", "off")
        ).strip().lower()
        if s2_mode not in ("off", "blind", "fast", "lean_elaborate"):
            s2_mode = "off"
        s2_lean_proj = str(
            (body.get("x_clover_studio_s2b_lean_project") if allow_unsafe_req else None)
            or os.environ.get("CLOVER_STUDIO_S2B_LEAN_PROJECT", "")
        ).strip()
        # S3 (`x-cruncher` exact math & BBP Pi verifier) is the sole Studio capability that cleared
        # the pre-registered AGENTS.md Keep Gate (`0/8 -> 8/8 = +100.0pp` on P3, `0` RH overhead) -> ON by default.
        # S1, S2, S4, S5, S6 remain available as explicit opt-in extensions (`default OFF`).
        use_s3 = (
            bool(body["x_clover_studio_s3"])
            if "x_clover_studio_s3" in body
            else os.environ.get("CLOVER_STUDIO_S3_XCRUNCHER", "1") != "0"
        )
        use_s4 = (
            bool(body["x_clover_studio_s4"])
            if "x_clover_studio_s4" in body
            else os.environ.get("CLOVER_STUDIO_S4_ONTOLOGY", "0") != "0"
        )
        use_s5 = (
            bool(body["x_clover_studio_s5"])
            if "x_clover_studio_s5" in body
            else os.environ.get("CLOVER_STUDIO_S5_MEMORY", "0") != "0"
        )
        use_s6 = (
            bool(body["x_clover_studio_s6"])
            if "x_clover_studio_s6" in body
            else os.environ.get("CLOVER_STUDIO_S6_JEPA", "0") != "0"
        )

        declined_set: set[str] = set()
        if use_s1:
            tools, tool_choice, s1_pre = ssem.apply_pre_semantics_guard(messages, tools, tool_choice)
            declined_set = set(s1_pre.get("declined_tools") or [])
            studio_telem["s1_semantics"] = s1_pre
            tools_active = bool(tg.active_tools(tools, tool_choice))

        if use_s5:
            messages, s5_meta = self.epistemic_memory.build_ledger_for_messages(messages)
            studio_telem["s5_epistemic_memory"] = s5_meta

        s3_applicable = False
        if use_s3:
            messages, s3_pre = sxc.apply_pre_xcruncher_context(messages)
            s3_applicable = bool(s3_pre.get("latest_has_math"))
            if s3_pre.get("pre_injected"):
                studio_telem["s3_xcruncher_pre"] = s3_pre

        req_params = parse_params(body)
        if use_s6:
            adj_max_tok, s6_meta = sjepa.evaluate_jepa_rpre(messages, tools, req_params.max_tokens)
            req_params.max_tokens = adj_max_tok
            body = {**body, "max_tokens": adj_max_tok}
            studio_telem["s6_jepa_rpre"] = s6_meta

        last_user_txt = ""
        for _m in reversed(messages):
            if _m.get("role") == "user":
                _c = _m.get("content")
                last_user_txt = _c if isinstance(_c, str) else str(_c or "")
                break
        s4_applicable = bool(use_s4 and last_user_txt and sonto.evaluate_ontology_closure(last_user_txt) is not None)

        model_ready = (
            model in self.backends
            or (model in VIRTUAL_MODELS and all(d in self.backends for d in self._virtual_deps(model)))
        )
        post_verifier_enabled = bool(use_s1 or s2_mode != "off" or s3_applicable or s4_applicable)
        if on_delta is not None and post_verifier_enabled and model_ready:
            on_delta("")
        streamer = ContentStreamer(on_delta, tools_active) if on_delta else None
        live_streamer = None if post_verifier_enabled else streamer
        steps: list[StepOutcome] = []
        final: StepOutcome

        if model in self.backends:
            final = self._run(model, "direct", messages, tools, tool_choice, req_params, live_streamer)
            steps.append(final)
        elif model == "clover-fusion-standard":
            target, reason = classify_standard(messages, tools, tool_choice)
            telem["route_reason"] = reason
            final = self._run(target, "single", messages, tools, tool_choice, req_params, live_streamer)
            steps.append(final)
        elif model == "clover-fusion-high":
            final = self._cascade(body, messages, tools, tool_choice, live_streamer, steps, telem)
        elif model == "clover-fusion-cluster":
            final = self._cluster_cascade(body, messages, tools, tool_choice, live_streamer, steps, telem)
        elif model == "clover-fusion-max":
            final = self._critique(body, messages, tools, tool_choice, steps, telem)
        elif model == "clover-egm-elephant":
            msgs = self._with_system(messages, ELEPHANT_SYSTEM)
            final = self._run("nemotron-3.5-super-vl", "elephant", msgs, tools, tool_choice, req_params,
                              live_streamer)
            steps.append(final)
        elif model == "clover-egm-goldfish":
            final = self._goldfish(body, messages, tools, tool_choice, live_streamer, steps, telem)
        else:
            raise RouterError(404, f"model '{requested}' not found", "model_not_found")

        # -- Post-generation Studio verifiers (1-turn LCP self-repair ONLY on objective finding) ----------------------
        if post_verifier_enabled and model != "clover-egm-goldfish":
            final, messages = self._apply_studio_post_verifiers(
                final=final,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                tools_active=tools_active,
                params=req_params,
                steps=steps,
                studio_telem=studio_telem,
                use_s1=use_s1,
                declined_set=declined_set,
                s2_mode=s2_mode,
                s2_lean_proj=s2_lean_proj,
                use_s3=s3_applicable,
                use_s4=s4_applicable,
            )
            tools_active = bool(tg.active_tools(tools, tool_choice))

        if studio_telem:
            telem["studio_extensions"] = studio_telem

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

        if (
            final.result.gen_ids
            and not final.retried
            and final.role not in ("revise", "goldfish-structured")
            and not final.role.endswith("-repair")
        ):
            self.backends[final.model].remember_assistant(content, tool_calls, final.result.gen_ids, messages)

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

    def _apply_studio_post_verifiers(
        self,
        *,
        final: StepOutcome,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        tools_active: bool,
        params: GenParams,
        steps: list[StepOutcome],
        studio_telem: dict[str, Any],
        use_s1: bool,
        declined_set: set[str],
        s2_mode: str,
        s2_lean_proj: str,
        use_s3: bool,
        use_s4: bool,
    ) -> tuple[StepOutcome, list[dict[str, Any]]]:
        """Run enabled Studio post-generation verifiers (`S1`, `S2`, `S3`, `S4`) with LCP cache preservation."""

        def _repair_once(
            role_tag: str,
            repair_prompt: str,
            repair_max_tokens: int | None = None,
            disable_tools: bool = False,
            cap_max_tokens: bool = False,
        ) -> tuple[StepOutcome, list[dict[str, Any]]]:
            cur_content = final.parsed.content
            cur_tcs = final.parsed.openai_tool_calls() if tools_active else []
            if (
                final.result.gen_ids
                and not final.retried
                and final.role not in ("revise", "goldfish-structured")
                and not final.role.endswith("-repair")
            ):
                self.backends[final.model].remember_assistant(cur_content, cur_tcs, final.result.gen_ids, messages)
            asst_msg: dict[str, Any] = {"role": "assistant", "content": cur_content}
            if cur_tcs:
                asst_msg["tool_calls"] = cur_tcs
            rep_msgs = list(messages) + [asst_msg, {"role": "user", "content": repair_prompt}]
            if cap_max_tokens and repair_max_tokens is not None:
                eff_max = min(max(params.max_tokens, 192), repair_max_tokens)
            else:
                eff_max = max(params.max_tokens, repair_max_tokens or params.max_tokens)
            rep_params = GenParams(
                max_tokens=eff_max,
                temperature=params.temperature,
                top_p=params.top_p,
                stop=list(params.stop),
                seed=params.seed,
            )
            r_tools = None if disable_tools else tools
            r_choice = "none" if disable_tools else tool_choice
            repaired = self._run(final.model, role_tag, rep_msgs, r_tools, r_choice, rep_params, streamer=None)
            steps.append(repaired)
            return repaired, rep_msgs

        cur_tcs = final.parsed.openai_tool_calls() if tools_active else []

        # S1: Post-generation declined-tool safety net + non-answer / truncated-promise continuation
        if use_s1:
            if declined_set and cur_tcs:
                violated = [tc["function"]["name"] for tc in cur_tcs if tc["function"]["name"] in declined_set]
                if violated:
                    studio_telem.setdefault("s1_semantics", {})["stripped_declined_calls"] = violated
                    final, messages = _repair_once(
                        "s1-declined-tool-repair",
                        f"Tool(s) {violated} were explicitly forbidden by the user. Answer directly without calling {violated}.",
                        disable_tools=(len(violated) == len(cur_tcs) and not tools),
                    )
                    cur_tcs = final.parsed.openai_tool_calls() if tools_active else []
            s1_cont = ssem.check_non_answer_or_truncated_promise(
                messages, final.parsed.content, cur_tcs, final.result.finish_reason
            )
            if s1_cont:
                studio_telem.setdefault("s1_semantics", {})["non_answer_repaired"] = True
                final, messages = _repair_once("s1-non-answer-repair", s1_cont, repair_max_tokens=640, disable_tools=True)
                cur_tcs = []

        # S2: Outcome Self-Verifier (`blind` | `fast` | `lean_elaborate`)
        if s2_mode != "off" and not cur_tcs:
            s2_finding, s2_status = sout.check_outcome_cycle(
                s2_mode, messages, final.parsed.content, cur_tcs, lean_project_dir=s2_lean_proj
            )
            studio_telem["s2_outcome_cycle"] = {"mode": s2_mode, "status": s2_status, "repaired": bool(s2_finding)}
            if s2_finding:
                final, messages = _repair_once(f"s2-{s2_mode}-repair", s2_finding, repair_max_tokens=640, disable_tools=True)

        # S3: Native C11 `x-cruncher` exact math & BBP Pi verifier
        if use_s3 and not cur_tcs:
            s3_repair, s3_meta = sxc.check_xcruncher_math(messages, final.parsed.content, cur_tcs)
            studio_telem["s3_xcruncher"] = s3_meta
            if s3_repair:
                final, messages = _repair_once(
                    "s3-xcruncher-repair", s3_repair, repair_max_tokens=384, disable_tools=True, cap_max_tokens=True
                )

        # S4: Symbolic Ontology Deductive Closure & Constraint Verifier
        if use_s4 and not cur_tcs:
            s4_repair, s4_meta = sonto.check_symbolic_ontology(messages, final.parsed.content, cur_tcs)
            studio_telem["s4_ontology"] = s4_meta
            if s4_repair:
                final, messages = _repair_once(
                    "s4-ontology-repair", s4_repair, repair_max_tokens=256, disable_tools=True, cap_max_tokens=True
                )

        return final, messages

    @staticmethod
    def _with_system(messages: list[dict[str, Any]], extra: str) -> list[dict[str, Any]]:
        msgs = list(messages)
        n = 0
        while n < len(msgs) and msgs[n].get("role") in ("system", "developer"):
            n += 1
        return msgs[:n] + [{"role": "system", "content": extra}] + msgs[n:]

    @staticmethod
    def _clean_slate_escalation_messages(messages: list[dict[str, Any]], tools_active: bool) -> tuple[list[dict[str, Any]], bool]:
        """Strip trailing failed non-tool-call `[assistant(draft), tool(error)]` pairs while preserving user context."""
        if tools_active or len(messages) < 3:
            return messages, False
        end = len(messages)
        stripped_any = False
        while end >= 3:
            idx = end - 1
            if messages[idx].get("role") not in ("tool", "function"):
                break
            while idx >= 0 and messages[idx].get("role") in ("tool", "function"):
                idx -= 1
            if idx >= 1 and messages[idx].get("role") == "assistant" and not messages[idx].get("tool_calls"):
                if any(m.get("role") == "user" for m in messages[:idx]):
                    end = idx
                    stripped_any = True
                    continue
            break
        return (list(messages[:end]), True) if stripped_any else (messages, False)

    def _cascade(self, body, messages, tools, tool_choice, streamer, steps, telem) -> StepOutcome:
        params = parse_params(body)
        tools_active = bool(tg.active_tools(tools, tool_choice))
        marker = objective_failure(messages)
        if marker:
            telem["escalations"] = 1
            telem["escalation_reason"] = "tool_failure"
            telem["escalation_marker"] = marker
            use_clean = (
                bool(body["x_clover_clean_slate_cascade"])
                if "x_clover_clean_slate_cascade" in body
                else os.environ.get("CLOVER_CASCADE_CLEAN_SLATE", "1") != "0"
            )
            esc_msgs = messages
            if use_clean:
                esc_msgs, stripped = self._clean_slate_escalation_messages(messages, tools_active)
                telem["clean_slate_escalation"] = stripped
            final = self._run("nemotron-3.5-super-vl", "cascade-escalated", esc_msgs, tools, tool_choice, params,
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

    def _cluster_cascade(self, body, messages, tools, tool_choice, streamer, steps, telem) -> StepOutcome:
        """3-Layer Multi-Server Cluster Router:
        Layer 1: 0 ms deterministic specialist input-type gate (`goedel-prover-v2-32b`, `boltz-2`, `vjepa-2.1`, `chronos-2`),
                 escalating into the 3-tier generalist cascade if a specialist turn previously failed verification.
        Layer 2: Fast `clover-35b-a3b` (`Qwen3.6-35B`) coordinator (or `lfm3-omni-2b` for short trivial triage).
        Layer 3: Objective verifier-gated cascade: 0 failures -> `35B`, 1 failure -> `Nemotron-120B`, >=2 -> `Kimi-2.8T`.
        """
        params = parse_params(body)
        tools_active = bool(tg.active_tools(tools, tool_choice))
        fail_rounds, marker = consecutive_objective_failures(messages)

        # Layer 1: Deterministic specialist domain match when entering fresh (0 consecutive failures)
        if fail_rounds == 0:
            spec_mid, spec_reason = classify_specialist_domain(messages, self.backends)
            if spec_mid is not None:
                telem["route_reason"] = spec_reason
                telem["specialist_routed"] = spec_mid
                out_spec = self._run(
                    spec_mid,
                    "specialist-primary",
                    messages,
                    tools,
                    tool_choice,
                    params,
                    None if tools_active else streamer,
                )
                steps.append(out_spec)
                if streamer is not None and tools_active:
                    streamer.feed(out_spec.parsed.content)
                return out_spec

            # Trivial <400-char triage / formatting -> LFM3-Omni-2B (0.83 GB)
            std_mid, std_reason = classify_standard(messages, tools, tool_choice)
            if std_mid == "lfm3-omni-2b":
                telem["route_reason"] = std_reason
                out_omni = self._run("lfm3-omni-2b", "cluster-triage", messages, tools, tool_choice, params, streamer)
                steps.append(out_omni)
                return out_omni

        # Layer 3: Objective Verifier-Gated Multi-Tier Escalation (1 failure -> Nemotron-120B, >=2 -> Kimi-2.8T)
        if fail_rounds >= 2 and "kimi-k3" in self.backends:
            telem["escalations"] = 2
            telem["escalation_tier"] = "kimi-k3"
            telem["escalation_reason"] = "consecutive_tool_failures"
            telem["escalation_marker"] = marker
            final = self._run("kimi-k3", "cluster-tier3-kimi", messages, tools, tool_choice, params, streamer)
            steps.append(final)
            return final

        if fail_rounds == 1:
            telem["escalations"] = 1
            telem["escalation_tier"] = "nemotron-3.5-super-vl"
            telem["escalation_reason"] = "tool_failure"
            telem["escalation_marker"] = marker
            esc_msgs = messages
            use_clean = (
                bool(body["x_clover_clean_slate_cascade"])
                if "x_clover_clean_slate_cascade" in body
                else os.environ.get("CLOVER_CASCADE_CLEAN_SLATE", "1") != "0"
            )
            if use_clean:
                esc_msgs, stripped = self._clean_slate_escalation_messages(messages, tools_active)
                telem["clean_slate_escalation"] = stripped
            second = self._run(
                "nemotron-3.5-super-vl",
                "cluster-tier2-nemotron",
                esc_msgs,
                tools,
                tool_choice,
                params,
                None if tools_active else streamer,
            )
            steps.append(second)
            if second.retried and self._invalid(second) and "kimi-k3" in self.backends:
                telem["escalations"] = 2
                telem["escalation_tier"] = "kimi-k3"
                telem["escalation_reason"] = "invalid_tool_call_after_nemotron_retry"
                third = self._run("kimi-k3", "cluster-tier3-kimi", messages, tools, tool_choice, params, streamer)
                steps.append(third)
                return third
            if streamer is not None and tools_active:
                streamer.feed(second.parsed.content)
            return second

        # Layer 2: Default fast coordinator (`clover-35b-a3b` = `Qwen3.6-35B-A3B`)
        telem["route_reason"] = "cluster_coordinator_35b"
        first = self._run(
            "clover-35b-a3b",
            "cluster-tier1-35b",
            messages,
            tools,
            tool_choice,
            params,
            None if tools_active else streamer,
        )
        steps.append(first)
        if first.retried and self._invalid(first):
            telem["escalations"] = 1
            telem["escalation_tier"] = "nemotron-3.5-super-vl"
            telem["escalation_reason"] = "invalid_tool_call_after_retry"
            second = self._run("nemotron-3.5-super-vl", "cluster-tier2-nemotron", messages, tools, tool_choice, params, streamer)
            steps.append(second)
            return second
        if streamer is not None and tools_active:
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
