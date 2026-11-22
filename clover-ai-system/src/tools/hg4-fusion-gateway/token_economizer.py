"""
Clover Native `TokenEconomizer` Middleware (`clover-ai-system/src/tools/hg4-fusion-gateway/token_economizer.py`).

Implements the verified, prefix-cache-safe subset of Microsoft `TokenEconomizer` tailored to
Clover's stateful C11 `HG4-16R` CPU engines (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron-3.5-Super-VL`):

  1. **Session-Start Per-Model Tool Profiles (`prune_tools_for_session`):**
     Deterministically filters the tool list based ONLY on the model family + the initial user/system
     intent (`messages[:2]`), so the active tool schema list stays **byte-identical across all turns of a
     session**, preserving 100% of the C11 Longest-Common-Prefix (LCP) KV/Mamba cache!
     Also compacts verbose multi-sentence parameter descriptions into crisp 1-line specs.
  2. **`0 ms` Deterministic `N=3` Failure-Loop Redirect (`inject_loop_redirect_suffix`):**
     Zero LLM-judge latency. Detects `>= 3` consecutive or repeated tool failures with the same tool/error
     class (HTTP 404/403/500, Lean kernel error, Python traceback, Linux/PowerShell command-not-found) and
     appends a single redirect note strictly at the **end of the last message** (`suffix-only -> 0% prefix
     cache invalidation`).
  3. **Tool-Output Budgeter (`budget_tool_outputs` — Phase H):**
     Caps oversized raw disassembly / reverse-engineering tool payloads (`> max_chars`) to `head + tail`
     slices with an explicit omitted-byte summary so multi-megabyte binary dumps never saturate CPU prefill.
"""

from __future__ import annotations

import copy
import re
from typing import Any

# Core local coding & verification tools always retained across all models
CORE_WORKSPACE_TOOLS = frozenset({
    "read", "edit", "write", "glob", "grep", "bash", "powershell",
    "apply_patch", "view", "create",
    "s2_lean_premise_search", "xcruncher_exact_math",
})

# Network / subagent / heavy orchestration schemas stripped unless explicitly requested in session intent
OPTIONAL_WEB_TOOLS = frozenset({"webfetch", "websearch", "web_fetch", "web_search"})
OPTIONAL_META_TOOLS = frozenset({"todowrite", "todo_write", "task", "lsp"})

_WEB_INTENT_RE = re.compile(r"(?:https?://|\bweb\b|\burl\b|\bfetch\s+http|\bbrowser\b|\bonline\b)", re.IGNORECASE)
_LSP_INTENT_RE = re.compile(r"(?:\blsp\b|\bgo\s*to\s*definition\b|\bfind\s*references\b|\bsymbol\b)", re.IGNORECASE)
_SUBAGENT_INTENT_RE = re.compile(r"(?:\bsubagent\b|\bdelegate\b|\bparallel\s+agent\b|\btodo\s*list\b)", re.IGNORECASE)

_ERROR_MARKER_RE = re.compile(
    r"(?:\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\b404\b|\b403\b|\b500\b|"
    r"not\s+found|no\s+such\s+file|commandnotfoundexception|parsererror|permission\s+denied)",
    re.IGNORECASE,
)


def _extract_initial_intent_text(messages: list[dict[str, Any]]) -> str:
    """Read ONLY the initial system + first user message so tool allow-list never mutates mid-session."""
    parts: list[str] = []
    seen_user = False
    for m in messages:
        role = str(m.get("role") or "")
        if role in ("system", "developer") and not seen_user:
            parts.append(str(m.get("content") or ""))
        elif role == "user" and not seen_user:
            c = m.get("content")
            if isinstance(c, list):
                parts.append(" ".join(str(x.get("text") or "") for x in c if isinstance(x, dict)))
            else:
                parts.append(str(c or ""))
            seen_user = True
            break
    return "\n".join(parts)


def _compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Keep exact JSON Schema types/required fields intact while trimming verbose prose descriptions."""
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return tool
    out_fn = dict(fn)
    desc = str(out_fn.get("description") or "").strip()
    if len(desc) > 90:
        first_sent = desc.split(". ", 1)[0].rstrip(".") + "."
        out_fn["description"] = first_sent[:96]
    params = out_fn.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        new_props: dict[str, Any] = {}
        for pk, pv in params["properties"].items():
            if isinstance(pv, dict):
                npv = {k: v for k, v in pv.items() if k != "description"}
                new_props[pk] = npv
            else:
                new_props[pk] = pv
        new_params = dict(params)
        new_params["properties"] = new_props
        out_fn["parameters"] = new_params
    return {"type": tool.get("type", "function"), "function": out_fn}


def prune_tools_for_session(
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if not tools:
        return tools, {"tools_in": 0, "tools_out": 0}

    intent = _extract_initial_intent_text(messages)
    is_small_2b = "omni-2b" in model_id or "2b" in model_id
    want_web = (not is_small_2b) and bool(_WEB_INTENT_RE.search(intent))
    want_lsp = (not is_small_2b) and bool(_LSP_INTENT_RE.search(intent))
    want_meta = (not is_small_2b) and bool(_SUBAGENT_INTENT_RE.search(intent))

    kept: list[dict[str, Any]] = []
    dropped_names: list[str] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        name = str((fn or {}).get("name") or "").strip()
        low = name.lower()
        keep = True
        if low in OPTIONAL_WEB_TOOLS and not want_web:
            keep = False
        elif low == "lsp" and not want_lsp:
            keep = False
        elif low in OPTIONAL_META_TOOLS and low != "lsp" and not want_meta:
            keep = False
        elif low.startswith(("notebook", "browser_")) and not want_web:
            keep = False

        if keep:
            kept.append(_compact_tool_schema(t))
        else:
            dropped_names.append(name)

    return kept, {"tools_in": len(tools), "tools_out": len(kept), "dropped": dropped_names}


def _classify_error_signature(content: str) -> str:
    low = content.lower()
    if "404" in low:
        return "http_404"
    if "commandnotfoundexception" in low or "command not found" in low:
        return "cmd_not_found"
    if "no such file" in low or "filenotfounderror" in low:
        return "file_not_found"
    if "permission denied" in low or "403" in low:
        return "permission_denied"
    return "generic_tool_error"


# Iterative compiler/test feedback tools should NOT be blocked during normal multi-turn repair
ITERATIVE_VERIFIER_TOOLS = frozenset({"lean4_kernel_check", "lean4_spec_check", "python_test_oracle"})


def _build_call_id_name_map(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map `tool_call_id -> function.name` from prior `assistant` `tool_calls` (OpenAI / OpenCode format)."""
    mapping: dict[str, str] = {}
    for m in messages:
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in m.get("tool_calls") or ():
            if not isinstance(tc, dict):
                continue
            cid = str(tc.get("id") or "").strip()
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            tname = str((fn or {}).get("name") or "").strip().lower()
            if cid and tname:
                mapping[cid] = tname
    return mapping


def inject_loop_redirect_suffix(
    messages: list[dict[str, Any]],
    threshold: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Detect >= `threshold` CONSECUTIVE trailing tool failures of the same `(tool_name, error_class)`
    (resolving `tool_call_id` when `name` is omitted by OpenAI-compatible clients) and append a concise
    redirect directive to ONLY the final message (`suffix-only -> 0% prefix-cache invalidation`).
    """
    if len(messages) < threshold + 1:
        return messages, {"loop_detected": False}

    id_to_name = _build_call_id_name_map(messages)
    streak_key: tuple[str, str] | None = None
    streak_count = 0

    for m in messages:
        role = str(m.get("role") or "")
        if role != "tool":
            continue
        raw_name = str(m.get("name") or "").strip().lower()
        cid = str(m.get("tool_call_id") or "").strip()
        tname = raw_name or id_to_name.get(cid, "tool")
        if tname in ITERATIVE_VERIFIER_TOOLS:
            streak_key = None
            streak_count = 0
            continue
        body = str(m.get("content") or "")
        if _ERROR_MARKER_RE.search(body):
            sig = _classify_error_signature(body)
            key = (tname, sig)
            if key == streak_key:
                streak_count += 1
            else:
                streak_key = key
                streak_count = 1
        else:
            # Any successful tool output resets the consecutive failure streak!
            streak_key = None
            streak_count = 0

    if streak_key is None or streak_count < threshold:
        return messages, {"loop_detected": False}

    blocked_tool = streak_key[0]
    new_msgs = list(messages[:-1]) + [copy.deepcopy(messages[-1])]
    last = new_msgs[-1]
    redirect_note = (
        f"\n[TokenEconomizer Loop Guard: tool `{blocked_tool}` failed {streak_count} consecutive times ({streak_key[1]}). "
        f"Do NOT call `{blocked_tool}` again; use local workspace tools (`read`, `glob`, `grep`) or answer directly.]"
    )
    if isinstance(last.get("content"), str):
        last["content"] = last["content"] + redirect_note
    return new_msgs, {"loop_detected": True, "blocked_tool": blocked_tool, "consecutive_failures": streak_count}


def apply_economizer(
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
    pruned_tools, tool_meta = prune_tools_for_session(model_id, messages, tools)
    guarded_msgs, loop_meta = inject_loop_redirect_suffix(messages, threshold=3)
    # CRITICAL: Never mutate `pruned_tools` mid-session on loop detection — keeping `pruned_tools`
    # strictly determined by `messages[:2]` preserves 100% of the C11 LCP prefix cache!
    meta = {**tool_meta, **loop_meta}
    return guarded_msgs, pruned_tools, meta


def budget_tool_outputs(
    messages: list[dict[str, Any]],
    max_chars: int = 3200,
    head_chars: int = 2000,
    tail_chars: int = 900,
) -> tuple[list[dict[str, Any]], int]:
    """Bound oversized tool outputs (e.g. raw `objdump`/`rea` dumps) without dropping entry/exit instructions."""
    trimmed_count = 0
    out: list[dict[str, Any]] = []
    for m in messages:
        if str(m.get("role") or "") == "tool" and isinstance(m.get("content"), str):
            c = m["content"]
            if len(c) > max_chars:
                omitted = len(c) - (head_chars + tail_chars)
                nm = dict(m)
                nm["content"] = (
                    f"{c[:head_chars]}\n"
                    f"... [OutputBudgeter: omitted {omitted} middle chars; narrow address/symbol range if needed] ...\n"
                    f"{c[-tail_chars:]}"
                )
                out.append(nm)
                trimmed_count += 1
                continue
        out.append(m)
    return out, trimmed_count
