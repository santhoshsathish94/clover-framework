"""Clover Studio Capability S3: `x-cruncher` Native Exact Math & BBP Pi Verifier.

Bridges `clover-ai-system/studio/clover/xcruncher_bridge.py` (`libxcruncher` C11 NTT/Karatsuba
big-int engine + exact BBP hexadecimal Pi extractor) as a zero-system-prompt post-generation
numeric claim verifier.

Strict anti-leakage contract:
  - Inspects ONLY the user's prompt text and the assistant's generated draft.
  - Detects explicit big-integer products (`A * B` where `max(bit_length) >= 48`, or Mersenne
    products `(2^p - 1) * (2^q - 1)`), exact factorials (`N!` for `15 <= N <= 50000`), and
    fractional hexadecimal digits of Pi at an explicit 0-based hex offset.
  - Triggers a 1-turn LCP repair ONLY when the draft omits or misstates the exact value.
"""

from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path
from typing import Any

_BRIDGE_MOD: Any = False  # False = unprobed; None = unavailable/failed KAT

_GATEWAY_PI_KAT: tuple[tuple[int, str], ...] = (
    (0, "243F6A8885"),
    (194, "7C9045F12C"),
    (1000, "49F1C09B07"),
    (5000, "CAD181156B"),
)


def _load_xcruncher_bridge() -> Any:
    global _BRIDGE_MOD
    if _BRIDGE_MOD is not False:
        return _BRIDGE_MOD
    here = Path(__file__).resolve().parent
    candidates = [
        here.parents[2] / "studio" / "clover" / "xcruncher_bridge.py",
        here / "xcruncher_bridge.py",
        Path("/opt/clover-k3/studio-clover/xcruncher_bridge.py"),
    ]
    for cand in candidates:
        if cand.is_file():
            try:
                spec = importlib.util.spec_from_file_location("clover_xcruncher_bridge", str(cand))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # Mandatory Known-Answer Test across mod-3 residue classes before trusting bridge
                    for off, expected in _GATEWAY_PI_KAT:
                        if str(mod.pi_hex_exact(off, len(expected))).upper() != expected:
                            raise RuntimeError(f"Bridge KAT mismatch at offset {off}")
                    _BRIDGE_MOD = mod
                    return _BRIDGE_MOD
            except Exception:
                continue
    _BRIDGE_MOD = None
    return None


def _bbp_pi_hex_fallback(offset: int, digits: int) -> str:
    def _s(j: int, n: int) -> float:
        acc = 0.0
        for k in range(n + 1):
            r = 8 * k + j
            acc = (acc + pow(16, n - k, r) / r) % 1.0
        k = n + 1
        while True:
            t = (16.0 ** (n - k)) / (8 * k + j)
            if t < 1e-15:
                break
            acc = (acc + t) % 1.0
            k += 1
        return acc

    out: list[str] = []
    for i in range(digits):
        n = offset + i
        x = (4.0 * _s(1, n) - 2.0 * _s(4, n) - _s(5, n) - _s(6, n)) % 1.0
        out.append(f"{int(x * 16.0):X}")
    return "".join(out)


_MAX_NATIVE_PI_OFFSET = 500_000   # <= ~12 ms per call on native C11 libxcruncher (100x above largest probe)
_MAX_FALLBACK_PI_OFFSET = 1_500     # <= ~35 ms even at 20 digits on pure-Python BBP fallback
_MAX_OBLIGATIONS_PER_TURN = 4
_MAX_TOTAL_SESSION_OBLIGATIONS = 12


@functools.lru_cache(maxsize=512)
def compute_mul_exact(a: int, b: int) -> tuple[int, str]:
    xb = _load_xcruncher_bridge()
    if xb is not None:
        info = xb.active_isa_info()
        return int(xb.mul_exact(a, b)), str(info.get("backend", "xcruncher"))
    return a * b, "python-bigint-fallback"


@functools.lru_cache(maxsize=512)
def compute_factorial_exact(n: int) -> tuple[int, str]:
    xb = _load_xcruncher_bridge()
    if xb is not None:
        info = xb.active_isa_info()
        return int(xb.factorial_exact(n)), str(info.get("backend", "xcruncher"))
    acc = 1
    for k in range(2, n + 1):
        acc *= k
    return acc, "python-bigint-fallback"


@functools.lru_cache(maxsize=512)
def compute_pi_hex_exact(offset: int, digits: int) -> tuple[str, str]:
    xb = _load_xcruncher_bridge()
    if xb is not None:
        if offset > _MAX_NATIVE_PI_OFFSET:
            raise ValueError(f"offset {offset} exceeds synchronous gateway budget {_MAX_NATIVE_PI_OFFSET}")
        info = xb.active_isa_info()
        return str(xb.pi_hex_exact(offset, digits)).upper(), str(info.get("backend", "xcruncher"))
    if offset > _MAX_FALLBACK_PI_OFFSET:
        raise ValueError(f"offset {offset} exceeds fallback gateway budget {_MAX_FALLBACK_PI_OFFSET}")
    return _bbp_pi_hex_fallback(offset, digits), "python-bbp-fallback"


_MERSENNE_MUL_RE = re.compile(
    r"\(\s*2\s*(?:\^|\*\*)\s*(\d{2,4})\s*-\s*1\s*\)\s*\*\s*\(\s*2\s*(?:\^|\*\*)\s*(\d{2,4})\s*-\s*1\s*\)"
)
_BIG_MUL_RE = re.compile(r"(?<!\d)(\d{12,1200})\s*\*\s*(\d{12,1200})(?!\d)")
_FACTORIAL_RE = re.compile(
    r"(?:"
    r"`(\d{2,4})\s*!(?!=)`"
    r"|(?:\bfactorial\b|\bcompute\b|\bcalculate\b|\bevaluate\b|\bexact(?:\s+\w+){0,4}\s+value\s+of\b|\bvalue\s+of\b|\bwhat\s+is\b)[^.!?\n]{0,60}?(?<![\w\d])(\d{2,4})\s*!(?!=)"
    r"|(?<![\w\d])(\d{2,4})\s*!(?!=)\s*(?:\(\s*\d+\s+factorial\s*\)|\bfactorial\b|\bwith\s+no\s+commas\b|\bexactly\b|=)"
    r")",
    re.IGNORECASE,
)
_PI_HEX_RE = re.compile(
    r"(?:exact\s+)?(\d{1,2})\s+(?:uppercase\s+|lowercase\s+)?hex(?:adecimal)?\s+digits\s+of\s+(?:the\s+fractional\s+part\s+of\s+)?pi\b.*?(?:offset|index|position)\s*[`'\"]?(\d{1,12})[`'\"]?",
    re.IGNORECASE | re.DOTALL,
)
_XCRUNCHER_NOTE_HEADER = "[Native C11 x-cruncher verified exact values (use verbatim, do not recompute by hand):]"
_XCRUNCHER_REPAIR_HEADER = "Exact native `x-cruncher` verification detected"


def extract_prompt_math_obligations(prompt_text: str) -> list[dict[str, Any]]:
    """Extract verifiable exact big-int / factorial / BBP Pi obligations from a user prompt."""
    if not prompt_text or _XCRUNCHER_REPAIR_HEADER in prompt_text:
        return []
    # Strip any already-attached x-cruncher note before scanning so re-scanning is idempotent
    base_text = prompt_text.split(_XCRUNCHER_NOTE_HEADER, 1)[0]
    obligations: list[dict[str, Any]] = []
    try:
        for m in _MERSENNE_MUL_RE.finditer(base_text):
            if len(obligations) >= _MAX_OBLIGATIONS_PER_TURN:
                return obligations
            p, q = int(m.group(1)), int(m.group(2))
            if 16 <= p <= 6000 and 16 <= q <= 6000 and (p + q) <= 12000:
                val, backend = compute_mul_exact((1 << p) - 1, (1 << q) - 1)
                obligations.append({
                    "kind": "mersenne_mul",
                    "expr": f"(2^{p} - 1) * (2^{q} - 1)",
                    "exact": str(val),
                    "backend": backend,
                })

        for m in _BIG_MUL_RE.finditer(base_text):
            if len(obligations) >= _MAX_OBLIGATIONS_PER_TURN:
                return obligations
            a, b = int(m.group(1)), int(m.group(2))
            val, backend = compute_mul_exact(a, b)
            obligations.append({
                "kind": "big_mul",
                "expr": f"{a} * {b}",
                "exact": str(val),
                "backend": backend,
            })

        seen_fact: set[int] = set()
        for m in _FACTORIAL_RE.finditer(base_text):
            if len(obligations) >= _MAX_OBLIGATIONS_PER_TURN:
                return obligations
            n_str = m.group(1) or m.group(2) or m.group(3)
            if not n_str:
                continue
            n = int(n_str)
            if 15 <= n <= 1200 and n not in seen_fact:
                seen_fact.add(n)
                val, backend = compute_factorial_exact(n)
                obligations.append({
                    "kind": "factorial",
                    "expr": f"{n}!",
                    "exact": str(val),
                    "backend": backend,
                })

        max_pi_off = _MAX_NATIVE_PI_OFFSET if _load_xcruncher_bridge() is not None else _MAX_FALLBACK_PI_OFFSET
        for m in _PI_HEX_RE.finditer(base_text):
            if len(obligations) >= _MAX_OBLIGATIONS_PER_TURN:
                return obligations
            digits, offset = int(m.group(1)), int(m.group(2))
            if 1 <= digits <= 20 and 0 <= offset <= max_pi_off:
                hex_str, backend = compute_pi_hex_exact(offset, digits)
                obligations.append({
                    "kind": "pi_hex",
                    "expr": f"pi_hex(offset={offset}, digits={digits})",
                    "exact": hex_str,
                    "backend": backend,
                })
    except Exception:
        return []

    return obligations


def _is_title_or_summary_meta_call(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        if m.get("role") in ("system", "developer"):
            s = str(m.get("content") or "").lower()
            if "generate a title" in s or "short title" in s or "concise title" in s or "title generator" in s:
                return True
    return False


def apply_pre_xcruncher_context(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically annotate every qualifying user turn so historical prefix tokens stay 100% LCP-stable across follow-up turns."""
    if not messages or _is_title_or_summary_meta_call(messages):
        return messages, {"obligations": 0, "pre_injected": False, "latest_has_math": False}

    try:
        last_u_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_u_idx = idx
                break
        if last_u_idx < 0:
            return messages, {"obligations": 0, "pre_injected": False, "latest_has_math": False}

        out_msgs = [dict(m) for m in messages]
        total_obs = 0
        latest_obs = 0
        backend_used = ""

        for idx, m in enumerate(out_msgs):
            if m.get("role") != "user" or total_obs >= _MAX_TOTAL_SESSION_OBLIGATIONS:
                continue
            c = m.get("content")
            txt = c if isinstance(c, str) else str(c or "")
            if not txt.strip() or _XCRUNCHER_NOTE_HEADER in txt:
                continue
            if re.search(r"\bwrite\s+a\s+python\s+(?:function|class)\b", txt, re.IGNORECASE):
                continue
            obs = extract_prompt_math_obligations(txt)
            if not obs:
                continue
            note_lines = [_XCRUNCHER_NOTE_HEADER] + [f"- {ob['expr']} = {ob['exact']}" for ob in obs]
            out_msgs[idx]["content"] = f"{txt}\n\n" + "\n".join(note_lines)
            total_obs += len(obs)
            backend_used = obs[0]["backend"]
            if idx == last_u_idx:
                latest_obs = len(obs)

        if total_obs == 0:
            return messages, {"obligations": 0, "pre_injected": False, "latest_has_math": False}
        return out_msgs, {
            "obligations": total_obs,
            "pre_injected": True,
            "latest_has_math": latest_obs > 0,
            "backend": backend_used,
        }
    except Exception:
        return messages, {"obligations": 0, "pre_injected": False, "latest_has_math": False}


def check_xcruncher_math(
    messages: list[dict[str, Any]],
    draft_content: str,
    tool_calls: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Verify exact numeric obligations in `draft_content`; return `(repair_prompt_or_None, meta)`."""
    if tool_calls or not messages or _is_title_or_summary_meta_call(messages):
        return None, {"obligations": 0, "mismatches": 0}
    prompt_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            prompt_text = c if isinstance(c, str) else str(c or "")
            break
    if not prompt_text.strip():
        return None, {"obligations": 0, "mismatches": 0}

    # Skip code-generation prompts (e.g. "Write a Python function ...") so we never interfere with SWE tasks.
    if re.search(r"\bwrite\s+a\s+python\s+(?:function|class)\b", prompt_text, re.IGNORECASE):
        return None, {"obligations": 0, "mismatches": 0}

    obligations = extract_prompt_math_obligations(prompt_text)
    if not obligations:
        return None, {"obligations": 0, "mismatches": 0}

    cleaned = re.sub(r"<think>.*?</think>", "", draft_content or "", flags=re.DOTALL)
    norm_draft = re.sub(r"[,\s_]", "", cleaned).upper()

    mismatches: list[dict[str, Any]] = []
    for ob in obligations:
        if ob["exact"].upper() not in norm_draft:
            mismatches.append(ob)

    meta = {
        "obligations": len(obligations),
        "mismatches": len(mismatches),
        "backend": obligations[0]["backend"],
    }
    if not mismatches:
        return None, meta

    corrections = "\n".join(f"- `{m['expr']} = {m['exact']}`" for m in mismatches)
    repair = (
        "Exact native `x-cruncher` verification detected arithmetic digit mismatch in your previous draft:\n"
        f"{corrections}\n"
        "Re-emit your complete final answer, preserving any non-arithmetic findings from your draft while "
        "stating the exact verified digits above with no commas or scientific notation."
    )
    return repair, meta
