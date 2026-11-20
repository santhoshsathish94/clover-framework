"""Independent Evaluator & Automatic Failure-Layer Diagnosis for Clover Studio Quality Engine.

Design rule: NEVER ask the tested model whether its own output succeeded. Every scenario is
evaluated by deterministic external checks (numeric recomputation, file byte/SHA-256 verification,
PNG header & pixel variance decoding, SSE tool/permission trace inspection, and UI DOM checks).
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any


FAILURE_LAYERS = (
    "ui",
    "transport",
    "routing",
    "permissions",
    "model",
    "tool",
    "backend_capability",
    "media_pipeline",
    "persistence",
    "evaluator",
)


def decode_png_summary(path: Path) -> dict[str, Any]:
    """Independently verify that `path` is a valid, non-empty PNG and measure dimensions & pixel diversity."""
    if not path.is_file():
        return {"valid": False, "reason": f"missing file {path.name}"}
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return {"valid": False, "reason": "not a valid PNG header", "bytes": len(data)}
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    idat_parts: list[bytes] = []
    pos = 8
    while pos + 8 <= len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        if tag == b"IDAT":
            idat_parts.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + ln
    try:
        raw = zlib.decompress(b"".join(idat_parts)) if idat_parts else b""
    except zlib.error as exc:
        return {"valid": False, "reason": f"corrupt PNG IDAT stream: {exc}", "width": width, "height": height}

    # Compute byte-level Shannon entropy on decompressed scanlines to detect blank/solid images
    counts = [0] * 256
    sample = raw[::max(1, len(raw) // 16384)] if raw else b""
    for b in sample:
        counts[b] += 1
    entropy = 0.0
    if sample:
        n = float(len(sample))
        for c in counts:
            if c > 0:
                p = c / n
                entropy -= p * math.log2(p)

    return {
        "valid": width > 0 and height > 0 and len(raw) > 0,
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "entropy_bits": round(entropy, 3),
    }


def extract_Execution_trace(sse_events: list[dict[str, Any]], turn_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize the full `User Input -> Model -> Routing -> Tool Call -> Permission -> Execution -> Output` trace."""
    stages = [e.get("stage") for e in sse_events if e.get("type") == "plan" and e.get("stage")]
    offered_tools: list[str] = []
    for e in sse_events:
        if e.get("type") == "plan" and isinstance(e.get("tools"), list):
            offered_tools.extend(str(t) for t in e["tools"] if str(t) not in offered_tools)
    tool_calls = [
        {"name": e.get("name"), "arguments": e.get("arguments")}
        for e in sse_events if e.get("type") == "tool_call"
    ]
    if not tool_calls and isinstance(turn_result.get("tool_calls"), list):
        tool_calls = [
            {"name": c.get("tool") or c.get("name"), "ran": c.get("ran"), "worked": c.get("worked")}
            for c in turn_result["tool_calls"] if isinstance(c, dict)
        ]
    perm_prompts = [
        {"tool": e.get("tool"), "request_id": e.get("request_id")}
        for e in sse_events if e.get("type") == "permission"
    ]
    perm_decisions = [
        {"request_id": e.get("request_id"), "decision": e.get("decision")}
        for e in sse_events if e.get("type") == "permission_decided"
    ]
    tool_results = [
        {"name": e.get("name"), "ran": e.get("ran"), "worked": e.get("worked"), "summary": str(e.get("summary") or "")[:160]}
        for e in sse_events if e.get("type") == "tool_result"
    ]
    return {
        "stages": stages,
        "offered_tools": offered_tools,
        "tool_calls": tool_calls,
        "permission_prompts": perm_prompts,
        "permission_decisions": perm_decisions,
        "tool_results": tool_results,
        "reply_chars": len(str(turn_result.get("reply") or "")),
        "images_count": len(turn_result.get("images") or []),
    }


def diagnose_failure(
    scenario_id: str,
    *,
    expected: str,
    observed: str,
    trace: dict[str, Any],
    http_error: str = "",
    ui_errors: list[str] | None = None,
    explicit_layer: str = "",
    proposed_repair: str = "",
) -> dict[str, Any]:
    """Automatically classify the failing layer from execution trace & error signals."""
    obs_low = (observed or "").lower()
    if any(t in obs_low for t in ("llama-server exited", "erroroutofdevicememory", "connectionrefused", "timed out")):
        layer = "transport"
    else:
        layer = explicit_layer
    if not layer:
        if ui_errors:
            layer = "ui"
        elif http_error or "timeout" in obs_low or "connection" in obs_low or "http 5" in obs_low:
            layer = "transport"
        elif "backend_capability" in obs_low or "unavailable" in obs_low or "not configured" in obs_low or "procedural" in obs_low:
            layer = "backend_capability"
        elif "permission" in obs_low:
            layer = "permissions"
        elif "expected tool" in obs_low and not trace.get("offered_tools"):
            layer = "routing"
        elif trace.get("tool_results") and any(not r.get("worked", True) for r in trace["tool_results"]):
            layer = "tool"
        elif "png" in obs_low or "audio" in obs_low or "video" in obs_low or "ffprobe" in obs_low or "ffmpeg" in obs_low:
            layer = "media_pipeline"
        elif "session" in obs_low or "memory" in obs_low:
            layer = "persistence"
        else:
            layer = "model"

    if not proposed_repair:
        repairs = {
            "ui": "Inspect browser console/pageerror trace and DOM rendering in clover/web/app.html.",
            "transport": "Check Studio HTTP/SSE stream handling, server logs, or increase model prefill timeout.",
            "routing": "Update capability pattern or quiet-tool filter in clover/semantics.py so the required tool is offered.",
            "permissions": "Verify PermissionStore mode, SSE permission event emission, and /api/decision handling.",
            "backend_capability": "Install local diffusion weights (`sd-cli` + checkpoint) or configure `config/image_provider.json`.",
            "tool": "Inspect tool execution arguments and error handling in local_tool_packs.py / lfm3_native.py.",
            "media_pipeline": "Verify media decoding, FFmpeg availability, and saved artifact headers in media_dir().",
            "persistence": "Check session repository / memory store read-write roundtrip in clover_repo.py / lfm3_layers.py.",
            "model": "Inspect model prompt grounding, tool-call generation, or factual synthesis for this scenario.",
        }
        proposed_repair = repairs.get(layer, "Inspect scenario trace and isolate failing stage.")

    return {
        "scenario_id": scenario_id,
        "failure_layer": layer if layer in FAILURE_LAYERS else "model",
        "expected": expected,
        "observed": observed[:600],
        "proposed_repair": proposed_repair,
        "safe_to_retry": True,
        "trace": trace,
    }


def contains_all_keywords(text: str, groups: list[tuple[str, ...]]) -> tuple[bool, list[str]]:
    """Return `(ok, missing_descriptions)`: each tuple in `groups` requires at least one matching synonym."""
    low = str(text or "").lower()
    missing = []
    for syn_group in groups:
        if not any(s.lower() in low for s in syn_group):
            missing.append("/".join(syn_group))
    return (len(missing) == 0), missing


def contains_number_close(text: str, target: float, tol: float = 0.05) -> bool:
    """Independently check whether `text` contains a numeric value within `tol` of `target`."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text or ""))
    for m in re.finditer(r"-?\d+(?:\.\d+)?", cleaned):
        try:
            val = float(m.group(0))
            if abs(val - target) <= tol:
                return True
        except ValueError:
            continue
    return False
