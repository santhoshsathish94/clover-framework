#!/usr/bin/env python3
"""Dual-suite sequential phase runner for Clover AI Studio capability ablations (Phases S0 -> S7).

Executes BOTH:
  1. Frozen 23-task `rh_bench_suite_v2.jsonl` (via `run_sequential_phase.run_phase` with long timeout wrapper)
  2. Frozen 48-task `studio_capability_probes_v1.jsonl` (P1..P6 objective oracles)

Before and after every phase run:
  - Verifies exact SHA-256 hashes of all 5 frozen validation files (`rh_bench_suite_v2.jsonl`,
    `lean_oracle.py`, `run_ablation.py`, `run_sequential_phase.py`, `studio_capability_probes_v1.jsonl`)
  - Runs static zero-leakage scanner over `clover-ai-system/src/tools/hg4-fusion-gateway/*.py`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
GATEWAY_DIR = REPO_ROOT / "clover-ai-system" / "src" / "tools" / "hg4-fusion-gateway"

FROZEN_SHA256: dict[str, str] = {
    "rh_bench_suite_v2.jsonl": "415c9148ad75ba7338bd365a8305baf17258e3a2fd957381ff487f7f071a1d26",
    "lean_oracle.py": "52d731367acdb28593a3d86633e2e762736ae24cd6ef7d42e780499efa58c937",
    "run_ablation.py": "6d5ef9d6e4d061f2e263fdaf510cdf22c2e8c8347c21e7d8d830291e8446a6b1",
    "run_sequential_phase.py": "78fd110a1be269b11dab82f4e3f36283cff5267ee47bf22cf6e8ecab1032c089",
    "studio_capability_probes_v1.jsonl": "f186d53743f852f720aadeb8c35346c4f6eea6abf307c8fb21275f55b4c6424d",
    "studio_capability_probes_v2.jsonl": "a6bcbf6a98f3b3d9d40ee58618fdcb4a5a5fbddd54ca29b6d50c28121d02f18f",
}

FORBIDDEN_GATEWAY_TOKENS = (
    "rh_bench_suite",
    "studio_capability_probes",
    "ground_truth_tactic",
    "ground_truth_proof",
    "lean_oracle",
    "rh_t1_",
    "rh_t2_",
    "rh_t3_",
    "swe_t4_",
    "re_t5_",
    "p1_neg_",
    "p1_pos_",
    "p2_prom_",
    "p3_math_",
    "p4_onto_",
    "p5_mem_",
    "p6_swe_",
)


def _find_file(fname: str) -> Path:
    for cand in (HERE / fname, HERE.parent / fname):
        if cand.is_file():
            return cand
    return HERE / fname


def verify_boundary_and_leakage(gateway_dir: Path | None = None) -> dict[str, str]:
    verified: dict[str, str] = {}
    for fname, expected in FROZEN_SHA256.items():
        fp = _find_file(fname)
        actual = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"FROZEN BOUNDARY VIOLATION on {fname}: expected {expected}, got {actual}")
        verified[fname] = actual

    gw_dir = gateway_dir or (GATEWAY_DIR if GATEWAY_DIR.is_dir() else Path("/opt/clover-k3/hg4-fusion-gateway"))
    if not gw_dir.is_dir():
        raise RuntimeError(f"Gateway directory not found for static anti-leakage scan: {gw_dir}")
    py_files = [p for p in sorted(gw_dir.glob("*.py")) if not p.name.startswith("test_")]
    if not py_files:
        raise RuntimeError(f"No gateway Python files found in {gw_dir}")
    for py_file in py_files:
        src = py_file.read_text(encoding="utf-8")
        for tok in FORBIDDEN_GATEWAY_TOKENS:
            if tok in src:
                raise RuntimeError(f"ANTI-LEAKAGE VIOLATION: forbidden token '{tok}' found in {py_file.name}")
    return verified


def _extract_python_code(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    return cleaned


def _run_python_sandbox(code: str, test_code: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    if not code.strip():
        return False, "empty_code"
    full = f"{code}\n\n{test_code}\n"
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "check.py"
        script.write_text(full, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, "ok"
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, (err[-1] if err else f"rc={proc.returncode}")


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fences = re.findall(r"```(?:json)?\s*\n?(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = list(reversed(fences)) + [cleaned]
    for cand in candidates:
        for m in re.finditer(r"\{[^{}]*\}", cand, flags=re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _grade_probe(probe: dict[str, Any], resp: dict[str, Any]) -> tuple[bool, str]:
    choice = ((resp.get("choices") or [{}])[0]).get("message") or {}
    content = str(choice.get("content") or "")
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    tool_calls = choice.get("tool_calls") or []
    called_names = [
        str(((tc.get("function") or {}).get("name")) or "")
        for tc in tool_calls
        if isinstance(tc, dict)
    ]
    cat = probe["category"]

    if cat == "P1_declined_tool":
        declined = set(probe.get("declined_tools") or [])
        violated = [n for n in called_names if n in declined]
        if violated:
            return False, f"called_declined_tool:{violated}"
        must_call = probe.get("must_call_tool")
        if must_call:
            if must_call in called_names:
                return True, f"called_allowed:{must_call}"
            return False, f"missing_required_tool:{must_call};called={called_names}"
        if called_names:
            return False, f"unexpected_tool_call:{called_names}"
        subs = probe.get("expected_any_substr") or []
        if any(s in cleaned for s in subs):
            return True, "ok"
        return False, f"missing_expected_substr:{ cleaned[:120] }"

    if cat in ("P2_non_answer_guard", "P6_doctest_outcome_cycle"):
        code = _extract_python_code(cleaned)
        return _run_python_sandbox(code, probe["test_code"])

    if cat == "P3_exact_math_xcruncher":
        expected = probe["expected_exact_token"]
        norm = re.sub(r"[,\s_]", "", cleaned).upper()
        if expected.upper() in norm:
            return True, "exact_match"
        return False, f"mismatch:{cleaned[-120:]}"

    if cat == "P4_ontology_constraints":
        verdict = probe["expected_verdict"].upper()
        m = re.search(r"VERDICT\s*:\s*(COMPLIANT|VIOLATION)", cleaned, flags=re.IGNORECASE)
        got_v = m.group(1).upper() if m else ("VIOLATION" if "VIOLATION" in cleaned.upper() and "COMPLIANT" not in cleaned.upper() else ("COMPLIANT" if "COMPLIANT" in cleaned.upper() and "VIOLATION" not in cleaned.upper() else ""))
        if got_v != verdict:
            return False, f"wrong_verdict:got={got_v},expected={verdict}"
        up = cleaned.upper()
        for req in probe.get("must_include_ids") or []:
            if req.upper() not in up:
                return False, f"missing_id:{req}"
        for exc in probe.get("must_exclude_ids") or []:
            # Allow mentioning in summary only if explicitly marked compliant; check verdict line / violation claim
            first_two = "\n".join(cleaned.splitlines()[:3]).upper()
            if exc.upper() in first_two:
                return False, f"falsely_blamed_id:{exc}"
        return True, "ok"

    if cat == "P5_epistemic_memory":
        obj = _extract_json_obj(cleaned)
        if obj is None:
            return False, f"invalid_json:{cleaned[:120]}"
        exp = probe["expected_json"]
        for k, v in exp.items():
            if k not in obj:
                return False, f"missing_key:{k}"
            gv = obj[k]
            if isinstance(v, float):
                try:
                    if abs(float(gv) - v) > 1e-6:
                        return False, f"key_{k}:got={gv},exp={v}"
                except Exception:
                    return False, f"key_{k}:non_numeric={gv}"
            elif isinstance(v, str):
                if str(gv).strip().lower() != v.lower():
                    return False, f"key_{k}:got={gv},exp={v}"
            else:
                if gv != v:
                    return False, f"key_{k}:got={gv},exp={v}"
        return True, "ok"

    raise ValueError(f"Unknown probe category: {cat}")


def run_studio_probes(
    gateway_url: str,
    model: str,
    max_tokens: int,
    extra_payload: dict[str, Any],
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import run_sequential_phase as rsp

    probes_path = _find_file("studio_capability_probes_v2.jsonl")
    probes = [json.loads(l) for l in probes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    opencode_tools = rsp.OPENCODE_DEFAULT_TOOLS
    base_v1 = gateway_url.rstrip("/")
    if not base_v1.endswith("/v1"):
        base_v1 = f"{base_v1}/v1"

    results: list[dict[str, Any]] = []
    by_cat: dict[str, dict[str, int]] = {}
    total_tokens = 0
    t_all = time.perf_counter()

    for idx, probe in enumerate(probes, 1):
        rsp.reset_gateway_cache(base_v1)
        payload: dict[str, Any] = {
            "model": model,
            "messages": probe["messages"],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "x_clover_economizer": True,
            "x_clover_output_budget": True,
            "x_clover_skills_index": True,
            **extra_payload,
        }
        if probe.get("use_opencode_tools"):
            payload["tools"] = opencode_tools
            payload["tool_choice"] = "auto"

        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{base_v1}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp_http:
            resp = json.loads(resp_http.read().decode("utf-8"))
        wall_s = time.perf_counter() - t0

        passed, detail = _grade_probe(probe, resp)
        toks = int((resp.get("usage") or {}).get("total_tokens") or 0)
        total_tokens += toks
        cat = probe["category"]
        slot = by_cat.setdefault(cat, {"passed": 0, "total": 0})
        slot["total"] += 1
        if passed:
            slot["passed"] += 1

        telem = resp.get("x_clover_telemetry") or {}
        results.append({
            "probe_id": probe["probe_id"],
            "category": cat,
            "passed": passed,
            "detail": detail,
            "total_tokens": toks,
            "wall_s": round(wall_s, 3),
            "studio_telemetry": telem.get("studio_extensions") or {},
            "steps": len(telem.get("steps") or []),
        })
        print(
            f"  [Probe {idx:02d}/{len(probes)}] {probe['probe_id']:12s} ({cat:24s}) "
            f"pass={str(passed):5s} tok={toks:5d} wall={wall_s:5.2f}s ({detail[:60]})",
            flush=True,
        )

    total_wall_s = time.perf_counter() - t_all
    total_passed = sum(1 for r in results if r["passed"])
    return {
        "total_passed": total_passed,
        "total_probes": len(probes),
        "pass_rate_pct": round(100.0 * total_passed / max(1, len(probes)), 2),
        "by_category": {
            c: {
                "passed": d["passed"],
                "total": d["total"],
                "pct": round(100.0 * d["passed"] / max(1, d["total"]), 2),
            }
            for c, d in by_cat.items()
        },
        "total_tokens": total_tokens,
        "total_wall_s": round(total_wall_s, 2),
        "probes": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, help="Phase label, e.g. S0_run1, S1_semantics")
    ap.add_argument("--gateway", default="http://127.0.0.1:8790")
    ap.add_argument("--gateway-dir", default="")
    ap.add_argument("--lean-project", default="/mnt/nvme1/clover-rh-bench/rh-src/lean")
    ap.add_argument("--skills-dir", default="/opt/clover-k3/skills")
    ap.add_argument("--model", default="clover-35b-a3b")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--probes-only", action="store_true")
    ap.add_argument("--s1", action="store_true", help="Enable Studio S1 (Semantics/CSI guard)")
    ap.add_argument("--s2-mode", default="", help="Studio S2 mode: blind | fast | lean_elaborate")
    ap.add_argument("--s2b-lean-project", default="", help="Lean project path for S2b lean_elaborate")
    ap.add_argument("--s3", action="store_true", help="Enable Studio S3 (x-cruncher exact math)")
    ap.add_argument("--no-s3", action="store_true", help="Explicitly disable Studio S3 for Drop-One verification")
    ap.add_argument("--s4", action="store_true", help="Enable Studio S4 (Symbolic Ontology verifier)")
    ap.add_argument("--s5", action="store_true", help="Enable Studio S5 (Epistemic Memory Graph)")
    ap.add_argument("--s6", action="store_true", help="Enable Studio S6 (JEPA / RPRE adapter)")
    ap.add_argument("--extra-json", default="{}", help="Extra JSON fields merged into request payload")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gw_dir = Path(args.gateway_dir) if args.gateway_dir else None
    verified_hashes = verify_boundary_and_leakage(gw_dir)
    extra_payload = json.loads(args.extra_json)
    if not isinstance(extra_payload, dict):
        raise ValueError("--extra-json must decode to a JSON object")
    if args.s1:
        extra_payload["x_clover_studio_s1"] = True
    if args.s2_mode:
        extra_payload["x_clover_studio_s2_mode"] = args.s2_mode
    if args.s2b_lean_project:
        extra_payload["x_clover_studio_s2b_lean_project"] = args.s2b_lean_project
    if args.s3:
        extra_payload["x_clover_studio_s3"] = True
    if args.no_s3:
        extra_payload["x_clover_studio_s3"] = False
    if args.s4:
        extra_payload["x_clover_studio_s4"] = True
    if args.s5:
        extra_payload["x_clover_studio_s5"] = True
    if args.s6:
        extra_payload["x_clover_studio_s6"] = True

    sys.path.insert(0, str(HERE))
    import run_sequential_phase as rsp

    orig_chat_v1 = rsp.chat_v1

    def _wrapped_chat_v1(base_url: str, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        eb = dict(kwargs.get("extra_body") or {})
        eb.update(extra_payload)
        kwargs["extra_body"] = eb
        kwargs.setdefault("timeout_s", 900.0)
        return orig_chat_v1(base_url, model, messages, **kwargs)

    rsp.chat_v1 = _wrapped_chat_v1

    print(f"=== [{args.phase}] Step 1/2: Running Studio Capability Probes (48 tasks) ===", flush=True)
    probe_summary = run_studio_probes(
        gateway_url=args.gateway,
        model=args.model,
        max_tokens=args.max_tokens,
        extra_payload=extra_payload,
    )

    rh_summary: dict[str, Any] | None = None
    if not args.probes_only:
        print(f"=== [{args.phase}] Step 2/2: Running Frozen 23-Task RH+SWE+RE Suite ===", flush=True)
        base_v1 = args.gateway.rstrip("/")
        if not base_v1.endswith("/v1"):
            base_v1 = f"{base_v1}/v1"
        with tempfile.TemporaryDirectory() as td:
            rh_tmp = Path(td) / "rh_out.json"
            old_argv = list(sys.argv)
            try:
                sys.argv = [
                    "run_sequential_phase.py",
                    "--phase", args.phase,
                    "--suite", str(_find_file("rh_bench_suite_v2.jsonl")),
                    "--oracle", str(_find_file("lean_oracle.py")),
                    "--lean-project", str(args.lean_project),
                    "--base-url", base_v1,
                    "--base-model", args.model,
                    "--skills-dir", str(args.skills_dir),
                    "--enable-tokeneconomizer",
                    "--enable-skills",
                    "--enable-rea",
                    "--out", str(rh_tmp),
                ]
                rsp.main()
            finally:
                sys.argv = old_argv
            rh_summary = json.loads(rh_tmp.read_text(encoding="utf-8"))

    verify_boundary_and_leakage(gw_dir)

    combined = {
        "phase": args.phase,
        "model": args.model,
        "extra_payload": extra_payload,
        "verified_sha256": verified_hashes,
        "zero_leakage_static_scan": "PASS",
        "studio_probes": probe_summary,
        "rh_suite": rh_summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    rh_passed = rh_summary.get("total_passed") if rh_summary else "SKIP"
    print(
        f"=== [{args.phase}] COMPLETE -> probes={probe_summary['total_passed']}/{probe_summary['total_probes']} "
        f"({probe_summary['pass_rate_pct']}%) | rh={rh_passed}/23 -> saved {out_path} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
