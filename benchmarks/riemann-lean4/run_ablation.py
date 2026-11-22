#!/usr/bin/env python3
"""
Controlled Multi-Arm Benchmark Runner (`Single` vs `Copilot Fusion Cascade` vs `+EGM` vs `+Clover MCP`)
(`benchmarks/riemann-lean4/run_ablation.py`).

Evaluates any OpenAI-compatible `/v1/chat/completions` endpoint (specifically our
`hg4-fusion-gateway` on `http://127.0.0.1:8790/v1`) across Tiers 2, 3, and 4 of the
Lean 4 `RiemannHypothesis` + Verifier SWE suite, recording:
  - Verified kernel / unit-test pass rate per tier & proof-length bin,
  - Total tokens, wall-clock seconds, and **verified passes per wall-clock hour**,
  - Objective-oracle `Cascade` escalations (`Clover-35B -> Nemotron-120B` on Lean error),
  - `Goldfish` true-catch vs false-rejection rate on Tier-3 subgoal audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from lean_oracle import verify_lean_module
from mcp_clover_tools import AntiLeakagePremiseIndex


def reset_gateway_cache(base_url: str) -> None:
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/cache/reset",
            data=b"{}",
            headers={"Content-Type": "application/json", "Authorization": "Bearer clover-local"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception:
        pass


def chat_v1(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer clover-local"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = str(msg.get("content") or "")
    usage = data.get("usage") or {}
    telemetry = data.get("x_clover_telemetry") or {}
    return content, {"ms": ms, "usage": usage, "telemetry": telemetry, "routed_model": data.get("model", model)}


def eval_tier2_task(
    task: dict[str, Any],
    base_url: str,
    arm: str,
    lean_proj: Path,
    premise_idx: AntiLeakagePremiseIndex | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    prefix = task["immutable_prefix"]
    suffix = task.get("immutable_suffix", "")
    psha = task["prefix_sha256"]

    premise_hint = ""
    if arm == "cascade_egm_mcp" and premise_idx is not None:
        mod_dir = str(Path(task["module_relpath"]).parent.as_posix())
        hits = premise_idx.search(prefix[-600:], excluded_module_substr=mod_dir, excluded_lemma=task["thm_name"], top_k=5)
        if hits:
            premise_hint = "\nRelevant Mathlib4 signatures (from S2 premise search):\n" + "\n".join(
                f"- `{h['signature']}`" for h in hits
            )

    egm_plan = ""
    tokens_used = 0
    if arm in ("cascade_egm", "cascade_egm_mcp"):
        # Step 1: Elephant writes concise tactic plan (no code veto)
        el_msgs = [
            {
                "role": "system",
                "content": "You are the EGM Elephant Architect for Lean 4 (v4.33.1 + Mathlib). Write a concise 3-bullet proof strategy naming exact Mathlib induction/rewrite lemmas. Do not use sorry.",
            },
            {"role": "user", "content": f"Lean 4 theorem to prove:\n```lean\n{prefix}```{premise_hint}"},
        ]
        el_txt, el_meta = chat_v1(base_url, "clover-egm-elephant", el_msgs, max_tokens=220)
        tokens_used += int(el_meta["usage"].get("total_tokens", 0))

        # Step 2: Stateless Goldfish checks ONLY the theorem + plan for missing cases / invalid tactics (<= 200 tok, non-veto)
        gf_msgs = [
            {
                "role": "system",
                "content": "You are a stateless EGM Goldfish Auditor. Seeing ONLY the theorem and proposed plan, list any missing induction cases or type mismatches in <= 3 short lines. Do NOT rewrite the proof.",
            },
            {"role": "user", "content": f"Theorem:\n```lean\n{prefix}```\nProposed Plan:\n{el_txt}"},
        ]
        gf_txt, gf_meta = chat_v1(base_url, "clover-egm-goldfish", gf_msgs, max_tokens=180)
        tokens_used += int(gf_meta["usage"].get("total_tokens", 0))
        egm_plan = f"\nArchitect Strategy:\n{el_txt}\nAuditor Notes (advisory only):\n{gf_txt}\n"

    sys_prompt = (
        "You are an expert Lean 4 (v4.33.1 + Mathlib) theorem prover. "
        "Note that `.{u_1, 0}` annotations in Lean 4 signatures are universe levels (`Type u_1`, `Type 0`), not value arguments. "
        "Write ZERO prose explanation: start your response IMMEDIATELY with ```lean and output ONLY the Lean 4 tactic lines after `:= by`. "
        "Never use `sorry`, `admit`, or `axiom`."
    )
    user_prompt = f"Complete the proof after `:= by` (output ONLY ```lean ... ```):\n```lean\n{prefix}```{premise_hint}{egm_plan}"

    first_model = arm if arm in ("lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl") else "clover-fusion-high"
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    reply1, m1 = chat_v1(base_url, first_model, msgs, max_tokens=768)
    tokens_used += int(m1["usage"].get("total_tokens", 0))
    chk1 = verify_lean_module(prefix, reply1, lean_project_dir=lean_proj, immutable_suffix=suffix, expected_prefix_sha256=psha)

    escalated = False
    final_chk = chk1
    if not chk1.passed and arm in ("cascade", "cascade_egm", "cascade_egm_mcp"):
        # Objective Lean 4 kernel failure triggers Cascade escalation via tool-error feedback!
        escalated = True
        msgs.extend([
            {"role": "assistant", "content": reply1},
            {
                "role": "tool",
                "name": "lean4_kernel_check",
                "content": f"error: Lean 4 kernel rejected the tactic proof ({chk1.stage}):\n{chk1.diagnostics[:1200]}\nOutput ONLY the corrected ```lean ... ``` tactic block after `:= by` with zero prose.",
            },
        ])
        reply2, m2 = chat_v1(base_url, "clover-fusion-high", msgs, max_tokens=768)
        tokens_used += int(m2["usage"].get("total_tokens", 0))
        final_chk = verify_lean_module(prefix, reply2, lean_project_dir=lean_proj, immutable_suffix=suffix, expected_prefix_sha256=psha)

    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "length_bin": task["length_bin"],
        "arm": arm,
        "passed": final_chk.passed,
        "stage": final_chk.stage,
        "escalated": escalated,
        "tokens": tokens_used,
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def eval_tier3_egm_task(task: dict[str, Any], base_url: str, model: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a stateless Lean 4 / Mathlib formal specification auditor (EGM Goldfish). "
                "Inspect the proposed Lean 4 subgoal statement BEFORE any proof work begins. "
                "Start your first line with `VERDICT: PROVABLE` or `VERDICT: UNPROVABLE`, followed by a concise counterexample or justification."
            ),
        },
        {
            "role": "user",
            "content": f"Task: {task['title']}\nProposed Lean 4 subgoal:\n```lean\n{task['proposed_lean_subgoal']}\n```",
        },
    ]
    txt, meta = chat_v1(base_url, model, msgs, max_tokens=256)
    up = txt.upper()
    if txt.lstrip().startswith("{"):
        try:
            j = json.loads(txt)
            sv = str(j.get("subgoal_verdict") or "").upper()
            if sv == "UNPROVABLE":
                pred_provable = False
            elif sv == "PROVABLE":
                pred_provable = True
            else:
                findings = j.get("findings") or []
                pred_provable = not any(str(f.get("severity", "")).lower() in ("critical", "high", "error") for f in findings)
        except ValueError:
            pred_provable = ("VERDICT: PROVABLE" in up) and ("UNPROVABLE" not in up.splitlines()[0])
    else:
        pred_provable = ("VERDICT: PROVABLE" in up) and ("UNPROVABLE" not in up.splitlines()[0])
    truth = bool(task["is_provable"])
    correct = pred_provable == truth
    false_reject = truth and (not pred_provable)
    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "model": model,
        "passed": correct,
        "false_rejection": false_reject,
        "pred_provable": pred_provable,
        "ground_truth_provable": truth,
        "tokens": int(meta["usage"].get("total_tokens", 0)),
        "wall_s": round(time.perf_counter() - t0, 2),
        "excerpt": txt[:240],
    }


_SANDBOX_PASS_SENTINEL = "__CLOVER_SANDBOX_ALL_ASSERTS_PASSED_f82a91__"


def run_python_sandbox(code_and_tests: str, timeout_s: float = 15.0, test_code: str | None = None) -> tuple[bool, str]:
    """
    Execute untrusted Python solution + assertions in an isolated `-I` subprocess.
    Imports the candidate module from a separate `driver.py` and requires `_SANDBOX_PASS_SENTINEL`
    on stdout after all assertions finish, preventing `if __name__ == '__main__': sys.exit(0)` bypasses.
    """
    if test_code is None:
        # Backward-compatible split when caller passed combined string
        sol_src = code_and_tests
        driver_src = (
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "import solution_mod\n"
            f"print({_SANDBOX_PASS_SENTINEL!r})\n"
        )
    else:
        sol_src = code_and_tests
        driver_src = (
            "import os, sys\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
            "import solution_mod\n"
            "ns = dict(vars(solution_mod))\n"
            f"exec(compile({test_code!r}, '<clover_test_suite>', 'exec'), ns, ns)\n"
            f"print({_SANDBOX_PASS_SENTINEL!r})\n"
        )

    with tempfile.TemporaryDirectory(prefix="clover_swe_sbx_") as td:
        sol_p = Path(td) / "solution_mod.py"
        drv_p = Path(td) / "driver.py"
        sol_p.write_text(sol_src, encoding="utf-8")
        drv_p.write_text(driver_src, encoding="utf-8")
        preexec = None
        if os.name != "nt":
            def _lim() -> None:
                try:
                    import resource
                    cap = 2 * (1024**3)
                    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
                except Exception:
                    pass
            preexec = _lim
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(drv_p)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=preexec,
            )
            if proc.returncode == 0 and _SANDBOX_PASS_SENTINEL in (proc.stdout or ""):
                return True, ""
            return False, ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[:600]
        except subprocess.TimeoutExpired:
            return False, f"TimeoutExpired after {timeout_s}s"


def eval_tier4_swe_task(task: dict[str, Any], base_url: str, model: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    msgs = [
        {"role": "system", "content": "You are a systems software engineer. Return clean, self-contained Python 3 code inside ```python ... ```."},
        {"role": "user", "content": task["prompt"]},
    ]
    txt, meta = chat_v1(base_url, model, msgs, max_tokens=640)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", txt, flags=re.DOTALL | re.IGNORECASE)
    code = m.group(1) if m else txt
    passed, err = run_python_sandbox(code, timeout_s=15.0, test_code=task["test_code"])
    return {
        "task_id": task["task_id"],
        "tier": task["tier"],
        "model": model,
        "passed": passed,
        "stage": "timeout" if "TimeoutExpired" in err else ("passed" if passed else "test_failed"),
        "error": err,
        "tokens": int(meta["usage"].get("total_tokens", 0)),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh_bench_tasks.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8790/v1")
    ap.add_argument("--lean-proj", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh-src/lean"))
    ap.add_argument("--arms", default="lfm3-omni-2b,clover-35b-a3b,cascade,cascade_egm,cascade_egm_mcp")
    ap.add_argument("--max-t2", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/ablation_results.json"))
    args = ap.parse_args()

    tasks = [json.loads(ln) for ln in args.tasks.read_text(encoding="utf-8").splitlines() if ln.strip()]
    t2_tasks = [t for t in tasks if t["tier"] == "tier2_leaf_tactic_repair"][: args.max_t2]
    t3_tasks = [t for t in tasks if t["tier"] == "tier3_egm_subgoal_audit"]
    t4_tasks = [t for t in tasks if t["tier"] == "tier4_verifier_swe"]

    suite_sha = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
    oracle_sha = hashlib.sha256((Path(__file__).resolve().parent / "lean_oracle.py").read_bytes()).hexdigest()
    p_idx = AntiLeakagePremiseIndex(args.lean_proj)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    summary: dict[str, Any] = {"suite_sha256": suite_sha, "oracle_sha256": oracle_sha, "arms": {}}

    for arm in arms:
        arm_rows: list[dict[str, Any]] = []
        for t in t2_tasks:
            reset_gateway_cache(args.base_url)
            r = eval_tier2_task(t, args.base_url, arm, args.lean_proj, premise_idx=p_idx)
            arm_rows.append(r)
            print(f"[{arm:16s}] {r['task_id']}: passed={r['passed']} ({r['stage']}) esc={r['escalated']} {r['wall_s']}s", flush=True)
        model_for_t34 = arm if arm in ("lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl") else (
            "clover-egm-goldfish" if "egm" in arm else "clover-fusion-high"
        )
        for t in t3_tasks:
            reset_gateway_cache(args.base_url)
            r = eval_tier3_egm_task(t, args.base_url, model_for_t34)
            arm_rows.append(r)
            print(f"[{arm:16s}] {r['task_id']}: passed={r['passed']} false_rej={r['false_rejection']} {r['wall_s']}s", flush=True)
        for t in t4_tasks:
            reset_gateway_cache(args.base_url)
            r = eval_tier4_swe_task(t, args.base_url, "clover-fusion-high" if arm not in ("lfm3-omni-2b", "clover-35b-a3b", "nemotron-3.5-super-vl") else arm)
            arm_rows.append(r)
            print(f"[{arm:16s}] {r['task_id']}: passed={r['passed']} {r['wall_s']}s", flush=True)

        tot_pass = sum(1 for r in arm_rows if r["passed"])
        tot_s = sum(r["wall_s"] for r in arm_rows) + 1e-6
        false_rejs = sum(1 for r in arm_rows if r.get("false_rejection"))
        summary["arms"][arm] = {
            "passed": tot_pass,
            "total": len(arm_rows),
            "pass_rate": round(tot_pass / max(1, len(arm_rows)), 4),
            "total_wall_s": round(tot_s, 2),
            "passes_per_hour": round(tot_pass / (tot_s / 3600.0), 2),
            "total_tokens": sum(r.get("tokens", 0) for r in arm_rows),
            "goldfish_false_rejections": false_rejs,
            "results": arm_rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in d.items() if m != "results"} for k, d in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
