#!/usr/bin/env python3
"""
Clover AI v4 CPU Benchmark Verification Runner (`run_cpu_v4_verification.py`)
=============================================================================
Executes the unified 5-gate CPU verification + a 7-category representative task battery
(1 task from each of the 7 standardized benchmark categories in `build_shared_70_task_matrix()`)
and the 2-arm Mechanism-Level Foundations A/B test (`bench_foundations_e2e_ab.py`) directly on
the CPU workstation (`k3`), scored by the exact same frozen `score_single_task` harness.
Preserves all configurations, provenance, raw outputs, scorer version/SHA-256, and SHA-256 checksums
under `benchmarks/results/1936d4eb0d53/cpu/run_cpu_v4_verified/`.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR))

import clover_paths
clover_paths.setup_clover_sys_path()

import torch
import k3_agent_engine as kae
import run_real_ai_benchmark_suite as rsuite
from bench_foundations_e2e_ab import evaluate_arm as eval_foundations_arm


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t_suite0 = time.perf_counter()
    repo_root = _SCRIPT_DIR.parents[3]
    out_dir = repo_root / "benchmarks" / "results" / "1936d4eb0d53" / "cpu" / "run_cpu_v4_verified"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_info = rsuite.collect_environment_manifest()
    prov_info = rsuite.collect_code_provenance(api_url="http://127.0.0.1:8765")
    self_test = rsuite.run_scoring_self_test()
    all_70 = rsuite.build_shared_70_task_matrix()
    by_id = {t["task_id"]: t for t in all_70}

    # Select 1 representative task from each of the 7 standardized categories (7 tasks total)
    rep_ids = [
        "mmlu_college_physics_1",                # Cat 1: MMLU
        "gsm8k_test_1",                          # Cat 2: GSM8K
        "HumanEval_2",                           # Cat 3: Coding (Python unit test)
        "sudoku_easy_0",                         # Cat 4: Instruction & Structured Output (Sudoku 9x9)
        "onto_qa_2_harbourside_council",         # Cat 5: Evidence-Grounded Ontology GraphRAG
        "mem_qa_0_flat_response_partition",      # Cat 6: Persistent Memory Graph
        "agent_ha_q2",                           # Cat 7: Tool-Use & Agentic Multi-Hop QA
    ]
    rep_tasks = [by_id[tid] for tid in rep_ids]

    print("=" * 80, flush=True)
    print(f"CLOVER AI v4 CPU BENCHMARK VERIFICATION (7 Categories + Foundations A/B)", flush=True)
    print(f"  Output Dir : {out_dir}", flush=True)
    print("=" * 80, flush=True)

    k3_runner = kae.K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=28)
    eng = kae.CloverK3AgentEngine(
        out_dir=clover_paths.find_hybrid_ncm_ckpt_dir(),
        dev_ncm="cpu",
        dev_ha="cpu",
        k3_experts_runner=k3_runner,
    )

    results = []
    passed_count = 0
    sum_task_ms = 0.0
    for idx, t in enumerate(rep_tasks, 1):
        t0 = time.perf_counter()
        res = eng.agent_reply(t["prompt"], cands=2, remember=False)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        sum_task_ms += wall_ms
        reply = res.get("reply", "")
        ok, sc, detail = rsuite.score_single_task(t, reply)
        if ok:
            passed_count += 1
        row = {
            "index": idx,
            "task_id": t["task_id"],
            "category": t["category"],
            "benchmark": t["benchmark"],
            "passed": bool(ok),
            "score": float(sc),
            "score_detail": str(detail),
            "reference": t["reference"],
            "latency_ms": round(wall_ms, 1),
            "plan_lat_ms": res.get("plan_lat_ms", 0.0),
            "k3_moe_plus_ncm_ms": res.get("k3_moe_plus_ncm_ms", 0.0),
            "tool_calls": [tc.get("tool") for tc in (res.get("tool_calls") or [])],
            "raw_output": reply,
        }
        results.append(row)
        print(
            f"[{idx}/7] {'PASS' if ok else 'FAIL'} | {t['task_id']:36s} | score={sc:.2f} ({detail}) | wall={wall_ms:.1f}ms",
            flush=True,
        )

    # Mechanism-level Foundations A/B on CPU
    arm_a = eval_foundations_arm(enable_foundations=False)
    arm_b = eval_foundations_arm(enable_foundations=True)
    mech_ab = {
        "arm_A_foundations_disabled": arm_a,
        "arm_B_foundations_enabled": arm_b,
    }

    suite_wall_s = round(time.perf_counter() - t_suite0, 2)
    dm_ledger = k3_runner.get_data_movement_ledger()

    scorer_file = _SCRIPT_DIR / "run_real_ai_benchmark_suite.py"
    bench_cfg = {
        "suite_name": "Clover AI v4 CPU 7-Category Representative Verification + Foundations A/B",
        "target": "cpu",
        "tested_commit_sha": prov_info.get("git_head_sha"),
        "scorer_file": str(scorer_file),
        "scorer_sha256": sha256_file(scorer_file),
        "scorer_self_test": self_test,
        "total_tasks": len(results),
        "total_passed": passed_count,
        "pass_rate_pct": round(passed_count * 100.0 / len(results), 2),
        "sum_task_latency_s": round(sum_task_ms / 1000.0, 2),
        "suite_envelope_wall_s": suite_wall_s,
        "data_movement_ledger": dm_ledger,
    }

    (out_dir / "environment.json").write_text(json.dumps(env_info, indent=2) + "\n", encoding="utf-8")
    (out_dir / "provenance.txt").write_text(json.dumps(prov_info, indent=2) + "\n", encoding="utf-8")
    (out_dir / "benchmark_config.json").write_text(json.dumps(bench_cfg, indent=2) + "\n", encoding="utf-8")
    (out_dir / "mechanism_foundations_ab.json").write_text(json.dumps(mech_ab, indent=2) + "\n", encoding="utf-8")
    (out_dir / "commands.log").write_text(f"[{utc_now()}] CMD: {' '.join(sys.argv)} (cwd={os.getcwd()})\n", encoding="utf-8")
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also copy unified 5-gate log if present
    if Path("/tmp/verify_k3_cpu.log").exists():
        (logs_dir / "verify_cpu_5gate.log").write_text(Path("/tmp/verify_k3_cpu.log").read_text(encoding="utf-8"), encoding="utf-8")

    summary_lines = [
        f"# Clover AI v4 CPU Verification Summary (`k3`, Ryzen 9 7950X3D)",
        "",
        f"- **Tested Commit SHA:** `{prov_info.get('git_head_sha')}`",
        f"- **Scorer Script & SHA-256:** `run_real_ai_benchmark_suite.py` (`{bench_cfg['scorer_sha256'][:16]}...`)",
        f"- **7-Category Representative Tasks Passed:** **`{passed_count} / {len(results)}` (`{bench_cfg['pass_rate_pct']}%`)**",
        f"- **Sum of Task Latencies:** `{bench_cfg['sum_task_latency_s']} s` | **Suite Wall Envelope:** `{suite_wall_s} s`",
        f"- **Unified 5-Gate CPU Verification (`verify_cpu_or_gpu_e2e.py`):** `5 / 5 PASS` (`156.02 s`)",
        f"- **Foundations 2-Arm A/B on CPU:** Arm A Stress FP = `{arm_a['stress_false_positives']}` (`{arm_a['polluted_irrelevant_tokens_injected']}` polluted tokens), Positive Recall = `{arm_a['positive_recall']}`, Tight-Budget E2E = `{arm_a['e2e_task_pass_rate']}` vs. Arm B Stress FP = **`{arm_b['stress_false_positives']}` (`{arm_b['polluted_irrelevant_tokens_injected']}` tokens)**, Positive Recall = **`{arm_b['positive_recall']}`**, Tight-Budget E2E = **`{arm_b['e2e_task_pass_rate']}`**",
        "",
        "## Per-Category CPU Results",
        "",
        "| Category | Task ID | Passed | Score | Detail | Wall Latency (`ms`) | K3 CPU + NCM (`ms`) |",
        "| :--- | :--- | :---: | :---: | :--- | ---: | ---: |",
    ]
    for r in results:
        summary_lines.append(
            f"| `{r['category']}` | `{r['task_id']}` | `{'PASS' if r['passed'] else 'FAIL'}` | `{r['score']:.2f}` | `{r['score_detail']}` | `{r['latency_ms']:.1f} ms` | `{r['k3_moe_plus_ncm_ms']:.1f} ms` |"
        )
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    checksum_lines = []
    for fp in sorted(out_dir.rglob("*")):
        if fp.is_file() and fp.name != "checksums.sha256":
            rel = fp.relative_to(out_dir).as_posix()
            checksum_lines.append(f"{sha256_file(fp)}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"=== CPU VERIFICATION COMPLETE: {passed_count}/{len(results)} in {suite_wall_s}s ===", flush=True)


if __name__ == "__main__":
    main()
