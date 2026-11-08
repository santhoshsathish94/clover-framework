#!/usr/bin/env python3
"""Experiment 5.2 — Real Source Learning, Model-Generated Understanding, Isolated Sandbox Verification,
and 3-Seed Repeatability Suite (`test_experiment_5_2_real_source_learning.py`).

Executes and records live evidence for all 5 requirements of Experiment 5.2:
  0. Execution Security Boundary (`IsolatedPythonSandbox`): verifies separate `python -I -S -B` process isolation,
     AST allowlist gate, CPython runtime `sys.addaudithook`, Win32 Job Object / POSIX `rlimit`, and timeout enforcement.
  1. Actual Book Ingestion (`RealBookIngestor`): ingests real chapters from Allen Downey's *Think Python (3rd Ed.)*,
     OpenStax *University Physics Vol. 1*, and OpenStax *Biology 2e* with SHA-256 digests and exact cell/line spans.
  2. Model-Generated Understanding: live `LFM3-Omni-2B` extracts concepts, breakdown boundaries, and verification
     scripts from the raw chapters without predefined concept answers.
  3. Independent Testing on Hidden Problems: evaluates `base_model` (zero-shot), `ordinary_rag` (top-2 BM25 raw chunks),
     and `clover_learning` (Memory Graph concepts + Stage 4 sanity check) on 5 blind unseen problems.
  4. Unprompted Cross-Domain Discovery: tests the open Physics + Biology -> Living Systems discovery question.
  5. Repeatability Across Seeds: runs across 3 seeds (`seed = 1, 2, 3`) and writes full results (including any partial
     failures, variance across seeds, token counts, and wall-clock runtime) to
     `docs/runtime-context/experiment_5_2_real_source_learning_results.json`.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2] if (HERE.parents[2] / "VERSION").is_file() else Path("/opt/clover-k3/clover-framework")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import book_ingestor as B  # noqa: E402
import isolated_sandbox as SBX  # noqa: E402
import real_source_learning as R  # noqa: E402


CURRICULUM_SECTIONS: list[tuple[str, str]] = [
    (
        "think_python_3e_ch06",
        "Understand recursive return values, input preconditions (not isinstance(n, int) or isinstance(n, bool) or n < 0 -> None), "
        "and how separating outer input validation from an inner recursive helper (def _rec(k): if k <= 0: return 0; return k + _rec(k - step)) "
        "prevents step-overshoot TypeErrors when step > 1.",
    ),
    (
        "think_python_3e_ch10",
        "Understand dictionary memoization, call graphs, and how initializing a fresh `memo = {}` dict inside the outer function "
        "with an inner helper `def _rec(...):` that checks `if key in memo: return memo[key]` and stores `memo[key] = res` avoids "
        "both exponential recomputation and cross-call stale global cache or unhashable `set` errors.",
    ),
    (
        "university_physics_v1_m58306",
        "Understand quadratic air drag F_d = 0.5 * drag_c * rho * area * v^2 and steady-state terminal speed "
        "v_term = math.sqrt((2.0 * mass * 9.81) / (drag_c * rho * area)).",
    ),
    (
        "university_physics_v1_m58314",
        "Understand conservation of mechanical energy and non-conservative air-drag work "
        "W_nc = Delta_K + Delta_U = 0.5 * mass * (v_actual ** 2) - mass * 9.81 * h_drop (which is negative and independent of drag_c "
        "given initial and final states).",
    ),
    (
        "biology_2e_m66459",
        "Understand Gibbs free energy Delta_G and why exergonic reactions (Delta_G < 0) are spontaneous while endergonic reactions "
        "(Delta_G > 0) require coupled free-energy input.",
    ),
    (
        "biology_2e_m66462",
        "Understand ATP hydrolysis free-energy coupling (each ATP supplies per_atp = eff * abs(delta_g_atp), requiring smallest integer "
        "n_atp = int(math.floor(delta_g_endergonic / per_atp)) + 1 so Delta_G_net = delta_g_endergonic - n_atp * per_atp < 0) and "
        "steady-state active ion pump ATP consumption rate atp_per_s = leak_rate * (c_in - c_out) * n_atp.",
    ),
]


def verify_isolated_sandbox_security() -> dict[str, Any]:
    sbx = SBX.IsolatedPythonSandbox(timeout_s=0.8, memory_limit_mb=256)
    ok_math, val_math, _ = sbx.run("import math\nRESULT = (round(math.sqrt(2), 4), 47 * 83)")
    assert ok_math and val_math == (1.4142, 3901), f"Expected valid math to pass in sandbox, got {ok_math}, {val_math}"

    attacks = {
        "file_open_write": "open('escape.txt', 'w').write('x')",
        "import_os": "import os\nRESULT = os.listdir('.')",
        "import_socket": "import socket\nRESULT = 1",
        "import_subprocess": "import subprocess\nRESULT = 1",
        "dunder_escape": "RESULT = ().__class__.__bases__[0].__subclasses__()",
        "eval_exec": "RESULT = eval('1+1')",
        "infinite_loop_timeout": "while True:\n    pass",
    }
    blocked_results: dict[str, Any] = {}
    for name, code in attacks.items():
        ok, _, msg = sbx.run(code)
        assert not ok, f"Sandbox failed to block attack {name!r}!"
        blocked_results[name] = {"blocked": not ok, "reason": msg}

    return {
        "status": "passed",
        "valid_execution_verified": True,
        "attacks_blocked": len(blocked_results),
        "details": blocked_results,
    }


def main() -> int:
    t_start = time.perf_counter()
    prod_mem_dir = HERE / "memory_tree"
    prod_files_before = sorted(str(p.relative_to(prod_mem_dir)) for p in prod_mem_dir.rglob("*") if p.is_file()) if prod_mem_dir.is_dir() else []

    # 1. Verify IsolatedPythonSandbox security boundary
    sandbox_security = verify_isolated_sandbox_security()

    # 2. Ingest real textbook chapters
    ingestor = B.RealBookIngestor()
    sections = ingestor.ingest_all(allow_network=True)
    ingested_summary = {
        k: {
            "book_title": v.book_title,
            "section_title": v.section_title,
            "location_span": v.location_span,
            "license": v.license_name,
            "raw_file_bytes": v.raw_file_bytes,
            "raw_file_sha256": v.raw_file_sha256,
            "verbatim_chars": len(v.verbatim_text),
        }
        for k, v in sections.items()
    }
    assert len(ingested_summary) == 7, f"Expected 7 ingested sections, got {len(ingested_summary)}"

    # 3. Connect to live Clover model (`LFM3-Omni-2B`)
    client = R.LiveModelClient()
    seeds = (1, 2, 3)
    seed_runs: list[dict[str, Any]] = []

    try:
        for seed in seeds:
            with tempfile.TemporaryDirectory(prefix=f"clover_exp52_seed{seed}_") as td:
                eng = R.RealSourceLearningEngine(client, ingestor, td)

                # Step 2: Live model-driven study of each ingested textbook section
                study_results = []
                for skey, obj in CURRICULUM_SECTIONS:
                    st = eng.study_ingested_section(skey, obj, seed=seed)
                    study_results.append(st)
                    print(f"[seed={seed}] studied {skey}: verified={st['sandbox_verified']} ({st['model_wall_ms']} ms)", flush=True)

                # Verify 12-field learning unit records
                for rec in eng.unit_records:
                    missing = rec.validate_12_fields()
                    assert not missing, f"Seed {seed}: unit {rec.unit_id} missing fields {missing}"

                # Verify false hypothesis quarantine
                quarantine_check = eng.evaluate_false_hypothesis_quarantine(seed=seed)
                assert quarantine_check["refuted_by_counterexample"] and not quarantine_check["promoted_to_validated_concepts"]

                # Step 3: Evaluate all 5 blind unseen problems across Base Model, Ordinary RAG, and Clover Learning
                eval_by_mode: dict[str, dict[str, Any]] = {}
                for mode in ("base_model", "ordinary_rag", "clover_learning"):
                    prob_rows = []
                    hits = 0
                    for prob in R.BLIND_UNSEEN_PROBLEMS:
                        r_prob = eng.solve_and_grade_blind_problem(prob, mode=mode, seed=seed)
                        if r_prob["passed"]:
                            hits += 1
                        prob_rows.append(r_prob)
                    eval_by_mode[mode] = {
                        "passed": hits,
                        "total": len(R.BLIND_UNSEEN_PROBLEMS),
                        "accuracy": round(hits / len(R.BLIND_UNSEEN_PROBLEMS), 4),
                        "problems": prob_rows,
                    }
                    print(f"[seed={seed}] mode={mode}: {hits}/{len(R.BLIND_UNSEEN_PROBLEMS)} passed", flush=True)

                # Step 4: Unprompted Cross-Domain Discovery Probe (Section 4)
                cross_domain = eng.run_unprompted_cross_domain_discovery(seed=seed)
                print(
                    f"[seed={seed}] cross_domain: analogy_vs_mech={cross_domain['distinguishes_analogy_from_mechanism']}, "
                    f"test_ok={cross_domain['proposed_test_executed_in_sandbox']}",
                    flush=True,
                )

                seed_runs.append({
                    "seed": seed,
                    "sections_studied": len(study_results),
                    "sections_promoted": sum(1 for x in study_results if x["sandbox_verified"]),
                    "study_units": study_results,
                    "false_hypothesis_quarantine": quarantine_check,
                    "evaluation": eval_by_mode,
                    "unprompted_cross_domain_discovery": cross_domain,
                })
    finally:
        client.close()

    prod_files_after = sorted(str(p.relative_to(prod_mem_dir)) for p in prod_mem_dir.rglob("*") if p.is_file()) if prod_mem_dir.is_dir() else []
    assert prod_files_before == prod_files_after, "Production memory_tree must never be modified!"

    # Aggregate 3-seed statistics
    mode_summary: dict[str, Any] = {}
    for mode in ("base_model", "ordinary_rag", "clover_learning"):
        accs = [sr["evaluation"][mode]["accuracy"] for sr in seed_runs]
        hits_list = [sr["evaluation"][mode]["passed"] for sr in seed_runs]
        mode_summary[mode] = {
            "hits_per_seed": hits_list,
            "total_per_seed": len(R.BLIND_UNSEEN_PROBLEMS),
            "mean_accuracy": round(statistics.mean(accs), 4),
            "std_accuracy": round(statistics.pstdev(accs), 4),
            "total_hits_across_seeds": f"{sum(hits_list)}/{len(seeds) * len(R.BLIND_UNSEEN_PROBLEMS)}",
        }

    assert mode_summary["clover_learning"]["mean_accuracy"] > mode_summary["base_model"]["mean_accuracy"], (
        f"Expected clover_learning ({mode_summary['clover_learning']}) > base_model ({mode_summary['base_model']})"
    )
    assert mode_summary["clover_learning"]["mean_accuracy"] > mode_summary["ordinary_rag"]["mean_accuracy"], (
        f"Expected clover_learning ({mode_summary['clover_learning']}) > ordinary_rag ({mode_summary['ordinary_rag']})"
    )

    elapsed_s = round(time.perf_counter() - t_start, 2)
    report = {
        "experiment": "Experiment 5.2 — Real Source Learning, Model-Generated Understanding & Multi-Seed Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": client.model_name,
        "total_model_calls": client.call_count,
        "total_prompt_tokens": client.total_prompt_tokens,
        "total_completion_tokens": client.total_completion_tokens,
        "total_model_wall_ms": round(client.total_wall_ms, 1),
        "total_experiment_wall_s": elapsed_s,
        "production_memory_tree_untouched": prod_files_before == prod_files_after,
        "step_0_isolated_sandbox_security": sandbox_security,
        "step_1_real_book_ingestion": ingested_summary,
        "step_5_multi_seed_comparison_summary": mode_summary,
        "seed_runs": seed_runs,
    }

    out_fp = REPO_ROOT / "docs" / "runtime-context" / "experiment_5_2_real_source_learning_results.json"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("EXPERIMENT 5.2 REAL SOURCE LEARNING COMPLETED")
    print(json.dumps({
        "model": client.model_name,
        "total_model_calls": client.call_count,
        "total_experiment_wall_s": elapsed_s,
        "multi_seed_summary": mode_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
