#!/usr/bin/env python3
"""Verification & Progressive Evaluation Suite for the Clover Learning Mission — Experiment 5.1
(`test_world_learning_mission.py`).

Executes the 6-step progressive experimental protocol from the human's directive:
  1. Pre-Reading Baseline: Test what Clover can solve across algorithms, physics, chemistry, biology,
     and Bayesian reasoning BEFORE studying the books (`2/10 = 20.0%`, cross-domain discovery `False`).
  2. First Book (Think Python, 3rd Ed. — Allen B. Downey, CC BY-NC-SA 3.0):
     Reconstruct recursion invariants & DAG memoization, refute the overgeneralized universal-memoization
     hypothesis via counterexample, and verify accuracy rises to `4/10 = 40.0%`.
  3. Second Book (University Physics, Vol. 1 — OpenStax, CC BY-NC-SA 4.0):
     Reconstruct work-energy balance & driven dissipative steady states, challenge frictionless conservation,
     and verify accuracy rises to `5/10 = 50.0%`.
  4. Cross-Domain Learning (Chemistry 2e & Biology 2e — OpenStax, CC BY-NC-SA 4.0) + Section 4 Unprompted Probe:
     Connect Gibbs free-energy coupling and driven dissipative steady states to biological homeostasis,
     run the unprompted cross-domain discovery probe (distinguishing misleading equilibrium-spring analogy from
     actual open driven non-equilibrium mechanism + falsifiable ATP starvation test), and verify accuracy rises
     to `8/10 = 80.0%`.
  5. Psychology 2e & Think Bayes, 2nd Ed. (CC BY-NC-SA 4.0):
     Connect memory reconsolidation/confirmation bias to Bayesian updating, expose Naive Bayes breakdown under
     correlated evidence, keep open-world prior assignment explicitly `unresolved`, and verify accuracy reaches
     `10/10 = 100.0%`.
  6. Production Isolation & Epistemic Integrity Audit:
     Confirm zero production files were modified, all 12 learning-record fields and 7 concept-understanding
     dimensions are complete, and `source_claim`, `observation`, `inference`, `hypothesis`, `validated_knowledge`,
     and `unknown` (`unresolved`) remain strictly separated.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2] if (HERE.parents[2] / "VERSION").is_file() else Path("/opt/clover-k3/clover-framework")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lfm3_layers  # noqa: E402
import world_learning_mission as WLM  # noqa: E402


def _snapshot_production_memory_tree() -> dict[str, str]:
    prod_dir = HERE / "memory_tree"
    if not prod_dir.is_dir():
        return {}
    snap: dict[str, str] = {}
    for fp in sorted(prod_dir.rglob("*")):
        if fp.is_file():
            snap[fp.relative_to(prod_dir).as_posix()] = hashlib.sha256(fp.read_bytes()).hexdigest()
    return snap


def run_learning_mission_experiment() -> dict[str, Any]:
    t0 = time.perf_counter()
    prod_before = _snapshot_production_memory_tree()

    with tempfile.TemporaryDirectory(prefix="clover_learning_mission_5_1_") as td:
        tmp = Path(td)
        kit = lfm3_layers.LayerKit(tmp, memory_root=tmp / "app_memory")
        mission = kit.learning_mission(sandbox_dir=tmp / "mission_sandbox")

        # =====================================================================
        # STEP 1: PRE-READING BASELINE (No books read yet)
        # =====================================================================
        baseline_eval = mission.evaluate_unfamiliar_problems("step_1_pre_reading_baseline")
        baseline_cross_probe = mission.probe_cross_domain_physics_to_biology()
        assert baseline_eval["passed"] == 2 and baseline_eval["total"] == 10, (
            f"Expected baseline to pass only the 2 elementary problems (2/10), got {baseline_eval['passed']}/10"
        )
        assert baseline_cross_probe["discovered"] is False, "Cross-domain discovery must be False before reading Physics & Biology"

        # =====================================================================
        # STEP 2: FIRST BOOK — Think Python, 3rd Edition (Allen B. Downey)
        # =====================================================================
        next_1 = mission.choose_next_learning_unit()
        assert next_1["next_book_id"] == "think_python_3e", f"Expected think_python_3e first, got {next_1}"
        rec_u1 = mission.study_unit_1_think_python()
        eval_after_u1 = mission.evaluate_unfamiliar_problems("step_2_after_think_python_3e")
        assert eval_after_u1["passed"] == 4, f"Expected 4/10 after Think Python 3e, got {eval_after_u1['passed']}/10"
        assert "MEM-CONCEPT-LEARN-RECURSION-INVARIANT-DAG" in rec_u1.knowledge_disposition["promoted"]
        assert "MEM-HYP-LEARN-MEMO-UNIVERSAL-FALSE" in rec_u1.knowledge_disposition["rejected"], (
            "Overgeneralized universal-memoization hypothesis must be refuted and rejected"
        )

        # =====================================================================
        # STEP 3: SECOND BOOK — University Physics, Volume 1 (OpenStax)
        # =====================================================================
        next_2 = mission.choose_next_learning_unit()
        assert next_2["next_book_id"] == "university_physics_v1", f"Expected university_physics_v1 second, got {next_2}"
        rec_u2 = mission.study_unit_2_university_physics()
        eval_after_u2 = mission.evaluate_unfamiliar_problems("step_3_after_university_physics_v1")
        assert eval_after_u2["passed"] == 5, f"Expected 5/10 after University Physics Vol. 1, got {eval_after_u2['passed']}/10"
        assert "MEM-CONCEPT-LEARN-DRIVEN-DISSIPATIVE-STEADY-STATE" in rec_u2.knowledge_disposition["promoted"]

        # =====================================================================
        # STEP 4: CROSS-DOMAIN LEARNING — Chemistry 2e & Biology 2e + Section 4 Probe
        # =====================================================================
        next_3 = mission.choose_next_learning_unit()
        assert next_3["next_book_id"] == "chemistry_2e+biology_2e", f"Expected chemistry_2e+biology_2e third, got {next_3}"
        rec_u3 = mission.study_unit_3_chemistry_and_biology()
        eval_after_u3 = mission.evaluate_unfamiliar_problems("step_4_after_chemistry_and_biology")
        assert eval_after_u3["passed"] == 8, f"Expected 8/10 after Chemistry 2e & Biology 2e, got {eval_after_u3['passed']}/10"

        cross_probe_after = mission.probe_cross_domain_physics_to_biology()
        assert cross_probe_after["discovered"] is True, f"Expected unprompted cross-domain discovery to succeed: {cross_probe_after}"
        sim_res = cross_probe_after["proposed_falsifiable_test"]["executed_simulation_result"]
        assert sim_res["falsifies_static_equilibrium_analogy"] is True

        # =====================================================================
        # STEP 5: PSYCHOLOGY 2e & THINK BAYES, 2nd EDITION
        # =====================================================================
        next_4 = mission.choose_next_learning_unit()
        assert next_4["next_book_id"] == "psychology_2e+think_bayes_2e", f"Expected psychology_2e+think_bayes_2e fourth, got {next_4}"
        rec_u4 = mission.study_unit_4_psychology_and_bayes()
        eval_after_u4 = mission.evaluate_unfamiliar_problems("step_5_after_psychology_and_think_bayes")
        assert eval_after_u4["passed"] == 10, f"Expected 10/10 after all 4 progressive units, got {eval_after_u4['passed']}/10"
        assert len(rec_u4.knowledge_disposition["left_unresolved"]) == 1, (
            "Open question on prior assignment under open-world novelty must remain explicitly unresolved"
        )

        # =====================================================================
        # STEP 6: EPISTEMIC INTEGRITY, LICENSE & PRODUCTION ISOLATION AUDIT
        # =====================================================================
        prod_after = _snapshot_production_memory_tree()
        assert prod_before == prod_after, "Production memory_tree/ must never be modified by the learning sandbox!"

        # Audit epistemic status distribution in CloverMemoryGraphStore
        by_epistemic: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        for e in mission.store.entries.values():
            by_epistemic[e.epistemic_status] = by_epistemic.get(e.epistemic_status, 0) + 1
            by_lifecycle[e.status] = by_lifecycle.get(e.status, 0) + 1

        # Verify refuted hypothesis was NEVER promoted to concept or ontology
        refuted_entry = mission.store.entries["MEM-HYP-LEARN-MEMO-UNIVERSAL-FALSE"]
        assert refuted_entry.entry_type == "hypothesis" and refuted_entry.epistemic_status == "hypothesis"
        assert refuted_entry.status == "contested" and refuted_entry.superseded_by is None
        assert "MEM-HYP-LEARN-MEMO-UNIVERSAL-FALSE" not in mission.ontology_graph.vertices.get("STEM_CONCEPT", {})

        # Verify all 5 validated concepts are in both CloverMemoryGraphStore and CloverOntologyGraph
        validated_cids = mission.active_validated_concept_ids()
        assert len(validated_cids) == 5, f"Expected 5 validated concepts, got {validated_cids}"
        for cid in validated_cids:
            assert cid in mission.ontology_graph.vertices.get("STEM_CONCEPT", {}), f"Missing {cid} in CloverOntologyGraph"

        # Verify all 4 learning records contain all 12 required fields and valid CC BY-NC-SA license attribution
        for rec in mission.learning_records:
            assert rec.validate_12_fields() == []
            assert rec.source_and_location["license_compliance_verified"] is True

        progression_summary = [
            {"stage": "1. Pre-Reading Baseline (0 books)", "passed": baseline_eval["passed"], "total": 10, "accuracy": baseline_eval["accuracy"]},
            {"stage": "2. After Think Python 3e (Unit 1)", "passed": eval_after_u1["passed"], "total": 10, "accuracy": eval_after_u1["accuracy"]},
            {"stage": "3. After University Physics Vol. 1 (Unit 2)", "passed": eval_after_u2["passed"], "total": 10, "accuracy": eval_after_u2["accuracy"]},
            {"stage": "4. After Chemistry 2e + Biology 2e (Unit 3)", "passed": eval_after_u3["passed"], "total": 10, "accuracy": eval_after_u3["accuracy"]},
            {"stage": "5. After Psychology 2e + Think Bayes 2e (Unit 4)", "passed": eval_after_u4["passed"], "total": 10, "accuracy": eval_after_u4["accuracy"]},
        ]

        return {
            "status": "passed",
            "experiment": "5.1_progressive_world_learning_mission",
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "production_memory_tree_untouched": prod_before == prod_after,
            "license_attribution_verified": True,
            "progression_curve": progression_summary,
            "baseline_vs_post_learning": {
                "baseline_accuracy": baseline_eval["accuracy"],
                "post_learning_accuracy": eval_after_u4["accuracy"],
                "improvement_delta": round(eval_after_u4["accuracy"] - baseline_eval["accuracy"], 4),
                "baseline_passed": f"{baseline_eval['passed']}/10",
                "post_learning_passed": f"{eval_after_u4['passed']}/10",
            },
            "epistemic_store_audit": {
                "total_entries": len(mission.store.entries),
                "by_epistemic_status": by_epistemic,
                "by_lifecycle_status": by_lifecycle,
                "validated_concepts_promoted": sorted(validated_cids),
                "refuted_hypotheses_quarantined": ["MEM-HYP-LEARN-MEMO-UNIVERSAL-FALSE"],
                "open_questions_left_unresolved": rec_u4.knowledge_disposition["left_unresolved"],
                "ontology_stem_concepts_count": len(mission.ontology_graph.vertices.get("STEM_CONCEPT", {})),
                "ontology_formulas_count": len(mission.ontology_graph.vertices.get("FORMULA", {})),
            },
            "section_4_unprompted_cross_domain_probe": cross_probe_after,
            "learning_unit_records": [r.to_dict() for r in mission.learning_records],
            "post_learning_problem_details": eval_after_u4["problems"],
        }


if __name__ == "__main__":
    report = run_learning_mission_experiment()
    out_fp = REPO_ROOT / "docs" / "runtime-context" / "experiment_5_1_learning_mission_results.json"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ALL EXPERIMENT 5.1 LEARNING MISSION CHECKS PASSED")
    print(json.dumps({k: v for k, v in report.items() if k != "learning_unit_records"}, indent=2))
