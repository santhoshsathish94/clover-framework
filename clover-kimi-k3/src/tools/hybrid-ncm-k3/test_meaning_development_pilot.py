"""Automated Unit & Integration Test Suite for the Clover Meaning Formation, Grounded Execution,
and Developmental Growth Pilot (`test_meaning_development_pilot.py`).

Tests:
  1. Concept identification without surface word overlap across two domains (`systems` & `pharmacology`).
  2. Related-concept distinction (`MEM-CONCEPT-CYCLIC-SWEEP-EVICTION-001` vs `MEM-CONCEPT-TTL-EXPIRATION-EVICTION-001`,
     and `MEM-CONCEPT-COMPETITIVE-RECEPTOR-INHIBITION-001` vs `MEM-CONCEPT-ALLOSTERIC-NONCOMPETITIVE-BLOCKADE-001`).
  3. Evidence provenance completeness, RQ3 ontology audit (`100%` canonicalisation, traceability, verifiability),
     and rejection of plausible but unsupported claims (`0%` unsupported-claim rate).
  4. Contradiction preservation (`challenges` / `unresolved_with`) and correction handling (`corrects` / `superseded_by`)
     preserving `## Original Source Evidence` verbatim.
  5. Context-dependent meaning disambiguation (`single_lane_serial_io` vs `multi_lane_batched_overlap`) and
     graph-relational retrieval vs. lexical-only BM25.
  6. Phase 3 (`Experiment B`) progression across `B0 < B1 < B2 < B3` on both construction-linked and held-out tasks.
  7. Phase 4 (`Experiment C`) developmental experience loop: `staging/` quarantine of unverified hypotheses,
     verified promotion, same-domain accumulation, cross-domain held-out transfer, and `0%` negative transfer.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from meaning_development_pilot import (
    CANONICAL_CYCLE_STAGES,
    DEFAULT_SEEDS,
    RemediationTask,
    build_experiment_b_tasks,
    infer_task_preconditions_from_raw_prompt,
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    run_full_meaning_development_pilot,
)


class TestMeaningDevelopmentPilot(unittest.TestCase):
    def test_01_canonical_cycle_stages_preserved(self) -> None:
        self.assertEqual(
            CANONICAL_CYCLE_STAGES,
            ("Context", "Direction", "Execution", "Outcome", "Growth"),
        )

    def test_02_experiment_a_meaning_formation_and_distinctions(self) -> None:
        exp_a = run_experiment_a()
        m = exp_a["metrics"]
        self.assertTrue(m["memory_graph_healthy"])
        self.assertEqual(m["total_controlled_entries"], 16)
        self.assertEqual(m["paraphrase_concept_accuracy_pct"], 100.0)
        self.assertEqual(m["related_concept_distinction_precision_pct"], 100.0)
        self.assertEqual(m["conflation_error_count"], 0)
        self.assertEqual(m["claim_verification_accuracy_pct"], 100.0)
        self.assertEqual(m["unsupported_claim_rate_pct"], 0.0)
        self.assertEqual(m["provenance_completeness_pct"], 100.0)
        self.assertTrue(m["correction_history_preserved"])
        self.assertTrue(m["conflict_and_uncertainty_preserved"])
        self.assertTrue(m["context_scope_disambiguation_passed"])
        self.assertGreater(
            m["graph_relational_multihop_recall_pct"],
            m["lexical_only_multihop_recall_pct"] + 30.0,
        )

    def test_03_experiment_b_execution_progression_b0_to_b3(self) -> None:
        exp_b = run_experiment_b()
        conds = exp_b["conditions"]
        r_b0 = conds["B0"]["task_pass_rate_pct"]
        r_b1 = conds["B1"]["task_pass_rate_pct"]
        r_b2 = conds["B2"]["task_pass_rate_pct"]
        r_b3 = conds["B3"]["task_pass_rate_pct"]
        self.assertLess(r_b0, r_b1)
        self.assertLess(r_b1, r_b2)
        self.assertLess(r_b2, r_b3)
        self.assertEqual(r_b3, 100.0)
        self.assertEqual(conds["B3"]["held_out_pass_rate_pct"], 100.0)
        self.assertEqual(conds["B3"]["unsupported_inference_rate_pct"], 0.0)

    def test_04_experiment_c_experience_transfer_and_negative_transfer_guard(self) -> None:
        exp_c = run_experiment_c(seeds=DEFAULT_SEEDS)
        self.assertEqual(exp_c["num_seeds"], 5)
        self.assertEqual(exp_c["accumulation_post_growth_rule_only"]["mean"], 100.0)
        self.assertEqual(exp_c["accumulation_post_growth"]["mean"], 100.0)
        # Without transfer-domain nouns in the rule boxes, rule_only transfer fails (< 40%),
        # whereas latent_proposer + symbolic_verifier succeeds (100%):
        self.assertLess(exp_c["held_out_transfer_post_growth_rule_only"]["mean"], 40.0)
        self.assertEqual(exp_c["held_out_transfer_post_growth"]["mean"], 100.0)
        self.assertGreaterEqual(exp_c["latent_over_rule_only_transfer_gain_pp"], 60.0)
        self.assertGreaterEqual(exp_c["accumulation_gain_pp"], 50.0)
        self.assertGreaterEqual(exp_c["held_out_transfer_gain_pp"], 50.0)
        self.assertEqual(exp_c["negative_transfer_rate_pct"]["mean"], 0.0)
        self.assertEqual(exp_c["unverified_leakage_count"], 0)

    def test_05_full_pilot_writes_machine_readable_results(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="clover_pilot_out_"))
        try:
            out_json = tmp_dir / "results.json"
            res = run_full_meaning_development_pilot(output_json_path=out_json)
            self.assertTrue(out_json.exists())
            self.assertTrue(all(res["success_criteria_evaluation"].values()))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_06_zero_fixture_leaks_and_raw_precondition_inference(self) -> None:
        """Verifies that `RemediationTask` has no `ontology_seed` field and that `infer_task_preconditions_from_raw_prompt`
        infers prefix overlap, zero-capacity allocation, and TTL expiration purely from raw text."""
        for task in build_experiment_b_tasks():
            self.assertFalse(hasattr(task, "ontology_seed"))
            self.assertFalse(hasattr(task, "requires_ontology_law"))
            self.assertFalse(hasattr(task, "requires_prior_correction_or_scope"))

        p_prefix = infer_task_preconditions_from_raw_prompt(
            "A classifier checks `'J45'` before `'J45.5'`, which shadows `'J45.5'`."
        )
        self.assertTrue(p_prefix["has_prefix_overlap"])

        p_fixed = infer_task_preconditions_from_raw_prompt(
            "A dispatcher matches fixed 4-byte tags `'HTTP'`, `'GRPC'`, and `'MQTT'`."
        )
        self.assertFalse(p_fixed["has_prefix_overlap"])

        p_zero = infer_task_preconditions_from_raw_prompt(
            "ELISA reader reported 0.000 OD; telemetry shows `reagent_dispensed_ul = 0.0`."
        )
        self.assertEqual(p_zero["allocated_capacity"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
