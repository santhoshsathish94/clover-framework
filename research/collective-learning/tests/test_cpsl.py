"""Unit tests for the 6-module Collective Problem-Solving Learning (CPSL) protocol."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLL_ROOT = HERE.parent
PPL_ROOT = COLL_ROOT.parent / "personal-process-learning"
for p in (str(COLL_ROOT), str(PPL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cpsl.collective import LocalCollectiveExtractor, is_allowed_collective_feature, load_closed_general_vocabulary
from cpsl.collective_gateway import CollectiveGateway
from cpsl.core_registry import CoreRegistry
from cpsl.core_sync import CoreSync, CoreSyncError


class TestCPSLProtocol(unittest.TestCase):
    def setUp(self):
        self.vocab = load_closed_general_vocabulary()

    def test_closed_vocabulary_and_pii_gate(self):
        self.assertTrue(is_allowed_collective_feature("tok=preflight", self.vocab))
        self.assertTrue(is_allowed_collective_feature("ctx=code", self.vocab))
        self.assertTrue(is_allowed_collective_feature("has=doc", self.vocab))
        self.assertTrue(is_allowed_collective_feature("pat=arrow", self.vocab))
        # Private / PII tokens and personal routine tokens must be blocked
        self.assertFalse(is_allowed_collective_feature("tok=secret_project_alpha", self.vocab))
        self.assertFalse(is_allowed_collective_feature("tok=alice@acmecorp.com", self.vocab))
        self.assertFalse(is_allowed_collective_feature("tok=friday", self.vocab))
        self.assertFalse(is_allowed_collective_feature("tok=usual", self.vocab))

    def test_symbolic_unlearning_ripple_blocked_by_extractor(self):
        """Even if a second experience is deleted, LocalCollectiveExtractor never unmasks a private token."""
        extractor = LocalCollectiveExtractor("inst_test_1", consent_contribute=True)
        exp1 = {
            "id": "s1-0001",
            "context": "docs",
            "pfeats": ["ctx=docs", "has=doc", "tok=distill", "tok=merger_target_alpha_99"],
            "general": {"intent": "docs.proofread", "conf": 0.9},
            "statement": None,
            "resolved": "docs.summarize",
        }
        exp2 = {
            "id": "s1-0002",
            "context": "docs",
            "pfeats": ["ctx=docs", "has=doc", "tok=distill", "tok=patient_record_id_88"],
            "general": {"intent": "docs.proofread", "conf": 0.9},
            "statement": None,
            "resolved": "docs.summarize",
        }
        cands = extractor.export_candidates([exp1, exp2])
        self.assertEqual(len(cands), 1)
        self.assertNotIn("tok=merger_target_alpha_99", cands[0]["trigger"])
        self.assertNotIn("tok=patient_record_id_88", cands[0]["trigger"])

    def test_consent_controls_and_sybil_deduplication(self):
        gw = CollectiveGateway()
        ext_optout = LocalCollectiveExtractor("inst_optout", consent_contribute=False)
        exp = {
            "id": "s1-0001", "context": "code",
            "pfeats": ["ctx=code", "has=code", "tok=preflight"],
            "general": {"intent": "code.review", "conf": 0.8}, "statement": None,
            "resolved": ("code.test", "code.format"),
        }
        self.assertEqual(ext_optout.export_candidates([exp, dict(exp, id="s1-0002")]), [])

        ext_sybil = LocalCollectiveExtractor("inst_sybil", consent_contribute=True)
        pkgs = ext_sybil.export_candidates([exp, dict(exp, id="s1-0002")])
        self.assertEqual(len(pkgs), 1)
        for _ in range(10):
            self.assertTrue(gw.ingest(pkgs[0], consent_verified=True))
        agg = gw.aggregate_candidates()
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0]["independent_installations"], 1)
        self.assertEqual(agg[0]["total_submissions"], 10)

    def test_registry_signing_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = CoreRegistry(tmp)
            fake_eval = {
                "total_evaluated": 1,
                "promoted_count": 1,
                "rejected_count": 0,
                "promoted": [{
                    "candidate_id": "agg_1234567890abcdef",
                    "category": "problem-solving",
                    "trigger": ["ctx=code", "has=code", "tok=preflight"],
                    "target": ["code.test", "code.format"],
                    "independent_installations": 5,
                    "metrics": {"heldout_accuracy": 1.0, "heldout_lift": 1.0, "regressions": 0, "matched_domains": ["code"], "personal_hijacks": 0},
                    "gates": {"G3_generalisation": True, "G4_cross_domain": True, "G5_zero_regression": True},
                }],
                "rejected": [],
                "all_reports": [],
            }
            reg.publish_release(fake_eval, note="unit test")
            sync = CoreSync()
            rules = sync.verify_and_load(tmp, consent_receive=True)
            self.assertEqual(len(rules), 1)
            self.assertEqual(sync.verify_and_load(tmp, consent_receive=False), [])

            # Tamper with the rule file on disk
            rule_file = next((Path(tmp) / "core" / "problem-solving").glob("*.json"))
            data = json.loads(rule_file.read_text(encoding="utf-8"))
            data["target"] = ["files.delete"]
            rule_file.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(CoreSyncError):
                sync.verify_and_load(tmp, consent_receive=True)


if __name__ == "__main__":
    unittest.main()
