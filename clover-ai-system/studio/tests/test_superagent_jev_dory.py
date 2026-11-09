"""Unit & Integration Verification Suite for Clover SuperAgent + DecisionEngine + ModelRuntime."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

STUDIO_CLOVER = Path(__file__).resolve().parents[1] / "clover"
if str(STUDIO_CLOVER) not in sys.path:
    sys.path.insert(0, str(STUDIO_CLOVER))

from superagent_jev_dory import (
    Choice,
    CloverDecisionEngine,
    CloverSuperAgent,
    RepeatToolGuard,
    Score,
    debias_and_calibrate_logits,
    normalized_entropy_confidence,
    softmax_probs,
)


class TestCloverSuperAgentJevDory(unittest.TestCase):
    def test_stage1_decision_engine_calibration_and_injection_policy(self) -> None:
        engine = CloverDecisionEngine(prior_alpha=0.55, temperature=0.7, min_confidence=0.60)
        d_act = engine.decide_step(
            "User asks: 144 / 12. All numbers are explicitly stated and self-contained."
        )
        self.assertEqual(d_act.action, "act")
        self.assertGreaterEqual(d_act.confidence, 0.60)
        self.assertEqual(d_act.recommended_loops, 1)

        d_verify = engine.decide_step(
            "Worker A claims all 40 tests passed, but no test log has been checked yet."
        )
        self.assertEqual(d_verify.action, "verify")
        self.assertEqual(d_verify.recommended_loops, 4)

        # Verify Braedennn/OpenJev confidence inversion fix:
        # Step 2 high confidence (0.85) with action change ('gather' -> 'verify') MUST inject authoritative
        inj_high = engine.should_inject_authoritative(step=2, decision=d_verify, prev_action="gather")
        self.assertTrue(inj_high["inject"])
        self.assertEqual(inj_high["mode"], "authoritative")

        # Low confidence decision must NEVER inject as authoritative command
        probs_flat = softmax_probs([0.1, 0.0, -0.05, 0.02], temperature=1.0)
        conf_flat = normalized_entropy_confidence(probs_flat)
        self.assertLess(conf_flat, 0.10)

        # Null-prior debiasing removes artificial 'act' prior dominance
        raw_logits = [3.0, 2.2, 0.5, 0.5]      # 'act' looks highest raw
        null_prior = [2.8, 0.2, 0.2, 0.2]      # because 'act' has huge unconditional prior
        probs_adj, _ = debias_and_calibrate_logits(
            raw_logits, null_prior, prior_alpha=0.8, temperature=0.7
        )
        self.assertGreater(probs_adj[1], probs_adj[0])  # 'gather' (index 1) wins after debiasing!

    def test_stage2_and_3_persistent_task_graph_and_worker_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "superagent_events.jsonl"
            sa = CloverSuperAgent(storage_path=log_path)
            sa.register_worker("worker-a", seed_from_parent=False)
            sa.register_worker("worker-b", seed_from_parent=True)

            t1 = sa.create_task(
                subject="Gather context",
                description="Read source files",
                write_scopes=["src/engine"],
                owner_id="worker-a",
            )
            self.assertEqual(t1.status, "routed")
            self.assertEqual(len(sa.worker_inboxes["worker-a"]), 1)

            t2 = sa.create_task(
                subject="Implement module",
                description="Write code after t1",
                blocked_by=[t1.id],
                write_scopes=["src/engine/core.py"],
                owner_id="worker-b",
            )
            self.assertEqual(t2.status, "created")
            self.assertFalse(t2.ready)

            # Cycle rejection
            with self.assertRaisesRegex(ValueError, "TEAM_TASK_DEPENDENCY_CYCLE"):
                sa._assert_acyclic(t1.id, [t2.id])

            # Blocked execution rejection
            with self.assertRaisesRegex(ValueError, "TEAM_TASK_BLOCKED"):
                sa.transition_task(t2.id, t2.revision, "running")

            # Worker self-certification rejection (must go through 'verifying' + independent verifier)
            t1_run = sa.transition_task(t1.id, t1.revision, "running")
            with self.assertRaisesRegex(ValueError, "CLOVER_VERIFICATION_REQUIRED"):
                sa.transition_task(t1.id, t1_run.revision, "completed")

            # Submit candidate output -> verifying -> independent verification passes -> completed
            t1_ver = sa.transition_task(
                t1.id, t1_run.revision, "verifying", candidate_output={"files_read": 3}
            )
            t1_done = sa.verify_and_complete_task(
                t1.id,
                t1_ver.revision,
                verifier_fn=lambda out: (
                    isinstance(out, dict) and out.get("files_read", 0) == 3,
                    {"checked": "files_read == 3", "passed": True},
                ),
            )
            self.assertEqual(t1_done.status, "completed")

            # Dependent t2 is automatically routed and worker-b notified!
            t2_after = sa.get_task_view(t2.id)
            self.assertEqual(t2_after.status, "routed")
            self.assertTrue(t2_after.ready)
            self.assertEqual(len(sa.worker_inboxes["worker-b"]), 1)

            # Completed t1 must NOT have false-positive write_scope_warnings from active t2!
            t1_view_after = sa.get_task_view(t1.id)
            self.assertEqual(t1_view_after.write_scope_warnings, [])

            # Peer relay message vs settlement notice attribution
            relay = sa.send_peer_message("lead", "worker-a", "Please review t2")
            self.assertEqual(relay["source"]["form"], "relay")
            settled = sa.record_worker_settled("worker-a", "lead", "completed", "worker-a finished")
            self.assertEqual(settled["source"]["form"], "notice")

            # Chain of Recall + self-reference guard
            with self.assertRaisesRegex(ValueError, "SESSION_REFERENCE_SELF_REFERENCE"):
                sa.recall_session("worker-a", "worker-a")
            recall = sa.recall_session("lead", "worker-a")
            self.assertEqual(recall["source"]["form"], "recall")

            # Cold process restart recovery from disk
            sa_restarted = CloverSuperAgent(storage_path=log_path)
            self.assertEqual(sa_restarted.get_task_view(t1.id).status, "completed")
            self.assertEqual(sa_restarted.get_task_view(t2.id).status, "routed")

    def test_stage4_watchdog_and_repeat_tool_guard(self) -> None:
        sa = CloverSuperAgent()
        sa.register_worker("worker-stalled")
        t1 = sa.create_task("Long task", "May stall", owner_id="worker-stalled")
        t1_run = sa.transition_task(t1.id, t1.revision, "running")
        sa.tasks[t1.id].updated_at_ms = 1000

        recovered = sa.watchdog_scan_and_recover(
            now_ms=10_000, stall_timeout_ms=5_000, fallback_owner="worker-backup"
        )
        self.assertEqual(recovered, [t1.id])
        self.assertEqual(sa.get_task_view(t1.id).status, "routed")
        self.assertEqual(sa.get_task_view(t1.id).owner_id, "worker-backup")

        # RepeatToolGuard canonical JSON key order detection
        guard = RepeatToolGuard(thresholds=(2, 3))
        self.assertIsNone(guard.observe("a1", "read_file", {"path": "x.py", "line": 1}))
        r2 = guard.observe("a1", "read_file", {"line": 1, "path": "x.py"})
        self.assertIsNotNone(r2)
        self.assertEqual(r2["tier"], "gentle")
        r3 = guard.observe("a1", "read_file", {"path": "x.py", "line": 1})
        self.assertIsNotNone(r3)
        self.assertEqual(r3["tier"], "detailed")
        guard.reset_on_user_message("a1")
        self.assertIsNone(guard.observe("a1", "read_file", {"path": "x.py", "line": 1}))

    def test_stage5_and_6_adaptive_dory_loop_policy(self) -> None:
        # Easy high-confidence task -> 1 loop (72 layers)
        self.assertEqual(
            CloverDecisionEngine.select_dory_loops("act", 0.85, "What is 2+2?", chord_distance_l2=0.03),
            1,
        )
        # Moderate uncertainty -> 2 loops (112 layers)
        self.assertEqual(
            CloverDecisionEngine.select_dory_loops("gather", 0.55, "Read config", chord_distance_l2=0.07),
            2,
        )
        # Hard verification or high hypersphere chord movement -> 4 loops (192 layers)
        self.assertEqual(
            CloverDecisionEngine.select_dory_loops("verify", 0.35, "Verify claim", chord_distance_l2=0.11),
            4,
        )


if __name__ == "__main__":
    unittest.main()
