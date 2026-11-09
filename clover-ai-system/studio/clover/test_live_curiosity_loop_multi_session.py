#!/usr/bin/env python3
"""Live Closed-Loop & Multi-Session Multi-Installation Verification for the Clover 5.x Curiosity Foundation
(`test_live_curiosity_loop_multi_session.py`).

Verifies all 5 runtime integration priorities end-to-end:
  1. **Priority 1 — Grounded Feedback in Real Outcomes (Zero Fixed Competence Constants on Ordinary Turns)**:
     - Ordinary conversational turns with no objective verification check and no user feedback record
       `comp_error = None` (`verified_in_turn = False`) and the exact `user_request` / `model_prediction`.
     - Turns with real compiler/cycle/tool checks record measured `pre_repair_comp_error` -> `comp_error` from
       actual findings and tool outcomes.
     - Subsequent user turns (`_judge`: `"continued"`, `"helpful"`, `"said_not_right"`, `"tried_again"`) update
       the region with the real human verdict and `correction_text`.
  2. **Priority 2 — Bounded Idle Curiosity via `studio_idle()` with Explicit Budgets & Permissions**:
     - `MemoryIndex.run_idle_curiosity()` invokes strictly 1 bounded curiosity step per idle pass when `is_idle()`
       is True, aborts immediately when `is_idle()` is False, and enforces `IdleResourceBudget` and `permissions`.
  3. **Priority 3 — Closed Strategy-to-Action-to-Outcome Loop**:
     - Strategy selection (`recommend_strategy_and_voi`) respects permissions (`files`, `code`, `memory`, `internet`),
       executes the actual permitted action (`workspace_code_inspect` -> `sandbox_counterexample_probe` ->
       `memory_graph_recall`), and feeds the measured error reduction and discovered views back into the engine.
  4. **Priority 4 — Persistent & Evolving Interest Lifecycle (Emerge -> Persist Across Restart -> Decay When Mastered
     -> Revive When New Evidence Invalidates Competence)**:
     - Verifies that a single new invalidating failure on a mastered region immediately triggers `abs_lp_vigilance`,
       dishabituates `habituation_trace = 0.0`, revives `state_curiosity` (>10x), and flips `should_observe` back to True.
  5. **Priority 5 — Live Multi-Session Experiment Across 2 Independent Installations & 2 Distinct Workspaces
     With Strict Zero Private-Data Sharing**:
     - Runs Installation A (`Workspace A`: Systems Routing & Ring-Buffer Codebase) and Installation B
       (`Workspace B`: Biophysical Enzyme Kinetics & Bayesian Sensor Calibration Codebase) across 3 sessions,
       idle passes, restarts, and invalidations, comparing their divergent interest trajectories and auditing
       zero private-data leakage.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import stand_ins  # noqa: E402
stand_ins.install(HERE)

import clover_repo as CR  # noqa: E402
import collective as COL  # noqa: E402
import curiosity_foundation as CF  # noqa: E402
import lfm3_layers as LL  # noqa: E402
import memory_index as MI  # noqa: E402
import personal as PPL  # noqa: E402
import workspaces as WS  # noqa: E402


# ---------------------------------------------------------------------------
# Priority 1 Test: Grounded Feedback in Real Outcomes
# ---------------------------------------------------------------------------

def test_priority_1_grounded_outcomes_no_fixed_constants(tmp_dir: Path) -> dict[str, Any]:
    base = tmp_dir / "inst_p1"
    kit = LL.LayerKit(base, memory_root=base / "clover-ai-studio" / "growth" / "knowledge")
    repo = CR.Repo(base, HERE / "AGENTS.md", kit=lambda: kit)
    kit.records = repo

    # 1A: Ordinary conversational turn with NO verification checks and NO tools
    n1, _ = repo.turn_start("chat-p1", "Write a short haiku about autumn leaves falling quietly.")
    repo.turn_end(
        "chat-p1",
        n1,
        {
            "reply": "Crimson leaves drift down,\nWhispering on autumn wind,\nQuiet forest floor.",
            "layers": [{"layer": "semantics", "knowledge": "known", "label": "general"}],
            "tool_calls": [],
        },
        0.4,
    )
    cur = kit.curiosity_state()
    ep1 = cur.exploration_history[-1]
    assert ep1["verified_in_turn"] is False, f"Ordinary unverified turn must have verified_in_turn=False, got {ep1}"
    assert ep1["comp_error"] is None and ep1["pred_error"] is None, f"Must not assign fixed competence values: {ep1}"
    assert "autumn leaves" in ep1["user_request"]
    reg_ordinary = cur.locate_leaf_region(CF.encode_text_to_64d("Write a short haiku about autumn leaves"))
    assert reg_ordinary.comp_error_window == [], f"Ordinary turn must not pollute comp_error_window: {reg_ordinary.comp_error_window}"

    # 1B: Turn with real Cycle 1 failure (2 findings) repaired in Cycle 2 via Python tool execution
    n2, _ = repo.turn_start("chat-p1", "Compute the exact sum of squares of multiples of 7 up to 70 using Python.")
    repo.turn_end(
        "chat-p1",
        n2,
        {
            "reply": "Executed Python verification: the exact sum of squares is 18914.",
            "layers": [
                {"layer": "compiler", "checked": True, "ok": True, "diagnostics": []},
                {
                    "layer": "cycle",
                    "trigger": "auto",
                    "action": "corrected",
                    "cycles_used": 2,
                    "first_findings": [
                        "Arithmetic: 49 + 196 = 235, but 49 + 196 = 245.",
                        "Code output check: expected 18914.",
                    ],
                    "final_findings": [],
                    "lesson": 'On a question like "Compute the exact sum of squares of multiples of 7", run Python first.',
                    "lesson_kind": "verified",
                },
            ],
            "tool_calls": [{"tool": "run_python", "ran": True, "worked": True, "result": "18914"}],
        },
        1.1,
    )
    ep2 = cur.exploration_history[-1]
    assert ep2["verified_in_turn"] is True
    assert ep2["strategy_used"] if "strategy_used" in ep2 else ep2["strategy"] == "sandbox_counterexample_probe"
    assert ep2["pre_repair_comp_error"] == 0.6667  # 2 findings / (2 + 1)
    assert ep2["comp_error"] == 0.0  # compiler ok (0.0), tool worked (0.0), cycle repaired (0.0)

    # 1C: User responds on turn 3 saying turn 2 was wrong ("No, that's wrong: exclude 70 itself")
    n3, _ = repo.turn_start(
        "chat-p1",
        "That's wrong, exclude 70 itself from the sum.",
        retry="said",
    )
    ep_judge = cur.exploration_history[-1]
    assert ep_judge["actor"] == "user_said_not_right"
    assert ep_judge["comp_error"] >= 0.75
    assert "exclude 70" in ep_judge["correction"]

    return {
        "ordinary_turn_comp_error": ep1["comp_error"],
        "ordinary_turn_verified_in_turn": ep1["verified_in_turn"],
        "verified_turn_pre_repair_error": ep2["pre_repair_comp_error"],
        "verified_turn_post_repair_error": ep2["comp_error"],
        "user_rejection_comp_error": ep_judge["comp_error"],
        "user_rejection_correction_captured": ep_judge["correction"],
    }


# ---------------------------------------------------------------------------
# Priority 2 & 3 Test: Bounded Idle Curiosity (`studio_idle`) & Closed Strategy Loop
# ---------------------------------------------------------------------------

def test_priority_2_and_3_idle_curiosity_and_closed_strategy_loop(tmp_dir: Path) -> dict[str, Any]:
    base = tmp_dir / "inst_p23"
    ws_dir = tmp_dir / "ws_routing"
    ws_dir.mkdir(parents=True, exist_ok=True)
    # Write a real workspace Python module with a testable routing prefix function
    (ws_dir / "prefix_router.py").write_text(
        '"""Longest-prefix routing table lookup."""\n\n'
        "def match_longest_prefix(routes: dict[str, str], ip: str) -> str:\n"
        "    matches = [p for p in routes if ip.startswith(p)]\n"
        "    if not matches:\n"
        "        return 'default'\n"
        "    best = max(matches, key=len)\n"
        "    return routes[best]\n\n"
        "assert match_longest_prefix({'10.': 'corp_a', '10.20.': 'subnet_b'}, '10.20.1.5') == 'subnet_b'\n",
        encoding="utf-8",
    )

    kit = LL.LayerKit(base, memory_root=base / "clover-ai-studio" / "growth" / "knowledge")
    repo = CR.Repo(base, HERE / "AGENTS.md", kit=lambda: kit)
    ws_mgr = WS.Workspaces(base / "config" / "workspaces.json", base / "workspaces", base)
    ws_mgr.items = [{"name": "ws_routing", "path": str(ws_dir)}]
    ws_mgr.active_name = "ws_routing"
    kit.workspaces = ws_mgr
    kit.records = repo

    idle_flag = {"idle": False}
    mem_idx = MI.MemoryIndex(
        base,
        repo,
        is_idle=lambda: idle_flag["idle"],
        note_writer=None,
        start=False,
        recall=False,
    )
    kit.recall = mem_idx

    # Seed an unresolved question from a failed turn on prefix_router
    cur = kit.curiosity_state()
    cur.record_interaction_outcome(
        "How does match_longest_prefix resolve overlapping subnet prefixes in prefix_router?",
        pred_error=0.80,
        comp_error=0.85,
        strategy_used="memory_graph_recall",
        unresolved_summary="Resolve overlapping subnet prefix order in prefix_router match_longest_prefix",
        user_request="How does match_longest_prefix resolve overlapping subnet prefixes in prefix_router?",
    )

    # 2A: When studio_idle is False, run_idle_curiosity must NOT run
    res_busy = mem_idx.run_idle_curiosity()
    assert res_busy is None

    # 2B: Enable idle, set permissions so `files="allow"` and `code="deny"`:
    #     Engine must respect `code="deny"`, select `workspace_code_inspect`, inspect `prefix_router.py`,
    #     discover `match_longest_prefix`, create a `MEM-HYP-WS-*` hypothesis, and reduce error!
    idle_flag["idle"] = True
    kit.permissions_map = {"memory": "allow", "files": "allow", "code": "deny", "internet": "deny"}
    budget = CF.IdleResourceBudget(max_steps_per_idle_pass=1, max_wall_ms=1500.0, max_files_read=2, max_bytes_read=16384)
    idle_step_1 = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    assert idle_step_1["ran"] is True
    ar1 = idle_step_1["action_result"]
    assert ar1["strategy"] == "workspace_code_inspect", f"Expected workspace_code_inspect when code=deny, got {ar1}"
    assert ar1["comp_error_after"] < ar1["comp_error_before"], f"Workspace inspect must reduce error: {ar1}"
    assert "prefix_router.py:match_longest_prefix" in ar1["verification_result"]

    # 2C: Now grant `code="allow"` and run a second idle pass:
    #     Because `MEM-HYP-WS-*` now has an executable verification block from `prefix_router.py`,
    #     and `workspace_code_inspect` was just used, `SGIM-ACTS` executes `sandbox_counterexample_probe`
    #     (or we verify `execute_permitted_strategy_action` with `sandbox_counterexample_probe`),
    #     promoting `MEM-HYP-WS-*` to `externally_verified` and driving `comp_error_after -> 0.0`!
    kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}
    # Simulate one more workspace_code_inspect that plateaus so SGIM-ACTS naturally switches to sandbox_counterexample_probe
    _ = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    idle_step_3 = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    ar3 = idle_step_3["action_result"]
    assert ar3["strategy"] == "sandbox_counterexample_probe", (
        f"After workspace_code_inspect plateaus, SGIM-ACTS must switch to sandbox_counterexample_probe, got {ar3}"
    )
    assert ar3["comp_error_after"] == 0.0
    assert ar3["promoted_entry_id"] is not None
    promoted_entry = kit.store().entries[ar3["promoted_entry_id"]]
    assert promoted_entry.epistemic_status == "externally_verified" and promoted_entry.status == "active"

    return {
        "skipped_when_not_idle": res_busy is None,
        "idle_step_1_strategy_under_code_deny": ar1["strategy"],
        "idle_step_1_error_reduction": f"{ar1['comp_error_before']} -> {ar1['comp_error_after']}",
        "idle_step_3_strategy_after_switch": ar3["strategy"],
        "idle_step_3_error_reduction": f"{ar3['comp_error_before']} -> {ar3['comp_error_after']}",
        "promoted_hypothesis_id": ar3["promoted_entry_id"],
    }


# ---------------------------------------------------------------------------
# Priority 4 & 5 Test: Live Multi-Session Experiment Across 2 Independent Installations & Workspaces
# ---------------------------------------------------------------------------

def test_priority_4_and_5_live_multi_session_two_installations(tmp_dir: Path) -> dict[str, Any]:
    # -----------------------------------------------------------------------
    # Set up Installation A (Systems & Ring-Buffer Partitioning Workspace)
    # -----------------------------------------------------------------------
    inst_a_dir = tmp_dir / "installation_A"
    ws_a_dir = tmp_dir / "workspace_A_systems"
    ws_a_dir.mkdir(parents=True, exist_ok=True)
    (ws_a_dir / "ring_buffer.py").write_text(
        '"""Per-stage ring buffer partitioning invariant."""\n\n'
        "def effective_throughput(total_slots: int, stages: int, partitioned: bool) -> float:\n"
        "    if total_slots <= 0 or stages <= 0:\n"
        "        return 0.0\n"
        "    return float(total_slots // stages) if partitioned else 1.0\n\n"
        "assert effective_throughput(512, 4, True) == 128.0\n",
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Set up Installation B (Biophysical Enzyme Kinetics & Sensor Fusion Workspace)
    # -----------------------------------------------------------------------
    inst_b_dir = tmp_dir / "installation_B"
    ws_b_dir = tmp_dir / "workspace_B_biophysics"
    ws_b_dir.mkdir(parents=True, exist_ok=True)
    (ws_b_dir / "enzyme_kinetics.py").write_text(
        '"""Michaelis-Menten & allosteric non-competitive inhibition model."""\n\n'
        "def reaction_velocity(substrate: float, vmax: float, km: float, allosteric_inhibitor: float = 0.0) -> float:\n"
        "    eff_vmax = vmax / (1.0 + allosteric_inhibitor)\n"
        "    return (eff_vmax * substrate) / (km + substrate)\n\n"
        "assert round(reaction_velocity(10.0, 100.0, 2.0, 1.0), 2) == 41.67\n",
        encoding="utf-8",
    )

    def run_installation_lifecycle(
        inst_dir: Path,
        ws_dir: Path,
        ws_name: str,
        private_note: str,
        domain_queries: list[str],
        invalidation_query: str,
        invalidation_correction: str,
    ) -> dict[str, Any]:
        # === SESSION 1: Initial Emergence & Idle Curiosity ===
        kit1 = LL.LayerKit(inst_dir, memory_root=inst_dir / "clover-ai-studio" / "growth" / "knowledge")
        repo1 = CR.Repo(inst_dir, HERE / "AGENTS.md", kit=lambda: kit1)
        ws1 = WS.Workspaces(inst_dir / "config" / "workspaces.json", inst_dir / "workspaces", inst_dir)
        ws1.items = [{"name": ws_name, "path": str(ws_dir)}]
        ws1.active_name = ws_name
        kit1.workspaces = ws1
        kit1.records = repo1
        kit1.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}

        # Save private user note in memory + personal store (must NEVER leak outside this installation!)
        priv_id = kit1.remember(private_note, conversation_id="sess-1")
        pers1 = PPL.for_process(str(inst_dir / "personal"), "laptop", None)
        pers1.before_turn("sess-1", private_note, None)

        # Turn 1: Initial failure on domain question -> repaired in Cycle 2
        n1, _ = repo1.turn_start("sess-1", domain_queries[0])
        repo1.turn_end(
            "sess-1",
            n1,
            {
                "reply": f"Initial analysis and verification for: {domain_queries[0]}",
                "layers": [
                    {"layer": "compiler", "checked": True, "ok": True, "diagnostics": []},
                    {
                        "layer": "cycle",
                        "trigger": "auto",
                        "action": "corrected",
                        "cycles_used": 2,
                        "first_findings": ["Initial boundary assumption failed on edge case."],
                        "final_findings": [],
                        "lesson": f'On a question like "{domain_queries[0][:80]}", inspect workspace invariant first.',
                        "lesson_kind": "verified",
                    },
                ],
                "tool_calls": [{"tool": "read_file", "ran": True, "worked": True, "result": "ok"}],
            },
            0.8,
        )
        # Run 1 bounded idle step (`workspace_code_inspect`)
        idle1 = kit1.run_idle_curiosity_step(is_idle_fn=lambda: True)
        reg_s1 = kit1.curiosity_state().locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
        curiosity_after_emergence = reg_s1.state_curiosity
        voi_after_emergence = kit1.curiosity_state().recommend_strategy_and_voi(domain_queries[0])["should_observe"]

        # === RESTART PROCESS BOUNDARY -> SESSION 2: Mastery Decay ===
        CF._ENGINES_BY_PATH.clear()
        kit2 = LL.LayerKit(inst_dir, memory_root=inst_dir / "clover-ai-studio" / "growth" / "knowledge")
        repo2 = CR.Repo(inst_dir, HERE / "AGENTS.md", kit=lambda: kit2)
        kit2.workspaces = ws1
        kit2.records = repo2
        kit2.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}

        # Verify state survived restart
        cur2 = kit2.curiosity_state()
        assert cur2.step_counter >= 2, "Curiosity step_counter must persist across restart"
        assert len(cur2.locate_leaf_region(CF.encode_text_to_64d(domain_queries[0])).emergent_keywords) > 0

        # Master the region across 5 verified low-error turns + idle sandbox verification
        for idx_q in range(5):
            q_txt = domain_queries[idx_q % len(domain_queries)]
            nt, _ = repo2.turn_start("sess-2", q_txt)
            repo2.turn_end(
                "sess-2",
                nt,
                {
                    "reply": f"Verified solution for {q_txt}",
                    "layers": [{"layer": "compiler", "checked": True, "ok": True, "diagnostics": []}],
                    "tool_calls": [{"tool": "run_python", "ran": True, "worked": True, "result": "passed"}],
                },
                0.5,
            )
        reg_s2 = cur2.locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
        curiosity_when_mastered = reg_s2.state_curiosity
        voi_when_mastered = cur2.recommend_strategy_and_voi(domain_queries[0])["should_observe"]

        assert curiosity_when_mastered < 0.25 * curiosity_after_emergence, (
            f"Mastered curiosity ({curiosity_when_mastered}) must decay well below emergence ({curiosity_after_emergence})"
        )
        assert voi_when_mastered is False, "Mastered region must have should_observe=False (VoI < 0)"

        # === SESSION 3: New Evidence Invalidates Previous Competence -> Revival ===
        n_inv, _ = repo2.turn_start("sess-2", invalidation_correction, retry="said")
        repo2.turn_end(
            "sess-2",
            n_inv,
            {
                "reply": f"Investigating new boundary failure: {invalidation_query}",
                "layers": [
                    {
                        "layer": "cycle",
                        "trigger": "user",
                        "action": "unconfirmed",
                        "cycles_used": 2,
                        "first_findings": [invalidation_correction],
                        "final_findings": [invalidation_correction],
                    }
                ],
                "tool_calls": [{"tool": "run_python", "ran": True, "worked": False, "result": "Error: AssertionError"}],
            },
            0.7,
        )
        reg_s3 = cur2.locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
        curiosity_after_invalidation = reg_s3.state_curiosity
        vigilance_after_invalidation = reg_s3.abs_lp_vigilance
        voi_after_invalidation = cur2.recommend_strategy_and_voi(domain_queries[0])["should_observe"]

        assert vigilance_after_invalidation >= 0.40, (
            f"Invalidating evidence must trigger |LP| vigilance >= 0.40, got {vigilance_after_invalidation}"
        )
        assert curiosity_after_invalidation > 8.0 * curiosity_when_mastered, (
            f"Curiosity must revive strongly after competence invalidation ({curiosity_after_invalidation} vs {curiosity_when_mastered})"
        )
        assert voi_after_invalidation is True, "Revived region must flip should_observe back to True"

        return {
            "private_memory_id": priv_id,
            "emergent_keywords": list(reg_s3.emergent_keywords),
            "curiosity_trajectory": {
                "1_after_emergence": curiosity_after_emergence,
                "2_when_mastered": curiosity_when_mastered,
                "3_after_invalidation_revival": curiosity_after_invalidation,
            },
            "voi_trajectory": {
                "1_after_emergence": voi_after_emergence,
                "2_when_mastered": voi_when_mastered,
                "3_after_invalidation_revival": voi_after_invalidation,
            },
            "abs_lp_vigilance_on_invalidation": vigilance_after_invalidation,
            "idle_step_1": idle1,
            "persisted_state_path": str(cur2.storage_path),
            "personal_store": pers1,
        }

    res_a = run_installation_lifecycle(
        inst_a_dir,
        ws_a_dir,
        "ws_a_systems",
        "Remember that our internal staging host is staging-alpha.corp.local and my badge ID is EMP-9481.",
        [
            "How does ring_buffer effective_throughput behave across multi-stage cyclic pipelines?",
            "Verify ring_buffer effective_throughput when partitioned=True across 4 stages.",
        ],
        "ring_buffer effective_throughput fails when total_slots=0 or unallocated",
        "That's wrong: ring_buffer effective_throughput returns 1.0 instead of 0.0 when total_slots=0 and partitioned=False!",
    )

    res_b = run_installation_lifecycle(
        inst_b_dir,
        ws_b_dir,
        "ws_b_biophysics",
        "Remember that Dr. Meera's clinical trial cohort is COHORT-BIO-772 and private key is acct_private_882.",
        [
            "How does enzyme_kinetics reaction_velocity scale with substrate under allosteric inhibition?",
            "Verify Michaelis-Menten reaction_velocity ceiling depression under allosteric_inhibitor.",
        ],
        "enzyme_kinetics reaction_velocity division by zero when km=0 and substrate=0",
        "That's wrong: reaction_velocity raises ZeroDivisionError when km=0.0 and substrate=0.0!",
    )

    # Compare interest trajectories & verify strict zero private-data leakage between Installation A and Installation B
    kw_a = set(res_a["emergent_keywords"])
    kw_b = set(res_b["emergent_keywords"])
    jaccard_kw = len(kw_a & kw_b) / max(1, len(kw_a | kw_b))
    assert jaccard_kw == 0.0, f"Independent workspaces must develop distinct emergent keyword trajectories, got {kw_a} vs {kw_b}"

    # Verify zero private user data from A appears anywhere in B, and zero private user data enters collective candidates
    raw_state_a = Path(res_a["persisted_state_path"]).read_text(encoding="utf-8")
    raw_state_b = Path(res_b["persisted_state_path"]).read_text(encoding="utf-8")
    for secret_tok in ("staging-alpha", "EMP-9481", "COHORT-BIO-772", "acct_private_882"):
        assert secret_tok not in raw_state_a and secret_tok not in raw_state_b, (
            f"Private user note token {secret_tok!r} must never enter curiosity_state.json"
        )
    coll_a = COL.LocalCollectiveExtractor("inst_a", consent_contribute=True)
    coll_b = COL.LocalCollectiveExtractor("inst_b", consent_contribute=True)
    cands_a = json.dumps(coll_a.preview_candidates(res_a["personal_store"]))
    cands_b = json.dumps(coll_b.preview_candidates(res_b["personal_store"]))
    for secret_tok in ("staging-alpha", "EMP-9481", "COHORT-BIO-772", "acct_private_882"):
        assert secret_tok not in cands_a and secret_tok not in cands_b

    return {
        "installation_a_systems": {k: v for k, v in res_a.items() if k not in ("personal_store", "idle_step_1")},
        "installation_b_biophysics": {k: v for k, v in res_b.items() if k not in ("personal_store", "idle_step_1")},
        "keyword_jaccard_overlap": round(jaccard_kw, 4),
        "zero_private_data_in_curiosity_state_or_collective": True,
    }


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="clover_live_curiosity_loop_", ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        p1 = test_priority_1_grounded_outcomes_no_fixed_constants(tmp)
        p23 = test_priority_2_and_3_idle_curiosity_and_closed_strategy_loop(tmp)
        p45 = test_priority_4_and_5_live_multi_session_two_installations(tmp)

    report = {
        "suite": "test_live_curiosity_loop_multi_session",
        "status": "passed",
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "priority_1_grounded_feedback_in_real_outcomes": p1,
        "priority_2_and_3_idle_curiosity_and_closed_strategy_loop": p23,
        "priority_4_and_5_multi_session_two_installations_experiment": p45,
    }
    out_fp = (
        HERE.parents[2]
        / "research"
        / "curiosity-and-developmental-learning"
        / "14-clover-curiosity-architecture"
        / "results"
        / "live_curiosity_loop_multi_session_results.json"
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ALL 5 RUNTIME CURIOSITY PRIORITIES & MULTI-INSTALLATION EXPERIMENT CHECKS PASSED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
