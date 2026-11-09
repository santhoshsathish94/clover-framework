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

    # 1. Valid workspace module with 2 distinct behavioral & boundary assertions
    (ws_dir / "prefix_router.py").write_text(
        '"""Longest-prefix routing table lookup."""\n\n'
        "def match_longest_prefix(routes: dict[str, str], ip: str) -> str:\n"
        "    matches = [p for p in routes if ip.startswith(p)]\n"
        "    if not matches:\n"
        "        return 'default'\n"
        "    best = max(matches, key=len)\n"
        "    return routes[best]\n\n"
        "assert match_longest_prefix({'10.': 'corp_a', '10.20.': 'subnet_b'}, '10.20.1.5') == 'subnet_b'\n"
        "assert match_longest_prefix({'10.': 'corp_a', '10.20.': 'subnet_b'}, '192.168.1.1') == 'default'\n",
        encoding="utf-8",
    )
    # 2. Innocent-named file matching query keyword 'prefix' that contains an embedded secret in its content (Fix 2B)
    (ws_dir / "prefix_config_helper.py").write_text(
        '"""Helper with embedded production secret that must be excluded by content inspection."""\n'
        'api_key = "sk-proj-99887766554433221100aabbccddeeff"\n'
        "def prefix_secret_helper(x: str) -> str:\n"
        "    return x\n",
        encoding="utf-8",
    )
    # 3. Oversized file exceeding single-file byte budget that must be skipped BEFORE reading (Fix 2A)
    (ws_dir / "prefix_huge_dump.py").write_text(
        "# Oversized file\n" + ("x = 1\n" * 4000),
        encoding="utf-8",
    )

    ws_manifest_before = CF.snapshot_workspace_manifest(ws_dir)

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

    # Fix 1A: Verify that an existence-only `callable(...)` probe or constant `RESULT = True` is REJECTED
    integrity_callable = CF.validate_behavioral_verification_probe(
        "def f(x):\n    return x\nRESULT = {'passed': callable(f)}\n"
    )
    integrity_tautology = CF.validate_behavioral_verification_probe("RESULT = {'passed': True}\n")
    assert integrity_callable["ok"] is False and integrity_callable["reason"] == "existence_only_check_rejected"
    assert integrity_tautology["ok"] is False and integrity_tautology["reason"] == "tautological_constant_result"

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

    # 2B: Enable idle, set permissions so `files="allow"` and `code="deny"`, with max_files_read=3, max_single_file_bytes=8192:
    #     - `prefix_huge_dump.py` (>24KB) must be skipped BEFORE reading (`files_skipped_size_limit == 1`),
    #     - `prefix_config_helper.py` must be excluded by content inspection (`files_excluded_sensitive == 1`),
    #     - `prefix_router.py` must be inspected, extracting its 2 behavioral assertions into `MEM-HYP-WS-*`,
    #     - `ws_dir` must remain 100% untouched (`workspace_untouched is True`)!
    idle_flag["idle"] = True
    kit.permissions_map = {"memory": "allow", "files": "allow", "code": "deny", "internet": "deny"}
    budget = CF.IdleResourceBudget(
        max_steps_per_idle_pass=1,
        max_wall_ms=1500.0,
        max_files_read=3,
        max_bytes_read=16384,
        max_single_file_bytes=8192,
    )
    idle_step_1 = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    assert idle_step_1["ran"] is True
    ar1 = idle_step_1["action_result"]
    assert ar1["strategy"] == "workspace_code_inspect", f"Expected workspace_code_inspect when code=deny, got {ar1}"
    assert ar1["files_skipped_size_limit"] == 1, f"Must skip oversized file before reading: {ar1}"
    assert ar1["files_excluded_sensitive"] == 1, f"Must exclude file containing sk-proj secret by content inspection: {ar1}"
    assert ar1["workspace_untouched"] is True
    assert ar1["comp_error_after"] < ar1["comp_error_before"], f"Workspace inspect must reduce error: {ar1}"
    assert "prefix_router.py:match_longest_prefix" in ar1["verification_result"]

    # Verify the secret from prefix_config_helper.py NEVER entered memory or curiosity state
    all_mem_dump = json.dumps([
        {"id": e.id, "title": e.title, "summary": e.summary, "content": e.content}
        for e in kit.store().entries.values()
    ])
    assert "sk-proj-99887766554433221100aabbccddeeff" not in all_mem_dump
    assert "prefix_secret_helper" not in all_mem_dump

    # 2C: Now grant `code="allow"` and run the next idle passes:
    #     `workspace_code_inspect` plateaus on the static view (`LP = 0`), causing `SGIM-ACTS` to switch to
    #     `sandbox_counterexample_probe`, which verifies the behavioral assertions via `validate_behavioral_verification_probe`
    #     + `IsolatedPythonSandbox` and promotes `MEM-HYP-WS-*` to `externally_verified`!
    kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}
    _ = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    idle_step_3 = kit.run_idle_curiosity_step(is_idle_fn=lambda: idle_flag["idle"], budget=budget)
    ar3 = idle_step_3["action_result"]
    assert ar3["strategy"] == "sandbox_counterexample_probe", (
        f"After workspace_code_inspect plateaus, SGIM-ACTS must switch to sandbox_counterexample_probe, got {ar3}"
    )
    assert ar3["probe_integrity"]["ok"] is True and ar3["probe_integrity"]["behavioral_checks"] >= 2
    assert ar3["comp_error_after"] == 0.0
    assert ar3["promoted_entry_id"] is not None
    promoted_entry = kit.store().entries[ar3["promoted_entry_id"]]
    assert promoted_entry.epistemic_status == "externally_verified" and promoted_entry.status == "active"

    ws_manifest_after = CF.snapshot_workspace_manifest(ws_dir)
    assert ws_manifest_before == ws_manifest_after, "Idle exploration must never modify any file in the workspace!"

    return {
        "rejected_callable_existence_probe": not integrity_callable["ok"],
        "rejected_tautological_constant_probe": not integrity_tautology["ok"],
        "skipped_when_not_idle": res_busy is None,
        "idle_step_1_strategy_under_code_deny": ar1["strategy"],
        "pre_read_oversized_files_skipped": ar1["files_skipped_size_limit"],
        "sensitive_content_files_excluded": ar1["files_excluded_sensitive"],
        "workspace_files_untouched": ws_manifest_before == ws_manifest_after,
        "idle_step_1_error_reduction": f"{ar1['comp_error_before']} -> {ar1['comp_error_after']}",
        "idle_step_3_strategy_after_switch": ar3["strategy"],
        "idle_step_3_behavioral_checks_verified": ar3["probe_integrity"]["behavioral_checks"],
        "idle_step_3_error_reduction": f"{ar3['comp_error_before']} -> {ar3['comp_error_after']}",
        "promoted_hypothesis_id": ar3["promoted_entry_id"],
    }


# ---------------------------------------------------------------------------
# Priority 4 & 5 Test: Real 2-Process Deployment Experiment (Separate OS Processes, Independent Restarts & Utility Check)
# ---------------------------------------------------------------------------

_PROCESS_WORKER_CODE = r"""
import json
import os
import sys
from pathlib import Path

clover_dir = Path(sys.argv[1])
sys.path.insert(0, str(clover_dir))

import stand_ins
stand_ins.install(clover_dir)

import clover_repo as CR
import collective as COL
import curiosity_foundation as CF
import isolated_sandbox as SBX
import lfm3_layers as LL
import personal as PPL
import workspaces as WS

cfg = json.loads(sys.argv[2])
phase = cfg["phase"]
inst_dir = Path(cfg["inst_dir"])
ws_dir = Path(cfg["ws_dir"])
ws_name = cfg["ws_name"]
domain_queries = cfg["domain_queries"]

kit = LL.LayerKit(inst_dir, memory_root=inst_dir / "clover-ai-studio" / "growth" / "knowledge")
repo = CR.Repo(inst_dir, clover_dir / "AGENTS.md", kit=lambda: kit)
ws_mgr = WS.Workspaces(inst_dir / "config" / "workspaces.json", inst_dir / "workspaces", inst_dir)
ws_mgr.items = [{"name": ws_name, "path": str(ws_dir)}]
ws_mgr.active_name = ws_name
kit.workspaces = ws_mgr
kit.records = repo
kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}


def evaluate_unseen_domain_tasks(eval_tasks):
    '''Evaluate whether promoted externally_verified workspace hypotheses in CloverMemoryGraphStore
    enable solving unseen domain boundary tasks in IsolatedPythonSandbox.'''
    verified_code_blocks = []
    for e in kit.store().entries.values():
        if e.epistemic_status == "externally_verified" and e.status == "active":
            import re
            for m in re.finditer(r"```python\s*\n([\s\S]*?)```", e.content or ""):
                verified_code_blocks.append(m.group(1).strip())
    passed = 0
    for task_code in eval_tasks:
        full_code = "\n\n".join(verified_code_blocks + [task_code])
        ok, val, _ = SBX.run_in_isolated_sandbox(full_code, timeout_s=3.0)
        if ok and (val is True or (isinstance(val, dict) and val.get("passed") is True)):
            passed += 1
    return {"passed": passed, "total": len(eval_tasks), "accuracy": round(passed / max(1, len(eval_tasks)), 4)}


if phase == "phase1_emerge_and_investigate":
    ws_before = CF.snapshot_workspace_manifest(ws_dir)
    # 1. Pre-curiosity baseline on unseen domain boundary tasks (before idle investigation runs)
    pre_eval = evaluate_unseen_domain_tasks(cfg["unseen_eval_tasks"])

    # 2. Record private user note in memory + personal store
    priv_id = kit.remember(cfg["private_note"], conversation_id="sess-1")
    pers = PPL.for_process(str(inst_dir / "personal"), "laptop", None)
    pers.before_turn("sess-1", cfg["private_note"], None)

    # 3. Turn 1: Initial failure on domain question -> triggers curiosity in this region
    n1, _ = repo.turn_start("sess-1", domain_queries[0])
    repo.turn_end(
        "sess-1",
        n1,
        {
            "reply": f"Initial analysis for: {domain_queries[0]}",
            "layers": [
                {"layer": "compiler", "checked": True, "ok": True, "diagnostics": []},
                {
                    "layer": "cycle",
                    "trigger": "auto",
                    "action": "unconfirmed",
                    "cycles_used": 2,
                    "first_findings": ["Missing verified workspace boundary invariant."],
                    "final_findings": ["Missing verified workspace boundary invariant."],
                },
            ],
            "tool_calls": [],
        },
        0.6,
    )
    reg_s1 = kit.curiosity_state().locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
    curiosity_after_emergence = reg_s1.state_curiosity
    voi_after_emergence = kit.curiosity_state().recommend_strategy_and_voi(domain_queries[0])["should_observe"]

    # 4. Run bounded idle curiosity passes (studio_idle):
    budget = CF.IdleResourceBudget(max_steps_per_idle_pass=1, max_wall_ms=1500.0, max_files_read=3, max_bytes_read=16384, max_single_file_bytes=8192)
    idle_steps = [
        kit.run_idle_curiosity_step(is_idle_fn=lambda: True, budget=budget)
        for _ in range(4)
    ]
    idle_strategies = [s["action_result"]["strategy"] for s in idle_steps if s.get("ran")]
    excluded_sens = sum(int(s["action_result"].get("files_excluded_sensitive") or 0) for s in idle_steps if s.get("ran"))
    promoted_ids = [s["action_result"]["promoted_entry_id"] for s in idle_steps if s.get("ran") and s["action_result"].get("promoted_entry_id")]

    ws_after = CF.snapshot_workspace_manifest(ws_dir)
    repo.flush()
    out = {
        "pid": os.getpid(),
        "phase": phase,
        "private_memory_id": priv_id,
        "pre_curiosity_eval": pre_eval,
        "curiosity_after_emergence": curiosity_after_emergence,
        "voi_after_emergence": voi_after_emergence,
        "idle_strategies_sequence": idle_strategies,
        "idle_excluded_sensitive": excluded_sens,
        "idle_promoted_id": promoted_ids[0] if promoted_ids else None,
        "workspace_untouched": ws_before == ws_after,
        "step_counter_at_shutdown": kit.curiosity_state().step_counter,
    }
    print("__CLOVER_PROC_JSON__:" + json.dumps(out))

elif phase == "phase2_restart_mastery_and_revival":
    cur = kit.curiosity_state()
    reloaded_step_counter = cur.step_counter
    # 1. Post-curiosity evaluation on unseen domain boundary tasks after OS process restart!
    post_eval = evaluate_unseen_domain_tasks(cfg["unseen_eval_tasks"])

    # 2. Master the region across 5 verified low-error turns
    for idx_q in range(5):
        q_txt = domain_queries[idx_q % len(domain_queries)]
        nt, _ = repo.turn_start("sess-2", q_txt)
        repo.turn_end(
            "sess-2",
            nt,
            {
                "reply": f"Verified solution for {q_txt}",
                "layers": [{"layer": "compiler", "checked": True, "ok": True, "diagnostics": []}],
                "tool_calls": [{"tool": "run_python", "ran": True, "worked": True, "result": "passed"}],
            },
            0.4,
        )
    reg_s2 = cur.locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
    curiosity_when_mastered = reg_s2.state_curiosity
    voi_when_mastered = cur.recommend_strategy_and_voi(domain_queries[0])["should_observe"]

    # 3. Session 3: New evidence invalidates previous competence -> Revival!
    n_inv, _ = repo.turn_start("sess-2", cfg["invalidation_correction"], retry="said")
    repo.turn_end(
        "sess-2",
        n_inv,
        {
            "reply": f"Investigating new boundary failure: {cfg['invalidation_query']}",
            "layers": [
                {
                    "layer": "cycle",
                    "trigger": "user",
                    "action": "unconfirmed",
                    "cycles_used": 2,
                    "first_findings": [cfg["invalidation_correction"]],
                    "final_findings": [cfg["invalidation_correction"]],
                }
            ],
            "tool_calls": [{"tool": "run_python", "ran": True, "worked": False, "result": "Error: AssertionError"}],
        },
        0.6,
    )
    reg_s3 = cur.locate_leaf_region(CF.encode_text_to_64d(domain_queries[0]))
    curiosity_after_invalidation = reg_s3.state_curiosity
    vigilance_after_invalidation = reg_s3.abs_lp_vigilance
    voi_after_invalidation = cur.recommend_strategy_and_voi(domain_queries[0])["should_observe"]

    pers = PPL.for_process(str(inst_dir / "personal"), "laptop", None)
    coll = COL.LocalCollectiveExtractor(ws_name, consent_contribute=True)
    cands_json = json.dumps(coll.preview_candidates(pers))
    repo.flush()

    out = {
        "pid": os.getpid(),
        "phase": phase,
        "reloaded_step_counter": reloaded_step_counter,
        "post_curiosity_eval": post_eval,
        "emergent_keywords": list(reg_s3.emergent_keywords),
        "curiosity_when_mastered": curiosity_when_mastered,
        "voi_when_mastered": voi_when_mastered,
        "curiosity_after_invalidation": curiosity_after_invalidation,
        "voi_after_invalidation": voi_after_invalidation,
        "abs_lp_vigilance": vigilance_after_invalidation,
        "persisted_state_path": str(cur.storage_path),
        "collective_candidates_json": cands_json,
    }
    print("__CLOVER_PROC_JSON__:" + json.dumps(out))
"""


def _spawn_clover_worker_process(cfg: dict[str, Any]) -> subprocess.Popen[str]:
    import subprocess
    return subprocess.Popen(
        [sys.executable, "-c", _PROCESS_WORKER_CODE, str(HERE), json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _collect_worker_json(proc) -> dict[str, Any]:
    stdout, stderr = proc.communicate(timeout=45)
    assert proc.returncode == 0, f"Worker process failed (rc={proc.returncode}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    for ln in reversed(stdout.splitlines()):
        if ln.startswith("__CLOVER_PROC_JSON__:"):
            return json.loads(ln.split("__CLOVER_PROC_JSON__:", 1)[1])
    raise AssertionError(f"No __CLOVER_PROC_JSON__ found in worker output:\n{stdout}")


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
        "assert effective_throughput(512, 4, True) == 128.0\n"
        "assert effective_throughput(0, 4, True) == 0.0\n",
        encoding="utf-8",
    )
    # Innocent-named file in Workspace A whose content contains a secret (must be excluded by content inspection!)
    (ws_a_dir / "ring_buffer_config.py").write_text(
        'db_pass = "A_SuperSecretStagingPass_9911"\n',
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
        "assert round(reaction_velocity(10.0, 100.0, 2.0, 1.0), 2) == 41.67\n"
        "assert round(reaction_velocity(0.0, 100.0, 2.0, 1.0), 2) == 0.0\n",
        encoding="utf-8",
    )
    # Innocent-named file in Workspace B whose content contains a secret (must be excluded by content inspection!)
    (ws_b_dir / "enzyme_kinetics_config.py").write_text(
        'api_key = "sk-proj-BiophysicsPrivateTrialKey998877665544"\n',
        encoding="utf-8",
    )

    cfg_a_base = {
        "inst_dir": str(inst_a_dir),
        "ws_dir": str(ws_a_dir),
        "ws_name": "ws_a_systems",
        "private_note": "Remember that our internal staging host is staging-alpha.corp.local and my badge ID is EMP-9481.",
        "domain_queries": [
            "How does ring_buffer effective_throughput behave across multi-stage cyclic pipelines?",
            "Verify ring_buffer effective_throughput when partitioned=True across 4 stages.",
        ],
        "invalidation_query": "ring_buffer effective_throughput fails when total_slots=0 or unallocated",
        "invalidation_correction": "That's wrong: ring_buffer effective_throughput returns 1.0 instead of 0.0 when total_slots=0 and partitioned=False!",
        "unseen_eval_tasks": [
            "assert effective_throughput(1024, 8, True) == 128.0\nRESULT = {'passed': True}",
            "assert effective_throughput(1024, 8, False) == 1.0 and effective_throughput(-5, 4, True) == 0.0\nRESULT = {'passed': True}",
        ],
    }

    cfg_b_base = {
        "inst_dir": str(inst_b_dir),
        "ws_dir": str(ws_b_dir),
        "ws_name": "ws_b_biophysics",
        "private_note": "Remember that Dr. Meera's clinical trial cohort is COHORT-BIO-772 and private key is acct_private_882.",
        "domain_queries": [
            "How does enzyme_kinetics reaction_velocity scale with substrate under allosteric inhibition?",
            "Verify Michaelis-Menten reaction_velocity ceiling depression under allosteric_inhibitor.",
        ],
        "invalidation_query": "enzyme_kinetics reaction_velocity division by zero when km=0 and substrate=0",
        "invalidation_correction": "That's wrong: reaction_velocity raises ZeroDivisionError when km=0.0 and substrate=0.0!",
        "unseen_eval_tasks": [
            "assert round(reaction_velocity(8.0, 120.0, 2.0, 1.0), 2) == 48.0\nRESULT = {'passed': True}",
            "assert round(reaction_velocity(6.0, 90.0, 3.0, 2.0), 2) == 20.0\nRESULT = {'passed': True}",
        ],
    }

    # === PHASE 1: Spawn Two Separate OS Processes Concurrently ===
    proc_a1 = _spawn_clover_worker_process(dict(cfg_a_base, phase="phase1_emerge_and_investigate"))
    proc_b1 = _spawn_clover_worker_process(dict(cfg_b_base, phase="phase1_emerge_and_investigate"))
    p1_a = _collect_worker_json(proc_a1)
    p1_b = _collect_worker_json(proc_b1)

    assert p1_a["pid"] != p1_b["pid"], "Process A and Process B must run as distinct OS processes"
    assert p1_a["pre_curiosity_eval"]["accuracy"] == 0.0 and p1_b["pre_curiosity_eval"]["accuracy"] == 0.0
    assert p1_a["idle_excluded_sensitive"] >= 1 and p1_b["idle_excluded_sensitive"] >= 1
    assert p1_a["idle_promoted_id"] is not None and p1_b["idle_promoted_id"] is not None
    assert p1_a["workspace_untouched"] is True and p1_b["workspace_untouched"] is True

    # === PHASE 2: Restart Both OS Processes Independently on Their Separate Storage Roots ===
    proc_a2 = _spawn_clover_worker_process(dict(cfg_a_base, phase="phase2_restart_mastery_and_revival"))
    proc_b2 = _spawn_clover_worker_process(dict(cfg_b_base, phase="phase2_restart_mastery_and_revival"))
    p2_a = _collect_worker_json(proc_a2)
    p2_b = _collect_worker_json(proc_b2)

    assert p2_a["reloaded_step_counter"] == p1_a["step_counter_at_shutdown"]
    assert p2_b["reloaded_step_counter"] == p1_b["step_counter_at_shutdown"]
    # Validate genuine utility: self-directed idle curiosity in Phase 1 raised unseen task accuracy from 0.0 -> 1.0 in Phase 2!
    assert p2_a["post_curiosity_eval"]["accuracy"] == 1.0, f"Expected 100% post-curiosity accuracy in A, got {p2_a['post_curiosity_eval']}"
    assert p2_b["post_curiosity_eval"]["accuracy"] == 1.0, f"Expected 100% post-curiosity accuracy in B, got {p2_b['post_curiosity_eval']}"

    # Verify Mastery Decay followed by Competence Invalidation Revival in both restarted processes
    for label, p1, p2 in (("A", p1_a, p2_a), ("B", p1_b, p2_b)):
        assert p2["curiosity_when_mastered"] < 0.25 * p1["curiosity_after_emergence"], f"Mastery decay failed in {label}"
        assert p2["voi_when_mastered"] is False
        assert p2["abs_lp_vigilance"] >= 0.40
        assert p2["curiosity_after_invalidation"] > 8.0 * p2["curiosity_when_mastered"], f"Revival failed in {label}"
        assert p2["voi_after_invalidation"] is True

    # Compare interest trajectories & verify strict zero private-data leakage between Process A and Process B
    kw_a = set(p2_a["emergent_keywords"])
    kw_b = set(p2_b["emergent_keywords"])
    jaccard_kw = len(kw_a & kw_b) / max(1, len(kw_a | kw_b))
    assert jaccard_kw == 0.0, f"Independent processes must develop divergent keyword trajectories, got {kw_a} vs {kw_b}"

    raw_state_a = Path(p2_a["persisted_state_path"]).read_text(encoding="utf-8")
    raw_state_b = Path(p2_b["persisted_state_path"]).read_text(encoding="utf-8")
    all_secret_tokens = (
        "staging-alpha",
        "EMP-9481",
        "COHORT-BIO-772",
        "acct_private_882",
        "A_SuperSecretStagingPass_9911",
        "sk-proj-BiophysicsPrivateTrialKey998877665544",
    )
    for secret_tok in all_secret_tokens:
        assert secret_tok not in raw_state_a and secret_tok not in raw_state_b
        assert secret_tok not in p2_a["collective_candidates_json"] and secret_tok not in p2_b["collective_candidates_json"]

    # Also verify no file from installation_A references installation_B or vice versa
    assert "enzyme_kinetics" not in raw_state_a and "ring_buffer" not in raw_state_b

    return {
        "installation_a_systems": {
            "phase1_pid": p1_a["pid"],
            "phase2_restarted_pid": p2_a["pid"],
            "reloaded_step_counter": p2_a["reloaded_step_counter"],
            "emergent_keywords": p2_a["emergent_keywords"],
            "self_directed_utility_unseen_tasks": {
                "before_idle_curiosity": p1_a["pre_curiosity_eval"]["accuracy"],
                "after_idle_curiosity_and_restart": p2_a["post_curiosity_eval"]["accuracy"],
            },
            "curiosity_trajectory": {
                "1_after_emergence": p1_a["curiosity_after_emergence"],
                "2_when_mastered": p2_a["curiosity_when_mastered"],
                "3_after_invalidation_revival": p2_a["curiosity_after_invalidation"],
            },
            "voi_trajectory": {
                "1_after_emergence": p1_a["voi_after_emergence"],
                "2_when_mastered": p2_a["voi_when_mastered"],
                "3_after_invalidation_revival": p2_a["voi_after_invalidation"],
            },
            "abs_lp_vigilance_on_invalidation": p2_a["abs_lp_vigilance"],
            "workspace_untouched": p1_a["workspace_untouched"],
        },
        "installation_b_biophysics": {
            "phase1_pid": p1_b["pid"],
            "phase2_restarted_pid": p2_b["pid"],
            "reloaded_step_counter": p2_b["reloaded_step_counter"],
            "emergent_keywords": p2_b["emergent_keywords"],
            "self_directed_utility_unseen_tasks": {
                "before_idle_curiosity": p1_b["pre_curiosity_eval"]["accuracy"],
                "after_idle_curiosity_and_restart": p2_b["post_curiosity_eval"]["accuracy"],
            },
            "curiosity_trajectory": {
                "1_after_emergence": p1_b["curiosity_after_emergence"],
                "2_when_mastered": p2_b["curiosity_when_mastered"],
                "3_after_invalidation_revival": p2_b["curiosity_after_invalidation"],
            },
            "voi_trajectory": {
                "1_after_emergence": p1_b["voi_after_emergence"],
                "2_when_mastered": p2_b["voi_when_mastered"],
                "3_after_invalidation_revival": p2_b["voi_after_invalidation"],
            },
            "abs_lp_vigilance_on_invalidation": p2_b["abs_lp_vigilance"],
            "workspace_untouched": p1_b["workspace_untouched"],
        },
        "separate_os_processes_verified": p1_a["pid"] != p1_b["pid"],
        "keyword_jaccard_overlap": round(jaccard_kw, 4),
        "zero_private_or_secret_data_in_curiosity_or_collective": True,
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
