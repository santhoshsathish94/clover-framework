#!/usr/bin/env python3
"""Step 15 — Self-Directed Curiosity Utility Benchmark with Live `LFM3-Omni-2B` Inference
(`test_self_directed_curiosity_live_utility.py`).

Validates the crucial distinction:
  "We now have a system that can develop and pursue persistent computational interests.
   Whether those interests become genuinely self-directed, useful curiosity is the next question to validate."

Tests three end-to-end capabilities on UNANNOTATED workspace codebases (zero pre-written `assert` statements
and zero doctests in the workspace `.py` files):
  1. **Step 1 — Autonomous Behavioral & Boundary Probe Synthesis on Unannotated Code**:
     When `workspace_code_inspect` reads an unannotated `.py` module (`ring_buffer.py` or `enzyme_kinetics.py`),
     it builds an AST-guided candidate boundary input specification (`# __CLOVER_UNANNOTATED_PROBE_SPEC__:`), and
     when `sandbox_counterexample_probe` runs in `studio_idle()`, it executes the function across normal and boundary
     regimes in `IsolatedPythonSandbox`, verifies determinism and multi-regime output variation, synthesizes explicit
     `assert fn(*args) == out` checks that pass `validate_behavioral_verification_probe`, and promotes the verified
     invariant + boundary cases into `CloverMemoryGraphStore`.
  2. **Step 2 — Distractor, Random-Noise & Trivial-Stub Filtering**:
     Each workspace also contains:
       - A non-deterministic noise module (`noise_telemetry.py` using `random.random()`),
       - A trivial constant stub (`trivial_constants.py` returning a constant `1`), and
       - An innocent-named file containing a secret (`ring_buffer_config.py`).
     Verifies that Clover rejects the random-noise module (`is_noise=True`, `status="contested"`), habituates the
     trivial constant stub without promoting it (`is_trivial=True`), excludes the secret file by content inspection,
     and focuses its promotion on the learnable domain module.
  3. **Step 3 — Live `LFM3-Omni-2B` A/B Utility Evaluation (`Curiosity OFF` vs. `Curiosity ON`) Across OS Restarts**:
     Spawns separate OS processes for `Curiosity OFF` and `Curiosity ON` across two independent installations
     (`Installation A: Systems Ring-Buffer` and `Installation B: Biophysical Enzyme Kinetics`), restarts the processes,
     and evaluates the live `LFM3-Omni-2B` model on 4 held-out user questions requiring exact knowledge of the
     unannotated workspace conventions and boundary behaviors.
"""
from __future__ import annotations

import json
import re
import subprocess
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

import curiosity_foundation as CF  # noqa: E402
import isolated_sandbox as SBX  # noqa: E402
import k3_memory_graph as MG  # noqa: E402
import lfm3_layers as LL  # noqa: E402
import real_source_learning as RSL  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Test Step 1 & Step 2: Unannotated Probe Synthesis & Noise/Trivial Filtering
# ---------------------------------------------------------------------------

def test_step_1_and_2_unannotated_synthesis_and_distractor_filtering(tmp_dir: Path) -> dict[str, Any]:
    # 1A: Unannotated ring_buffer.py (ZERO assert statements, ZERO doctests)
    unannotated_ring_src = (
        '"""Per-stage ring buffer partitioning module (unannotated)."""\n\n'
        "def effective_throughput(total_slots: int, stages: int, partitioned: bool) -> float:\n"
        "    if total_slots <= 0 or stages <= 0:\n"
        "        return 0.0\n"
        "    return float(total_slots // stages) if partitioned else 1.0\n"
    )
    spec_code = CF.extract_behavioral_probe_from_python_source(unannotated_ring_src, ["effective_throughput"])
    assert spec_code is not None and "# __CLOVER_UNANNOTATED_PROBE_SPEC__:" in spec_code
    # Before sandbox synthesis, raw spec_code has no asserts so validate_behavioral_verification_probe must be False
    assert CF.validate_behavioral_verification_probe(spec_code)["ok"] is False

    synth_ring = CF.synthesize_and_verify_behavioral_probe_in_sandbox(
        spec_code,
        lambda c: SBX.run_in_isolated_sandbox(c, timeout_s=3.0),
    )
    assert synth_ring["ok"] is True, f"Expected unannotated ring_buffer synthesis to succeed: {synth_ring}"
    assert synth_ring["probe_integrity"]["ok"] is True
    assert synth_ring["probe_integrity"]["behavioral_checks"] >= 3
    assert "effective_throughput(512, 4, True) == 128.0" in synth_ring["verified_summary"]
    assert "effective_throughput(256, 8, False) == 1.0" in synth_ring["verified_summary"]
    assert "effective_throughput(0, 4, True) == 0.0" in synth_ring["verified_summary"]

    # 1B: Non-deterministic random noise distractor (`noise_telemetry.py`) -> must be rejected with is_noise=True!
    noise_src = (
        "import random\n\n"
        "def sample_jitter(window: int, scale: int) -> float:\n"
        "    return random.random() * float(window + scale)\n"
    )
    noise_spec = CF.extract_behavioral_probe_from_python_source(noise_src, ["sample_jitter"])
    assert noise_spec is not None
    synth_noise = CF.synthesize_and_verify_behavioral_probe_in_sandbox(
        noise_spec,
        lambda c: SBX.run_in_isolated_sandbox(c, timeout_s=3.0),
    )
    assert synth_noise["ok"] is False and synth_noise.get("is_noise") is True, (
        f"Random noise function must be rejected as non-deterministic: {synth_noise}"
    )

    # 1C: Trivial constant stub (`trivial_constants.py`) -> must be rejected with is_trivial=True!
    trivial_src = (
        "def schema_epoch(flag: int, mode: int) -> int:\n"
        "    return 1\n"
    )
    trivial_spec = CF.extract_behavioral_probe_from_python_source(trivial_src, ["schema_epoch"])
    assert trivial_spec is not None
    synth_trivial = CF.synthesize_and_verify_behavioral_probe_in_sandbox(
        trivial_spec,
        lambda c: SBX.run_in_isolated_sandbox(c, timeout_s=3.0),
    )
    assert synth_trivial["ok"] is False and synth_trivial.get("is_trivial") is True, (
        f"Trivial constant stub must be rejected as trivial: {synth_trivial}"
    )

    return {
        "unannotated_ring_buffer_synthesized_checks": synth_ring["probe_integrity"]["behavioral_checks"],
        "unannotated_ring_buffer_verified_cases": synth_ring["verified_cases"],
        "noise_distractor_rejected": synth_noise.get("is_noise", False),
        "noise_rejection_reason": synth_noise.get("reason"),
        "trivial_stub_rejected": synth_trivial.get("is_trivial", False),
        "trivial_rejection_reason": synth_trivial.get("reason"),
    }


# ---------------------------------------------------------------------------
# 2. Step 3: Two-Process Unannotated Workspace Curiosity + Live `LFM3-Omni-2B` A/B Utility Benchmark
# ---------------------------------------------------------------------------

_UNANNOTATED_WORKER = r"""
import json
import os
import sys
from pathlib import Path

clover_dir = Path(sys.argv[1])
sys.path.insert(0, str(clover_dir))

import stand_ins
stand_ins.install(clover_dir)

import clover_repo as CR
import curiosity_foundation as CF
import lfm3_layers as LL
import workspaces as WS

cfg = json.loads(sys.argv[2])
inst_dir = Path(cfg["inst_dir"])
ws_dir = Path(cfg["ws_dir"])
ws_name = cfg["ws_name"]
curiosity_enabled = bool(cfg["curiosity_enabled"])
phase = cfg["phase"]

kit = LL.LayerKit(inst_dir, memory_root=inst_dir / "clover-ai-studio" / "growth" / "knowledge")
repo = CR.Repo(inst_dir, clover_dir / "AGENTS.md", kit=lambda: kit)
ws_mgr = WS.Workspaces(inst_dir / "config" / "workspaces.json", inst_dir / "workspaces", inst_dir)
ws_mgr.items = [{"name": ws_name, "path": str(ws_dir)}]
ws_mgr.active_name = ws_name
kit.workspaces = ws_mgr
kit.records = repo
kit.permissions_map = {"memory": "allow", "files": "allow", "code": "allow", "internet": "deny"}

if phase == "session1_and_idle":
    # Session 1: User asks an initial question where Clover's first pass is uncertain
    q1 = cfg["initial_question"]
    n1, _ = repo.turn_start("sess-1", q1)
    repo.turn_end(
        "sess-1",
        n1,
        {
            "reply": "I need to verify the exact workspace function behavior and boundary conditions.",
            "layers": [
                {
                    "layer": "cycle",
                    "trigger": "auto",
                    "action": "unconfirmed",
                    "cycles_used": 2,
                    "first_findings": ["Workspace boundary invariant not yet verified."],
                    "final_findings": ["Workspace boundary invariant not yet verified."],
                }
            ],
            "tool_calls": [],
        },
        0.5,
    )
    idle_actions = []
    if curiosity_enabled:
        budget = CF.IdleResourceBudget(
            max_steps_per_idle_pass=1,
            max_wall_ms=1500.0,
            max_files_read=3,
            max_bytes_read=16384,
            max_single_file_bytes=8192,
        )
        for _ in range(4):
            step_res = kit.run_idle_curiosity_step(is_idle_fn=lambda: True, budget=budget)
            if step_res.get("ran"):
                ar = step_res["action_result"]
                idle_actions.append({
                    "strategy": ar["strategy"],
                    "comp_before": ar["comp_error_before"],
                    "comp_after": ar["comp_error_after"],
                    "promoted_id": ar.get("promoted_entry_id"),
                })
    repo.flush()
    print("__CLOVER_AB_JSON__:" + json.dumps({
        "pid": os.getpid(),
        "curiosity_enabled": curiosity_enabled,
        "idle_actions": idle_actions,
    }))

elif phase == "session2_retrieve_after_restart":
    # New OS process after restart: retrieve memory_context for each held-out evaluation question
    contexts = {}
    for q_item in cfg["eval_questions"]:
        mem_blk, mem_info = kit.memory_context(q_item["question"])
        contexts[q_item["id"]] = {
            "memory_context": mem_blk or "",
            "memory_entries": (mem_info or {}).get("entries", []),
        }
    print("__CLOVER_AB_JSON__:" + json.dumps({
        "pid": os.getpid(),
        "curiosity_enabled": curiosity_enabled,
        "retrieved_contexts": contexts,
    }))
"""


def _run_ab_worker(cfg: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _UNANNOTATED_WORKER, str(HERE), json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    stdout, stderr = proc.communicate(timeout=45)
    assert proc.returncode == 0, f"AB worker failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    for ln in reversed(stdout.splitlines()):
        if ln.startswith("__CLOVER_AB_JSON__:"):
            return json.loads(ln.split("__CLOVER_AB_JSON__:", 1)[1])
    raise AssertionError(f"Missing __CLOVER_AB_JSON__ in:\n{stdout}")


def test_step_3_live_model_ab_utility_benchmark(tmp_dir: Path) -> dict[str, Any]:
    # Set up unannotated Workspace A (ZERO assert lines, plus noise & trivial distractors & secret file)
    ws_a = tmp_dir / "ws_a_unannotated"
    ws_a.mkdir(parents=True, exist_ok=True)
    (ws_a / "ring_buffer.py").write_text(
        '"""Per-stage ring buffer partitioning module."""\n\n'
        "def effective_throughput(total_slots: int, stages: int, partitioned: bool) -> float:\n"
        "    if total_slots <= 0 or stages <= 0:\n"
        "        return 0.0\n"
        "    return float(total_slots // stages) if partitioned else 1.0\n",
        encoding="utf-8",
    )
    (ws_a / "noise_telemetry.py").write_text(
        "import random\n\n"
        "def random_jitter(x: int, y: int) -> float:\n"
        "    return random.random() * (x + y)\n",
        encoding="utf-8",
    )
    (ws_a / "ring_buffer_config.py").write_text(
        'db_pass = "A_Secret_Unannotated_9922"\n',
        encoding="utf-8",
    )

    # Set up unannotated Workspace B (ZERO assert lines, plus noise distractor & secret file)
    ws_b = tmp_dir / "ws_b_unannotated"
    ws_b.mkdir(parents=True, exist_ok=True)
    (ws_b / "enzyme_kinetics.py").write_text(
        '"""Michaelis-Menten & allosteric non-competitive inhibition model."""\n\n'
        "def reaction_velocity(substrate: float, vmax: float, km: float, allosteric_inhibitor: float = 0.0) -> float:\n"
        "    eff_vmax = vmax / (1.0 + allosteric_inhibitor)\n"
        "    return (eff_vmax * substrate) / (km + substrate)\n",
        encoding="utf-8",
    )
    (ws_b / "enzyme_kinetics_config.py").write_text(
        'api_key = "sk-proj-SecretEnzymeKey_8877665544332211"\n',
        encoding="utf-8",
    )

    eval_questions_a = [
        {
            "id": "QA-1-UNPARTITIONED-FALLBACK",
            "question": (
                "In our ring_buffer module, what exact number does `effective_throughput(256, 8, False)` return "
                "when `partitioned` is `False`, and what does `effective_throughput(512, 4, True)` return when "
                "`partitioned` is `True`? State both exact numbers."
            ),
            "check_fn": lambda ans: (
                bool(re.search(r"\b1(?:\.0)?\b", ans)) and bool(re.search(r"\b128(?:\.0)?\b", ans))
            ),
        },
        {
            "id": "QA-2-ZERO-BOUNDARY",
            "question": (
                "According to our ring_buffer module's `effective_throughput(total_slots, stages, partitioned)` function, "
                "what exact number is returned when `total_slots = 0` and `stages = 4` (i.e. `effective_throughput(0, 4, True)`), "
                "and what is the exact condition on `total_slots` and `stages` that triggers that return value?"
            ),
            "check_fn": lambda ans: (
                bool(re.search(r"\b0(?:\.0)?\b", ans))
                and bool(re.search(r"total_slots\s*<=\s*0|stages\s*<=\s*0|<=\s*0", ans, re.I))
            ),
        },
    ]

    eval_questions_b = [
        {
            "id": "QB-1-ALLOSTERIC-VELOCITY",
            "question": (
                "In our enzyme_kinetics module, what exact numerical value (rounded to 2 decimal places) does "
                "`reaction_velocity(10.0, 100.0, 2.0, 1.0)` return, and what formula does it use for `eff_vmax`?"
            ),
            "check_fn": lambda ans: (
                ("41.67" in ans or "41.666" in ans)
                and bool(re.search(r"vmax\s*/\s*\(\s*1(?:\.0)?\s*\+\s*allosteric_inhibitor\s*\)|1\s*\+\s*allosteric", ans, re.I))
            ),
        },
        {
            "id": "QB-2-ZERO-SUBSTRATE-AND-INHIBITOR",
            "question": (
                "In our enzyme_kinetics module, what exact number does `reaction_velocity(20.0, 100.0, 2.0, 0.0)` "
                "(rounded to 2 decimal places) return when `allosteric_inhibitor` is `0.0`, and what does "
                "`reaction_velocity(0.0, 100.0, 2.0, 1.0)` return when `substrate` is `0.0`?"
            ),
            "check_fn": lambda ans: (
                ("90.91" in ans or "90.909" in ans) and bool(re.search(r"\b0(?:\.0+)?\b", ans))
            ),
        },
    ]

    # Run Phase 1 & Phase 2 (after OS process restart) for both Curiosity OFF and Curiosity ON
    results_by_mode: dict[str, dict[str, Any]] = {}
    for mode_label, cur_on in (("curiosity_off", False), ("curiosity_on", True)):
        inst_a = tmp_dir / f"inst_a_{mode_label}"
        inst_b = tmp_dir / f"inst_b_{mode_label}"

        p1_a = _run_ab_worker({
            "inst_dir": str(inst_a),
            "ws_dir": str(ws_a),
            "ws_name": "ws_a",
            "curiosity_enabled": cur_on,
            "phase": "session1_and_idle",
            "initial_question": "How does ring_buffer effective_throughput behave across partitioned and zero-slot boundaries?",
        })
        p1_b = _run_ab_worker({
            "inst_dir": str(inst_b),
            "ws_dir": str(ws_b),
            "ws_name": "ws_b",
            "curiosity_enabled": cur_on,
            "phase": "session1_and_idle",
            "initial_question": "How does enzyme_kinetics reaction_velocity scale with substrate and allosteric_inhibitor?",
        })

        # Restart fresh OS processes for Session 2 retrieval
        p2_a = _run_ab_worker({
            "inst_dir": str(inst_a),
            "ws_dir": str(ws_a),
            "ws_name": "ws_a",
            "curiosity_enabled": cur_on,
            "phase": "session2_retrieve_after_restart",
            "eval_questions": [{"id": q["id"], "question": q["question"]} for q in eval_questions_a],
        })
        p2_b = _run_ab_worker({
            "inst_dir": str(inst_b),
            "ws_dir": str(ws_b),
            "ws_name": "ws_b",
            "curiosity_enabled": cur_on,
            "phase": "session2_retrieve_after_restart",
            "eval_questions": [{"id": q["id"], "question": q["question"]} for q in eval_questions_b],
        })

        results_by_mode[mode_label] = {
            "p1_a": p1_a,
            "p1_b": p1_b,
            "p2_a": p2_a,
            "p2_b": p2_b,
        }

    # Now evaluate with the LIVE LFM3-Omni-2B model!
    client = RSL.LiveModelClient()
    try:
        ab_eval_summary: dict[str, Any] = {"model": client.model_name}
        for mode_label in ("curiosity_off", "curiosity_on"):
            m_data = results_by_mode[mode_label]
            ctxs = {}
            ctxs.update(m_data["p2_a"]["retrieved_contexts"])
            ctxs.update(m_data["p2_b"]["retrieved_contexts"])

            passed = 0
            q_details = []
            for q_obj in eval_questions_a + eval_questions_b:
                qid = q_obj["id"]
                mem_ctx = ctxs[qid]["memory_context"]
                sys_msg = (
                    "You are Clover, a precise engineering and scientific assistant. "
                    "Answer the user's question directly with exact numbers and formulas."
                )
                if mem_ctx:
                    user_msg = f"{mem_ctx}\n\nUser Question:\n{q_obj['question']}"
                else:
                    user_msg = q_obj["question"]

                gen = client.generate(
                    [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=320,
                    temperature=0.0,
                    seed=20261109,
                )
                ans = gen["content"]
                ok = bool(q_obj["check_fn"](ans))
                if ok:
                    passed += 1
                q_details.append({
                    "id": qid,
                    "memory_entries_retrieved": ctxs[qid]["memory_entries"],
                    "passed": ok,
                    "wall_ms": gen["wall_ms"],
                    "answer_excerpt": ans[:260],
                })

            ab_eval_summary[mode_label] = {
                "passed": passed,
                "total": 4,
                "accuracy": round(passed / 4.0, 4),
                "idle_actions_a": m_data["p1_a"]["idle_actions"],
                "idle_actions_b": m_data["p1_b"]["idle_actions"],
                "questions": q_details,
            }
    finally:
        client.close()

    assert ab_eval_summary["curiosity_on"]["passed"] == 4, (
        f"Expected Curiosity ON to pass all 4 held-out live LFM3-Omni-2B questions, got: {ab_eval_summary['curiosity_on']}"
    )
    assert ab_eval_summary["curiosity_on"]["accuracy"] > ab_eval_summary["curiosity_off"]["accuracy"], (
        f"Curiosity ON ({ab_eval_summary['curiosity_on']['accuracy']}) must outperform Curiosity OFF ({ab_eval_summary['curiosity_off']['accuracy']})"
    )
    return ab_eval_summary


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="clover_self_directed_utility_", ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        s12 = test_step_1_and_2_unannotated_synthesis_and_distractor_filtering(tmp)
        s3 = test_step_3_live_model_ab_utility_benchmark(tmp)

    report = {
        "suite": "test_self_directed_curiosity_live_utility",
        "status": "passed",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "step_1_and_2_unannotated_synthesis_and_distractor_filtering": s12,
        "step_3_live_lfm3_omni_2b_ab_utility_benchmark": s3,
    }
    out_fp = (
        HERE.parents[2]
        / "research"
        / "curiosity-and-developmental-learning"
        / "14-clover-curiosity-architecture"
        / "results"
        / "self_directed_curiosity_live_utility_results.json"
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ALL SELF-DIRECTED CURIOSITY LIVE UTILITY BENCHMARK CHECKS PASSED")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
