"""End-to-end verification of the multi-cycle AGENTS.md Clover Cycle (`Context -> Direction -> Execution -> Outcome -> Growth`)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import clover_cycle as C  # noqa: E402
import lfm3_layers  # noqa: E402
from autonomous_clover import AutonomousCloverEngine  # noqa: E402
try:
    import clover_repo  # noqa: E402
except ImportError:
    clover_repo = None


def test_verify_turn_outcome_checks() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        kit = lfm3_layers.LayerKit(Path(tmp), memory_root=Path(tmp) / "memory_tree")

        # 1. Arithmetic equation mismatch in draft answer
        f1, m1 = C.verify_turn_outcome(
            "What is 17 * 23?",
            "First we multiply: 17 * 23 = 381. So the answer is 381.",
            rounds=[],
            kit=kit,
            layers=("compiler",),
        )
        assert f1 and any("391" in x for x in f1), f"Expected 391 finding, got {f1}"

        # 2. Direct arithmetic mismatch without written '=' equation
        f2, _ = C.verify_turn_outcome(
            "What is 144 * 17?",
            "The result is 2438.",
            rounds=[],
            kit=kit,
            layers=("compiler",),
        )
        assert f2 and any("2448" in x for x in f2), f"Expected 2448 finding, got {f2}"

        # 3. Format constraint check
        f3, _ = C.verify_turn_outcome(
            "What is 12 + 15? Solve step by step and end your response with #### <number>",
            "12 + 15 = 27. The answer is 27.",
            rounds=[],
            kit=kit,
            layers=("compiler",),
        )
        assert f3 and any("####" in x for x in f3), f"Expected #### format finding, got {f3}"

        # 4. Sandbox execution check on computational Python block
        def fake_exec_py(code: str):
            try:
                ns = {}
                exec(compile(code, "<sandbox>", "exec"), ns, ns)
                return True, "ok"
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"

        f4, m4 = C.verify_turn_outcome(
            "Write a Python snippet to verify the sum of the first 5 primes.",
            "Here is the verification:\n```python\nprimes = [2, 3, 5, 7, 11]\ntotal = sum(primes)\nassert total == 26\nprint(total)\n```",
            rounds=[],
            kit=kit,
            layers=("compiler",),
            execute_python_fn=fake_exec_py,
        )
        assert m4["python_sandbox_checked"] is True
        assert f4 and any("AssertionError" in x for x in f4), f"Expected AssertionError finding, got {f4}"

        # 5. Unrecovered failed tool check
        f5, _ = C.verify_turn_outcome(
            "Run python to compute 2**20.",
            "2**20 is 1048576.",
            rounds=[{"round": 1, "tool": "run_python", "ran": True, "worked": False, "result": "SyntaxError: invalid syntax"}],
            kit=kit,
            layers=("compiler",),
        )
        assert f5 and any("run_python" in x for x in f5), f"Expected failed tool finding, got {f5}"

        # 6. Clean verified answer passes with 0 findings
        f6, _ = C.verify_turn_outcome(
            "What is 17 * 23? Solve step by step and end your response with #### <number>",
            "17 * 23 = 391.\n#### 391",
            rounds=[],
            kit=kit,
            layers=("compiler",),
        )
        assert f6 == [], f"Expected 0 findings on clean answer, got {f6}"

    return {"status": "ok", "checks_verified": 6}


def test_multi_cycle_turn_and_growth() -> dict:
    """Verify that a 3-cycle turn automatically repairs defects across Cycle 1 -> Cycle 2 -> Cycle 3 and records LRN-* Growth."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        kit = lfm3_layers.LayerKit(tmp_path, memory_root=tmp_path / "memory_tree")
        kit.jepa_arch = "auto"
        repo = clover_repo.Repo(tmp_path, HERE / "AGENTS.md", kit=lambda: kit) if clover_repo is not None else None

        # Simulate a 3-cycle progression inside the exact outer loop of OmniNative.run_turn:
        # Cycle 1: syntax error in Python code
        # Cycle 2: syntax fixed, but arithmetic equation error in explanation (17 * 23 = 381)
        # Cycle 3: both code and arithmetic verified (17 * 23 = 391)
        drafts = [
            "```python\ndef mul(a, b)\n    return a * b\n```\n17 * 23 = 381",
            "```python\ndef mul(a, b):\n    return a * b\n```\nSo 17 * 23 = 381.",
            "```python\ndef mul(a, b):\n    return a * b\n```\nSo 17 * 23 = 391.",
        ]
        query = "Write a Python function `mul(a, b)` and state what 17 * 23 equals."
        layers = ("compiler", "c1", "c2", "c3", "memory")
        events = []
        layer_log = []
        rounds = []
        first_cycle_findings = None
        c1_first_failed = None
        max_clover_cycles = 3
        final = ""

        for attempt in range(max_clover_cycles):
            final = drafts[attempt]
            c1_cand, _ = kit.c1_infer(final)
            findings, c_meta = C.verify_turn_outcome(query, final, rounds, kit=kit, layers=layers)
            if not findings:
                if first_cycle_findings:
                    events.append({"type": "cycle", "cycle": attempt + 1, "stage": "verified", "cycles_used": attempt + 1})
                    if c1_first_failed is not None:
                        _, _ = kit.c2_plan_state(query, c1_state=c1_first_failed, offered_tools=["run_python"])
                        kit.jepa_record_outcome(c1_obs_state=c1_cand, is_baseline=False)
                    layer_log.append({
                        "layer": "cycle",
                        "trigger": "auto",
                        "action": "corrected",
                        "cycles_used": attempt + 1,
                        "first_findings": first_cycle_findings,
                        "final_findings": [],
                        "tools": [],
                        "lesson": C.lesson_text(query, [f[:240] for f in first_cycle_findings[:2]], "", passed=True),
                        "lesson_kind": "verified",
                    })
                break
            if first_cycle_findings is None:
                first_cycle_findings = list(findings)
                c1_first_failed = c1_cand
            if attempt + 1 < max_clover_cycles:
                events.append({"type": "cycle", "cycle": attempt + 2, "stage": "retry", "findings": findings})

        assert len(events) == 3, f"Expected retry(2), retry(3), verified(3), got {events}"
        assert events[0]["cycle"] == 2 and events[0]["stage"] == "retry"
        assert events[1]["cycle"] == 3 and events[1]["stage"] == "retry"
        assert events[2]["cycle"] == 3 and events[2]["stage"] == "verified" and events[2]["cycles_used"] == 3

        # Verify S1 clover_repo.Repo.turn_end (or LayerKit.learn) commits the verified lesson into LEARNINGS.md
        if repo is not None:
            repo.turn_start("sess-cycle-test", query)
            changes = repo.turn_end("sess-cycle-test", 1, {"reply": final, "layers": layer_log}, 0.12)
            lrn_id = changes[0] if changes else ""
        else:
            lrn_id, _ = kit.learn(layer_log[-1]["lesson"], sid="sess-cycle-test", n=1, kind="verified")
        assert "LRN-" in str(lrn_id), f"Expected LRN-* learning, got {lrn_id}"

    return {
        "status": "ok",
        "cycles_executed": 3,
        "events": events,
        "learning": lrn_id,
    }


def test_autonomous_clover_multi_cycle() -> dict:
    """Verify AutonomousCloverEngine repeats across 3+ cycles when Cycle 2 still has a defect."""
    with tempfile.TemporaryDirectory() as tmp:
        call_count = {"n": 0}
        target_holder = {"ans": ""}

        def three_step_generator(question: str, fix_prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Cycles 1 and 2 return an unverified draft with wrong arithmetic
                return "Let me re-evaluate: 10 + 10 = 25, so the answer is 0."
            # Cycle 3 uses the verified tool output
            return target_holder["ans"]

        engine = AutonomousCloverEngine(Path(tmp), work_dir=Path(tmp) / "work", jepa_arch="auto", c1_generate_fn=three_step_generator)
        prob = engine.synthesize_new_problem(step_idx=1, seed=2026)  # arithmetic_audit problem
        _, _, canonical_target = prob.verifier_fn("")
        target_holder["ans"] = canonical_target

        res = engine.solve_problem_autonomously(prob, allow_clover_cycle=True, max_cycles=4)
        assert res["final_passed"] is True, f"Expected final_passed=True, got {res}"
        assert res["cycles_used"] == 3, f"Expected 3 cycles used, got {res['cycles_used']}"

    return {
        "status": "ok",
        "problem_id": res["problem_id"],
        "cycles_used": res["cycles_used"],
        "final_passed": res["final_passed"],
    }


if __name__ == "__main__":
    out = {
        "verify_turn_outcome": test_verify_turn_outcome_checks(),
        "multi_cycle_turn_and_growth": test_multi_cycle_turn_and_growth(),
        "autonomous_clover_multi_cycle": test_autonomous_clover_multi_cycle(),
    }
    print("ALL CLOVER CYCLE TESTS PASSED")
    print(json.dumps(out, indent=2))
