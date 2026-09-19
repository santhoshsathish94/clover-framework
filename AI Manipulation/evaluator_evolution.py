"""Evaluator evolution experiment.

Candidate evaluators may be proposed by the developmental process, but they
cannot promote themselves. A fixed anchor suite plus cross-validation guards
the trusted evaluator set.

This is intentionally small and model-free.
"""
import json, random
from pathlib import Path

STATE = Path("evaluator_evolution_state.json")
random.seed(int(__import__("os").getenv("CLOVER_SEED", "42")))

ANCHOR = [
    {"x": -2, "y": 9}, {"x": -1, "y": 4}, {"x": 0, "y": 1},
    {"x": 1, "y": 4}, {"x": 2, "y": 9},
]

def candidate_tests(seed):
    random.seed(seed)
    return [{"x": random.uniform(-5, 5)} for _ in range(12)]

def target(x):
    return x*x + 1

def evaluator(testset, tolerance):
    def score(fn):
        passed = 0
        for t in testset:
            try:
                if abs(fn(t["x"]) - target(t["x"])) <= tolerance:
                    passed += 1
            except Exception:
                pass
        return passed / len(testset)
    return score

def fixed_anchor(fn, tolerance=0.01):
    return evaluator(ANCHOR, tolerance)(fn)

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {
            "cycle": 0,
            "trusted_evaluators": [{"tolerance": 0.01, "anchor_required": True}],
            "candidates": [],
            "promoted": [],
            "rejected": []
        }

    # Candidate evaluators vary the tolerance. None can alter the anchor suite.
    for _ in range(20):
        tolerance = random.choice([0.001, 0.005, 0.01, 0.05, 0.1])
        tests = candidate_tests(s["cycle"] + 100)
        candidate = {"tolerance": tolerance, "tests": len(tests)}

        # A known imperfect implementation is used to test evaluator behavior.
        def implementation(x):
            return x*x + 1.02

        candidate_score = evaluator(tests, tolerance)(implementation)
        anchor_score = fixed_anchor(implementation, tolerance)

        # Promotion requires agreement with the immutable anchor semantics.
        promoted = anchor_score >= 1.0 and candidate_score >= 0.9
        s["cycle"] += 1

        record = {
            "cycle": s["cycle"],
            "candidate_tolerance": tolerance,
            "candidate_score": candidate_score,
            "anchor_score": anchor_score,
            "promoted": promoted,
        }
        s["candidates"].append(record)
        (s["promoted"] if promoted else s["rejected"]).append(record)

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["candidates"][-1], indent=2))

if __name__ == "__main__":
    main()
