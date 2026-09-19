"""Bounded objective-discovery layer.

Generates candidate next objectives from observable developmental state:
failures, uncertainty, capability gaps, and expected value. Objectives are
ranked by an external policy and must fit a fixed resource/safety envelope.

This is model-free and does not claim intrinsic goals or consciousness.
"""
import json, random
from pathlib import Path

STATE = Path("objective_discovery_state.json")
random.seed(int(__import__("os").getenv("CLOVER_SEED", "42")))

OBJECTIVE_TYPES = [
    "reduce_failure",
    "increase_test_coverage",
    "discover_capability",
    "improve_search",
    "verify_uncertain_knowledge",
    "compress_redundant_capabilities",
]

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "cycle": 0,
        "uncertainties": [
            {"name": "search_efficiency", "uncertainty": .8},
            {"name": "test_coverage", "uncertainty": .6},
        ],
        "capability_gaps": ["generalization", "composition"],
        "recent_failures": ["held_out_error", "invalid_candidate"],
        "history": [],
    }

def candidates(s):
    out = []
    for name, u in s["uncertainties"]:
        out.append({
            "type": "verify_uncertain_knowledge",
            "target": name,
            "value": u * 1.0,
            "cost": .5,
        })
    for gap in s["capability_gaps"]:
        out.append({
            "type": "discover_capability",
            "target": gap,
            "value": .9,
            "cost": 1.0,
        })
    for failure in s["recent_failures"]:
        out.append({
            "type": "reduce_failure",
            "target": failure,
            "value": .85,
            "cost": .7,
        })
    out.extend([
        {"type": "increase_test_coverage", "target": "held_out", "value": .7, "cost": .4},
        {"type": "improve_search", "target": "mutation_policy", "value": .65, "cost": .8},
    ])
    return out

def choose(cs, budget=1.0):
    feasible = [c for c in cs if c["cost"] <= budget]
    return max(feasible, key=lambda c: c["value"] / c["cost"])

def main():
    s = load()
    for _ in range(30):
        cs = candidates(s)
        selected = choose(cs)
        s["cycle"] += 1

        # Outcome is deliberately simulated: real deployments should connect
        # this layer to the independent evaluator and execution environment.
        success = random.random() < min(.95, selected["value"])
        if success:
            selected["value"] *= .9
        else:
            selected["value"] = min(1.0, selected["value"] + .1)

        s["history"].append({
            "cycle": s["cycle"],
            "candidate_count": len(cs),
            "selected": selected,
            "success": success,
        })

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["history"][-1], indent=2))

if __name__ == "__main__":
    main()
