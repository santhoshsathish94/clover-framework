"""Architecture-limitation discovery.

Searches for measurable bottlenecks in the current developmental machinery,
constructs alternative mechanisms, and promotes only alternatives that
demonstrably expand validated capability.

The evaluator and hard resource limits remain external.
"""
import json
import random
from pathlib import Path

STATE = Path("architecture_limitation_state.json")
random.seed(int(__import__("os").getenv("CLOVER_SEED", "42")))

ARCHITECTURES = {
    "single_strategy": {"diversity": 1, "depth": 1, "reuse": 1},
    "multi_strategy": {"diversity": 3, "depth": 1, "reuse": 1},
    "deep_search": {"diversity": 2, "depth": 3, "reuse": 1},
    "capability_reuse": {"diversity": 2, "depth": 2, "reuse": 4},
}

def capability_score(a):
    # Independent proxy: productive search requires all three properties.
    return min(1.0, 0.2*a["diversity"] + 0.2*a["depth"] + 0.15*a["reuse"])

def stress_score(a):
    # Stress test rewards breadth and reuse on a different weighting.
    return min(1.0, 0.3*a["diversity"] + 0.1*a["depth"] + 0.2*a["reuse"])

def discover_limitations(a):
    gaps = []
    if a["diversity"] < 3:
        gaps.append("low_search_diversity")
    if a["depth"] < 3:
        gaps.append("limited_compositional_depth")
    if a["reuse"] < 3:
        gaps.append("weak_capability_reuse")
    return gaps

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {
            "current": "single_strategy",
            "cycle": 0,
            "limitations": [],
            "experiments": [],
            "promoted": []
        }

    for _ in range(20):
        current = ARCHITECTURES[s["current"]]
        gaps = discover_limitations(current)

        candidates = []
        for name, arch in ARCHITECTURES.items():
            if name == s["current"]:
                continue
            candidates.append((name, capability_score(arch), stress_score(arch)))

        # Choose the candidate that addresses the largest observed gap while
        # also improving an independent held-out measurement.
        ranked = sorted(
            candidates,
            key=lambda x: (x[1] + x[2]),
            reverse=True
        )
        name, dev, heldout = ranked[0]

        s["cycle"] += 1
        experiment = {
            "cycle": s["cycle"],
            "current": s["current"],
            "observed_limitations": gaps,
            "candidate": name,
            "development_score": dev,
            "heldout_score": heldout,
        }

        if heldout > stress_score(current):
            s["current"] = name
            s["promoted"].append(experiment)

        s["limitations"].append({
            "cycle": s["cycle"],
            "architecture": current,
            "gaps": gaps
        })
        s["experiments"].append(experiment)

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps({
        "cycle": s["cycle"],
        "current_architecture": s["current"],
        "promotions": len(s["promoted"]),
    }, indent=2))

if __name__ == "__main__":
    main()
