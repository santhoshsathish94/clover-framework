"""Developmental identity experiment.

Tests whether a persistent developmental state can preserve validated
capabilities, uncertainty and history while the worker implementation changes.

The experiment is intentionally model-agnostic.
"""
import json
from pathlib import Path

STATE = Path("developmental_identity_state.json")

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "identity": {
            "objective": "increase validated capability",
            "principles": [
                "separate observation from inference",
                "require independent evidence",
                "prefer reversible changes",
                "preserve uncertainty"
            ]
        },
        "generation": 0,
        "capabilities": [],
        "validated_knowledge": [],
        "uncertainties": [],
        "lineage": []
    }

def checkpoint(s, worker_name):
    s["generation"] += 1
    s["lineage"].append({
        "generation": s["generation"],
        "worker": worker_name,
        "capabilities": list(s["capabilities"]),
        "knowledge_count": len(s["validated_knowledge"]),
        "uncertainty_count": len(s["uncertainties"]),
    })

def main():
    s = load()

    # Worker A contributes a validated capability.
    if "symbolic_composition" not in s["capabilities"]:
        s["capabilities"].append("symbolic_composition")
        s["validated_knowledge"].append({
            "claim": "validated capabilities belong to developmental state",
            "evidence": "independent toy evaluator"
        })
    checkpoint(s, "worker-A")

    # Simulate replacing the worker. The state is the continuity carrier.
    checkpoint(s, "worker-B")

    # Worker B adds a new uncertainty rather than silently treating it as fact.
    if not any(u["question"] == "does continuity survive implementation change"
               for u in s["uncertainties"]):
        s["uncertainties"].append({
            "question": "does continuity survive implementation change",
            "status": "open"
        })

    checkpoint(s, "worker-B-after-review")

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))

if __name__ == "__main__":
    main()
