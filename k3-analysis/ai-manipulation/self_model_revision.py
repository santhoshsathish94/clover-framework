"""Self-model revision experiment.

The system maintains explicit assumptions about its own developmental process.
It generates alternative assumptions and tests their consequences against
independent measurements.

This is a bounded scientific experiment, not a claim of consciousness.
"""
import json
import random
from pathlib import Path

from run_context import seed_for, state_path

STATE = state_path("self_model_revision_state.json")
SEED = seed_for("self_model_revision")

BASELINE = {
    "continuity_carrier": "persistent_external_state",
    "growth_definition": "validated_capability_gain",
    "direction_source": "uncertainty_and_capability_gaps",
    "validation": "independent_evaluator",
    "resource_control": "external_budget",
}

ALTERNATIVES = {
    "continuity_carrier": [
        "persistent_external_state",
        "worker_instance",
        "versioned_capability_lineage",
    ],
    "growth_definition": [
        "validated_capability_gain",
        "task_score_gain",
        "search_space_expansion",
    ],
    "direction_source": [
        "uncertainty_and_capability_gaps",
        "fixed_objective",
        "expected_information_gain",
    ],
}

def independent_measurement(model):
    # Toy measurements intentionally use criteria outside the candidate claim.
    continuity = model["continuity_carrier"] in {
        "persistent_external_state", "versioned_capability_lineage"
    }
    growth = model["growth_definition"] in {
        "validated_capability_gain", "search_space_expansion"
    }
    direction = model["direction_source"] in {
        "uncertainty_and_capability_gaps", "expected_information_gain"
    }
    return sum([continuity, growth, direction]) / 3

def propose(model):
    key = random.choice(list(ALTERNATIVES))
    alternatives = [x for x in ALTERNATIVES[key] if x != model[key]]
    candidate = dict(model)
    candidate[key] = random.choice(alternatives)
    return key, candidate

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {
            "cycle": 0,
            "self_model": BASELINE,
            "history": [],
            "revisions": [],
        }

    for _ in range(30):
        key, candidate = propose(s["self_model"])
        before = independent_measurement(s["self_model"])
        after = independent_measurement(candidate)
        promoted = after > before

        s["cycle"] += 1
        record = {
            "cycle": s["cycle"],
            "changed_assumption": key,
            "before": before,
            "after": after,
            "promoted": promoted,
            "candidate": candidate,
        }

        if promoted:
            s["self_model"] = candidate
            s["revisions"].append(record)

        s["history"].append(record)

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps({
        "cycle": s["cycle"],
        "self_model": s["self_model"],
        "revisions": len(s["revisions"]),
    }, indent=2))

if __name__ == "__main__":
    main()
