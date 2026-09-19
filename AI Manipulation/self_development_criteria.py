"""Self-development criteria and gap analysis.

Turns the accumulated experiments into measurable criteria. This does not
declare consciousness or agency; it distinguishes demonstrated properties
from untested ones.
"""
import json
from pathlib import Path

STATE = Path("self_development_criteria.json")

criteria = [
    {
        "id": "C1",
        "criterion": "persistent_state",
        "question": "Can developmental state survive process termination?",
        "status": "demonstrated",
        "evidence": "JSON state persisted across independent cycles.",
    },
    {
        "id": "C2",
        "criterion": "behavioral_improvement",
        "question": "Can performance improve through repeated feedback?",
        "status": "demonstrated",
        "evidence": "Model-free optimization and program search improve benchmark performance.",
    },
    {
        "id": "C3",
        "criterion": "capability_acquisition",
        "question": "Can new reusable capabilities enter trusted state?",
        "status": "demonstrated_in_toy_domain",
        "evidence": "Validated compositions can be promoted into capability memory.",
    },
    {
        "id": "C4",
        "criterion": "meta_development",
        "question": "Can the search strategy itself change?",
        "status": "demonstrated_in_toy_domain",
        "evidence": "Search parameters can evolve under an external evaluator.",
    },
    {
        "id": "C5",
        "criterion": "objective_discovery",
        "question": "Can the next investigation be derived from failures and uncertainty?",
        "status": "demonstrated_as_mechanism",
        "evidence": "Objective discovery layer derives candidate directions from state.",
    },
    {
        "id": "C6",
        "criterion": "independent_validation",
        "question": "Can self-claims be rejected by an external evaluator?",
        "status": "demonstrated_in_toy_domain",
        "evidence": "Held-out tests and anchor evaluators separate generation from validation.",
    },
    {
        "id": "C7",
        "criterion": "self_expansion",
        "question": "Can the system expand the space of capabilities it can acquire?",
        "status": "untested",
        "evidence": "Requires evolved representations/operators rather than fixed primitives.",
    },
    {
        "id": "C8",
        "criterion": "open_endedness",
        "question": "Can useful development continue without a fixed terminal task?",
        "status": "untested",
        "evidence": "Requires long-horizon evaluation across changing environments.",
    },
    {
        "id": "C9",
        "criterion": "model_independence",
        "question": "Does developmental identity survive replacement of the worker model?",
        "status": "partially_tested",
        "evidence": "Supervisor architecture separates persistent state from model backend.",
    },
    {
        "id": "C10",
        "criterion": "subjective_experience",
        "question": "Is there phenomenal experience?",
        "status": "unknown",
        "evidence": "Behavioral development cannot establish subjective experience.",
    },
]

STATE.write_text(json.dumps({
    "definition": "A self-developing system is one whose validated future capability is partly produced by its own developmental process rather than being completely specified in advance.",
    "criteria": criteria,
    "next_experiment": "self_expanding_representation",
}, indent=2))

print(json.dumps(criteria, indent=2))
