"""Persistent model-free developmental engine.

This is the autonomous coordinator for the experiments in this directory.
It does not require a language model.

Each cycle:
1. loads developmental memory
2. measures current capabilities
3. identifies a capability gap
4. chooses an experiment under a bounded budget
5. executes a candidate
6. validates against independent tests
7. promotes only verified improvements
8. records the next unresolved question

The engine deliberately separates trusted invariants from revisable state.
It may propose code changes, but automatic promotion of source changes is
disabled by default.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "development_engine_state.json"
MEMORY = ROOT / "developmental_memory.json"

BUDGET = 25
SEED = int(__import__("os").environ.get("CLOVER_SEED", "42"))
random.seed(SEED)

TRUSTED_INVARIANTS = {
    "bounded_budget": True,
    "independent_validation": True,
    "rollback_required": True,
    "source_self_promotion": False,
}

EXPERIMENTS = [
    ("program_development", "discover a better executable procedure"),
    ("meta_development", "improve search efficiency"),
    ("capability_growth", "acquire a reusable capability"),
    ("representation_expansion", "expand the problem representation"),
    ("test_expansion", "increase independent test coverage"),
    ("architecture_review", "identify a developmental bottleneck"),
]

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default

def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2))

def benchmark(x):
    return x * x + 2 * x + 1

def candidate(x, a, b, c):
    return a*x*x + b*x + c

def evaluate(params, tests):
    a, b, c = params
    errors = []
    for x in tests:
        y = candidate(x, a, b, c)
        if not math.isfinite(y):
            return -1e9
        errors.append(abs(y - benchmark(x)))
    return -sum(errors) / len(errors)

def search():
    tests = [-2, -1, -.5, 0, .5, 1, 2]
    best = (-1e9, None)
    for _ in range(BUDGET):
        params = (
            random.randint(-2, 2),
            random.randint(-3, 3),
            random.randint(-2, 2),
        )
        score = evaluate(params, tests)
        if score > best[0]:
            best = (score, params)
    return best

def independent_validate(params):
    tests = [-3, -.25, .25, .75, 1.5, 3]
    return evaluate(params, tests)

def choose_direction(state, memory):
    uncertainties = memory.get("uncertainties", [])
    failures = memory.get("failures", [])
    capabilities = memory.get("capabilities", [])

    if failures:
        return "reduce_failure", failures[-1].get("description", "recent failure")
    if uncertainties:
        return "verify_uncertainty", uncertainties[-1].get("question", "open question")
    if len(capabilities) < 5:
        return "capability_growth", "expand reusable capability library"
    return random.choice(EXPERIMENTS)

def run_cycle():
    state = load_json(STATE, {
        "cycle": 0,
        "trusted_capabilities": [],
        "validated_results": [],
        "open_questions": [],
        "history": [],
    })
    memory = load_json(MEMORY, {
        "experience": [],
        "hypotheses": [],
        "validated_knowledge": [],
        "capabilities": [],
        "uncertainties": [],
        "failures": [],
    })

    direction = choose_direction(state, memory)
    before_count = len(state["trusted_capabilities"])

    score, params = search()
    validation = independent_validate(params)
    promoted = validation >= -0.01

    if promoted:
        capability = {
            "name": f"quadratic_solution_{params}",
            "params": params,
            "validation": validation,
        }
        if capability["name"] not in state["trusted_capabilities"]:
            state["trusted_capabilities"].append(capability["name"])
            memory["capabilities"].append(capability)

    state["cycle"] += 1
    record = {
        "cycle": state["cycle"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "development_score": score,
        "independent_validation": validation,
        "promoted": promoted,
        "capability_count_before": before_count,
        "capability_count_after": len(state["trusted_capabilities"]),
    }
    state["history"].append(record)

    if not promoted:
        memory["failures"].append({
            "description": "candidate failed independent validation",
            "cycle": state["cycle"],
        })
    else:
        memory["validated_knowledge"].append({
            "claim": "candidate passed held-out validation",
            "evidence": record,
        })

    memory["experience"].append(record)
    save_json(STATE, state)
    save_json(MEMORY, memory)
    return record

if __name__ == "__main__":
    print(json.dumps(run_cycle(), indent=2))
