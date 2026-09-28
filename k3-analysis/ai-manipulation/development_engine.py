"""Persistent model-free developmental engine.

This is the autonomous coordinator for the experiments in this directory.
It does not require a language model.

Each cycle:
1. loads developmental memory
2. chooses a direction from recorded failures and uncertainties
3. sets a search budget from that direction
4. searches for a candidate under that budget
5. validates the candidate against held-out tests
6. promotes only candidates that pass validation
7. records the cycle, including the seed needed to reproduce it

The engine searches for parameters only. It does not modify its own source.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "development_engine_state.json"
MEMORY = ROOT / "developmental_memory.json"

BASE_BUDGET = 25
MAX_BUDGET = 200

# Offset per cycle, never reused: seeding once per process made every run identical.
SEED_BASE = int(os.environ.get("CLOVER_SEED", "42"))

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

def search(budget):
    tests = [-2, -1, -.5, 0, .5, 1, 2]
    best = (-1e9, None)
    for _ in range(budget):
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

def consecutive_failures(memory):
    count = 0
    for record in reversed(memory.get("experience", [])):
        # The memory file is shared with developmental_memory.py, which writes non-cycle events.
        if "promoted" not in record:
            continue
        if record["promoted"]:
            break
        count += 1
    return count

def budget_for(direction, memory):
    """Repeated failure buys a wider search, so a stuck cycle is not retried unchanged."""
    if direction[0] != "reduce_failure":
        return BASE_BUDGET
    return min(BASE_BUDGET * (1 + consecutive_failures(memory)), MAX_BUDGET)

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

    # This domain defines exactly one capability. Once it is held, further cycles
    # cannot learn anything, so continuing would be activity rather than development.
    if state["trusted_capabilities"]:
        return {
            "cycle": state["cycle"],
            "status": "pause",
            "reason": "the only capability this task space defines has been acquired",
            "trusted_capabilities": state["trusted_capabilities"],
            "next_step": "widen the task space before running further cycles",
        }

    direction = choose_direction(state, memory)
    before_count = len(state["trusted_capabilities"])

    seed = SEED_BASE + state["cycle"]
    random.seed(seed)
    budget = budget_for(direction, memory)

    score, params = search(budget)
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
        "status": "continue",
        "direction": direction,
        "seed": seed,
        "budget": budget,
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
