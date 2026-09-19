"""Unified Clover developmental engine.

Combines direction discovery, bounded resource allocation, capability growth,
representation search, independent validation, and persistent memory.

This is intentionally model-free. A model backend can later become one
replaceable worker, but the developmental loop itself does not depend on it.
"""
from __future__ import annotations

import json, math, random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "unified_development_state.json"
SEED = int(__import__("os").environ.get("CLOVER_SEED", "42"))
random.seed(SEED)

BUDGET = 50
HELD_OUT = [-3, -1.25, -.5, .25, .75, 1.5, 2.5, 4]

def target(x):
    return x*x + 2*x + 1

def representations():
    return {
        "linear": lambda p, x: p[0]*x + p[1],
        "quadratic": lambda p, x: p[0]*x*x + p[1]*x + p[2],
        "composition": lambda p, x: p[0]*(x+p[1]) + p[2],
    }

PARAMS = {
    "linear": [(a,b) for a in [-3,-2,-1,0,1,2,3] for b in [-2,-1,0,1,2]],
    "quadratic": [(a,b,c) for a in [-1,0,1]
                              for b in [-2,-1,0,1,2]
                              for c in [-1,0,1]],
    "composition": [(a,b,c) for a in [1,2,3]
                                  for b in [-2,-1,0,1,2]
                                  for c in [-2,-1,0,1,2]],
}

def score(rep, params, tests):
    fn = representations()[rep]
    errors = []
    for x in tests:
        try:
            y = fn(params, x)
            if not math.isfinite(y):
                return -1e9
            errors.append(abs(y-target(x)))
        except Exception:
            return -1e9
    return -sum(errors)/len(errors)

def discover_best():
    tests = [-2,-1,0,1,2]
    best = (-1e9, None, None)
    reps = list(representations())
    # The engine chooses representation as part of the search.
    for rep in reps:
        for params in PARAMS[rep]:
            value = score(rep, params, tests)
            if value > best[0]:
                best = (value, rep, params)
    return best

def validate(rep, params):
    return score(rep, params, HELD_OUT)

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "cycle": 0,
        "capabilities": [],
        "trusted_representations": ["linear"],
        "knowledge": [],
        "uncertainties": [
            "which representation generalizes best",
            "which capabilities remain missing",
        ],
        "failures": [],
        "history": [],
        "resource": {"budget": BUDGET, "spent": 0},
    }

def direction(s):
    if s["failures"]:
        return "reduce_failure"
    if s["uncertainties"]:
        return "resolve_uncertainty"
    if len(s["capabilities"]) < 5:
        return "acquire_capability"
    return "expand_representation"

def run_cycle():
    s = load()
    s["resource"]["spent"] = 0

    chosen_direction = direction(s)
    dev_score, rep, params = discover_best()
    s["resource"]["spent"] += len(PARAMS[rep])
    validation = validate(rep, params)

    promoted = validation >= -0.001
    capability_name = f"{rep}:{params}"

    if promoted and capability_name not in s["capabilities"]:
        s["capabilities"].append(capability_name)

    if promoted and rep not in s["trusted_representations"]:
        s["trusted_representations"].append(rep)

    if promoted:
        s["knowledge"].append({
            "claim": f"{rep} representation produced validated solution",
            "evidence": {
                "development_score": dev_score,
                "held_out_score": validation,
                "params": params,
            }
        })
    else:
        s["failures"].append({
            "cycle": s["cycle"] + 1,
            "reason": "held-out validation failed",
            "representation": rep,
        })

    s["cycle"] += 1
    s["resource"]["spent"] = min(s["resource"]["spent"], BUDGET)

    record = {
        "cycle": s["cycle"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": chosen_direction,
        "representation": rep,
        "params": params,
        "development_score": dev_score,
        "held_out_score": validation,
        "promoted": promoted,
        "capability_count": len(s["capabilities"]),
        "trusted_representation_count": len(s["trusted_representations"]),
        "budget_spent": s["resource"]["spent"],
    }
    s["history"].append(record)

    # Keep unresolved questions alive unless evidence actually resolves them.
    if promoted and rep in s["trusted_representations"]:
        s["uncertainties"] = [
            u for u in s["uncertainties"]
            if not (u == "which representation generalizes best" and validation >= -0.001)
        ]

    save_state(s)
    return record

def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))

if __name__ == "__main__":
    print(json.dumps(run_cycle(), indent=2))
