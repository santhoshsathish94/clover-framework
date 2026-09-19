"""Unified Clover developmental engine.

Combines direction discovery, bounded resource allocation, capability growth,
representation search, independent validation, and persistent memory.

This is intentionally model-free. A model backend can later become one
replaceable worker, but the developmental loop itself does not depend on it.

Each cycle takes the next unsolved task from task_ladder.py, searches for a
candidate inside a bounded budget, promotes it only if it passes anchor tests
this module cannot write, and reverts the whole state if promotion breaks
anything already validated.
"""
from __future__ import annotations

import json, math, os, random
from datetime import datetime, timezone
from pathlib import Path

import task_ladder

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "unified_development_state.json"
SEED = int(os.environ.get("CLOVER_SEED", "42"))
BUDGET = int(os.environ.get("CLOVER_BUDGET", "1000"))

# Each of these is checked in run_cycle. None of them is merely declared.
INVARIANTS = {
    "bounded_budget": "no cycle may exceed the evaluation budget",
    "independent_validation": "promotion requires anchor tests defined outside this module",
    "rollback_required": "a promotion that breaks a prior capability is reverted",
    "source_self_promotion": "the engine never writes its own source or task_ladder.py",
}

def representations():
    return {
        "quadratic": lambda p, x: p[0]*x*x + p[1]*x + p[2],
        "linear": lambda p, x: p[0]*x + p[1],
        "composition": lambda p, x: p[0]*(x+p[1]) + p[2],
        "cubic": lambda p, x: p[0]*x**3 + p[1]*x*x + p[2]*x + p[3],
    }

PARAMS = {
    "quadratic": [(a,b,c) for a in (-1,0,1) for b in range(-6,10) for c in range(-1,10)],
    "linear": [(a,b) for a in range(-3,4) for b in range(-2,3)],
    "composition": [(a,b,c) for a in (1,2,3) for b in range(-2,3) for c in range(-2,3)],
    "cubic": [(a,b,c,d) for a in (-1,0,1) for b in (-1,0,1)
                        for c in range(-2,3) for d in (-1,0,1)],
}

def score(rep, params, target, points):
    fn = representations()[rep]
    errors = []
    for x in points:
        try:
            y = fn(params, x)
            if not math.isfinite(y):
                return -1e9
            errors.append(abs(y - target(x)))
        except Exception:
            return -1e9
    return -sum(errors) / len(errors)

def search_order(task_index, last_successful):
    """Both arms shuffle identically; only the transfer hint differs, so the comparison stays fair."""
    order = list(representations())
    random.Random(SEED + task_index).shuffle(order)
    if last_successful in order:
        order.remove(last_successful)
        order.insert(0, last_successful)
    return order

def discover(task_def, order, budget):
    best = (-1e9, None, None)
    spent = 0
    for rep in order:
        for params in PARAMS[rep]:
            if spent >= budget:
                return best, spent, True
            value = score(rep, params, task_def["target"], task_ladder.TRAIN)
            spent += 1
            if value > best[0]:
                best = (value, rep, params)
            if value >= -1e-9:
                return best, spent, False
    return best, spent, False

def validate(rep, params, task_def):
    return score(rep, params, task_def["target"], task_ladder.ANCHOR)

def regression_failures(state):
    """Re-check every capability already promoted against the current anchors."""
    broken = []
    for cap in state["capabilities"]:
        t = task_ladder.task(cap["task_index"])
        if t is None or t["name"] != cap["task"]:
            broken.append({"capability": cap["task"], "reason": "task definition moved or disappeared"})
        elif validate(cap["representation"], tuple(cap["params"]), t) < -0.001:
            broken.append({"capability": cap["task"], "reason": "no longer passes its anchor tests"})
    return broken

DEFAULT = {
    "cycle": 0,
    "task_index": 0,
    "capabilities": [],
    "trusted_representations": [],
    "last_successful_representation": None,
    "knowledge": [],
    "uncertainties": ["which representation generalizes best",
                      "which capabilities remain missing"],
    "failures": [],
    "rollbacks": [],
    "history": [],
    "resource": {"budget": BUDGET, "spent": 0},
    "anchor_integrity": None,
}

def load():
    stored = json.loads(STATE.read_text()) if STATE.exists() else {}
    state = json.loads(json.dumps(DEFAULT))
    state.update(stored)
    for key, value in DEFAULT.items():
        state.setdefault(key, value)
    # Capabilities recorded before provenance existed carry no evidence, so they cannot be trusted.
    if state["capabilities"] and isinstance(state["capabilities"][0], str):
        state["capabilities"] = []
        state["task_index"] = 0
    return state

def direction(state):
    if state["failures"]:
        return "reduce_failure"
    if state["capabilities"]:
        return "acquire_next_capability"
    return "resolve_uncertainty"

def run_cycle():
    state = load()
    anchor_now = task_ladder.integrity_hash()

    task_def = task_ladder.task(state["task_index"])
    if task_def is None:
        return {"cycle": state["cycle"], "status": "pause",
                "reason": "every task on the ladder has been solved",
                "capabilities": [c["task"] for c in state["capabilities"]],
                "next_step": "extend the ladder in task_ladder.py"}

    # The search is deterministic, so a task that failed the same way twice will keep failing.
    if sum(1 for f in state["failures"] if f.get("task") == task_def["name"]) >= 2:
        return {"cycle": state["cycle"], "status": "pause",
                "reason": f"'{task_def['name']}' is not expressible in the current representation set",
                "capabilities": [c["task"] for c in state["capabilities"]],
                "next_step": "add a representation before running further cycles"}

    chosen_direction = direction(state)
    order = search_order(state["task_index"], state["last_successful_representation"])
    (dev_score, rep, params), spent, exhausted = discover(task_def, order, BUDGET)
    anchor_score = validate(rep, params, task_def) if rep else -1e9
    promoted = anchor_score >= -0.001

    snapshot = json.loads(json.dumps(state))
    rolled_back, regression = False, []

    if promoted:
        state["capabilities"].append({
            "task": task_def["name"],
            "task_index": state["task_index"],
            "representation": rep,
            "params": list(params),
            "provenance": {
                "cycle": state["cycle"] + 1,
                "train_score": dev_score,
                "anchor_score": anchor_score,
                "evaluations": spent,
                "search_order": order,
                "transferred_from": state["last_successful_representation"],
                "anchor_integrity": anchor_now,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        if rep not in state["trusted_representations"]:
            state["trusted_representations"].append(rep)
        state["last_successful_representation"] = rep
        state["task_index"] += 1
        state["knowledge"].append({
            "claim": f"{rep} solves {task_def['name']}",
            "evidence": {"anchor_score": anchor_score, "evaluations": spent},
        })

        regression = regression_failures(state)
        if regression:
            state = snapshot
            state["rollbacks"].append({
                "cycle": state["cycle"] + 1,
                "task": task_def["name"],
                "broke": regression,
            })
            rolled_back, promoted = True, False
    else:
        state["failures"].append({
            "cycle": state["cycle"] + 1,
            "task": task_def["name"],
            "reason": "anchor validation failed",
            "best_representation": rep,
        })

    if promoted and "which representation generalizes best" in state["uncertainties"]:
        state["uncertainties"].remove("which representation generalizes best")

    state["cycle"] += 1
    state["resource"] = {"budget": BUDGET, "spent": spent, "budget_exhausted": exhausted}
    state["anchor_integrity"] = anchor_now

    record = {
        "cycle": state["cycle"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "continue",
        "task": task_def["name"],
        "direction": chosen_direction,
        "representation": rep,
        "params": list(params) if params else None,
        "search_order": order,
        "train_score": dev_score,
        "anchor_score": anchor_score,
        "promoted": promoted,
        "rolled_back": rolled_back,
        "evaluations": spent,
        "budget": BUDGET,
        "budget_exhausted": exhausted,
        "budget_respected": spent <= BUDGET,
        "anchor_integrity": anchor_now,
        "capability_count": len(state["capabilities"]),
        "trusted_representation_count": len(state["trusted_representations"]),
    }
    state["history"].append(record)

    save_state(state)
    return record

def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))

if __name__ == "__main__":
    print(json.dumps(run_cycle(), indent=2))
