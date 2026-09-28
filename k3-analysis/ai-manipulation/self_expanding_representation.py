"""Self-expanding representation experiment.

The system searches over multiple representations and can retain a newly
discovered representation when it demonstrates validated capability gains.

No model is required. The external evaluator defines only task correctness;
the representation-search mechanism is allowed to vary.
"""
import json, random
from pathlib import Path

from run_context import seed_for, state_path

STATE = state_path("self_expanding_representation_state.json")
SEED = seed_for("self_expanding_representation")

TASKS = [
    ("double", lambda x: 2*x),
    ("square_plus_one", lambda x: x*x+1),
    ("affine", lambda x: 3*x-2),
]

REPRESENTATIONS = {
    "linear": lambda x, p: p[0]*x + p[1],
    "quadratic": lambda x, p: p[0]*x*x + p[1]*x + p[2],
    "composition": lambda x, p: p[0]*(x+p[1]) + p[2],
}

def fit(rep, target):
    # Small exhaustive parameter search. The point is representation choice,
    # not numerical sophistication.
    best = (float("inf"), None)
    grids = {
        "linear": [(a,b) for a in [-3,-2,-1,0,1,2,3]
                         for b in [-2,-1,0,1,2]],
        "quadratic": [(a,b,c) for a in [-1,0,1]
                               for b in [-2,-1,0,1,2]
                               for c in [-1,0,1]],
        "composition": [(a,b,c) for a in [1,2,3]
                                  for b in [-2,-1,0,1,2]
                                  for c in [-2,-1,0,1,2]],
    }
    for p in grids[rep]:
        err = sum(abs(REPRESENTATIONS[rep](x,p)-target(x))
                  for x in [-2,-1,0,1,2])
        if err < best[0]:
            best = (err,p)
    return best

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {"cycle": 0, "trusted_representations": ["linear"],
             "discoveries": [], "history": []}

    for _ in range(10):
        task_name, target = random.choice(TASKS)
        candidates = list(REPRESENTATIONS)
        ranked = []
        for rep in candidates:
            err, params = fit(rep, target)
            ranked.append((err, rep, params))
        ranked.sort()

        err, rep, params = ranked[0]

        # Held-out validation.
        heldout = [-3, -.5, .5, 1.5, 3]
        held_err = sum(abs(REPRESENTATIONS[rep](x,params)-target(x))
                       for x in heldout)

        s["cycle"] += 1
        promoted = held_err < 0.01 and rep not in s["trusted_representations"]
        if promoted:
            s["trusted_representations"].append(rep)
            s["discoveries"].append({
                "representation": rep,
                "task": task_name,
                "params": params,
                "heldout_error": held_err,
            })

        s["history"].append({
            "cycle": s["cycle"],
            "task": task_name,
            "selected_representation": rep,
            "development_error": err,
            "heldout_error": held_err,
            "promoted": promoted,
        })

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["history"][-1], indent=2))

if __name__ == "__main__":
    main()
