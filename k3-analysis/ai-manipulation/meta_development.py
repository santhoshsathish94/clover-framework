"""Meta-development experiment.

A model-free population evolves both candidate solutions and a small set of
search parameters. The evaluator remains fixed and independent.
"""
import json, random, math
from pathlib import Path

from run_context import seed_for, state_path

STATE = state_path("meta_development_state.json")
SEED = seed_for("meta_development")

# Each strategy is a tiny developmental policy:
# mutation rate, exploration rate, and elite fraction.
def clamp(x, lo, hi): return max(lo, min(hi, x))

def task(x): return 2*x + 0.5

def candidate_error(a, b):
    # Candidate program is y = a*x + b.
    return sum(abs((a*x + b) - task(x)) for x in [0, .25, .5, .75, 1]) / 5

def strategy_score(strategy):
    a = random.uniform(-1, 3)
    b = random.uniform(-1, 2)
    best = candidate_error(a, b)
    for _ in range(max(1, int(20 * strategy["exploration"]))):
        na = a + random.gauss(0, strategy["mutation"])
        nb = b + random.gauss(0, strategy["mutation"])
        e = candidate_error(na, nb)
        if e < best:
            a, b, best = na, nb, e
    return -best

def mutate_strategy(s):
    return {
        "mutation": clamp(s["mutation"] * math.exp(random.gauss(0, .35)), .001, 1),
        "exploration": clamp(s["exploration"] * math.exp(random.gauss(0, .35)), .01, 1),
    }

def main():
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = {
            "cycle": 0,
            "strategies": [
                {"mutation": .05, "exploration": .1},
                {"mutation": .2, "exploration": .3},
                {"mutation": .5, "exploration": .8},
            ],
            "history": []
        }

    for _ in range(30):
        scored = sorted(
            [(strategy_score(s), s) for s in state["strategies"]],
            reverse=True, key=lambda z: z[0]
        )
        state["cycle"] += 1
        state["history"].append({
            "cycle": state["cycle"],
            "best_score": scored[0][0],
            "best_strategy": scored[0][1]
        })
        elites = [s for _, s in scored[:2]]
        state["strategies"] = elites + [
            mutate_strategy(random.choice(elites)) for _ in range(6)
        ]

    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps(state["history"][-1], indent=2))

if __name__ == "__main__":
    main()
