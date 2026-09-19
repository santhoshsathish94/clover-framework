"""Capability acquisition + rollback layer.

Capabilities are executable transformations discovered by the search process.
A capability is promoted only after passing regression and held-out tests.
"""
import json, random, math
from pathlib import Path

from run_context import seed_for, state_path

STATE = state_path("capability_growth_state.json")
SEED = seed_for("capability_growth")

def square(x): return x*x
def double(x): return x*2
def increment(x): return x+1

BUILTINS = {"square": square, "double": double, "increment": increment}

def compose(a, b):
    return lambda x: a(b(x))

def evaluate(fn, tests):
    try:
        vals = [fn(x) for x in tests]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            return -1e9
        # General benchmark: approximate x^2 + 2x + 1 = square(increment(x))
        return -sum(abs(v - (x*x + 2*x + 1)) for x, v in zip(tests, vals)) / len(tests)
    except Exception:
        return -1e9

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {
            "cycle": 0,
            "capabilities": ["square", "double", "increment"],
            "archive": [],
            "rollbacks": 0
        }

    tests = [-2, -1, -.5, 0, .25, .5, 1, 1.5, 2, 3]
    # A composition promoted in an earlier run cannot be rebuilt from the state file,
    # so the callable registry starts from what this process can actually call.
    registry = dict(BUILTINS)
    names = [n for n in s["capabilities"] if n in registry]

    for _ in range(20):
        candidates = []
        for a in names:
            for b in names:
                candidates.append((f"{a}({b}(x))", compose(registry[a], registry[b])))

        ranked = sorted(
            ((evaluate(fn, tests), label) for label, fn in candidates),
            reverse=True
        )
        score, label = ranked[0]
        s["cycle"] += 1

        # Independent promotion criterion.
        held_out = [-3, -.25, .1, 1.2, 2.5, 4]
        fn = next(fn for l, fn in candidates if l == label)
        validation = evaluate(fn, held_out)

        promoted = validation > -0.01
        if promoted and label not in names:
            # Register only validated compositions as new capabilities.
            names.append(label)
            registry[label] = fn
            if label not in s["capabilities"]:
                s["capabilities"].append(label)
        else:
            s["rollbacks"] += 1

        s["archive"].append({
            "cycle": s["cycle"],
            "candidate": label,
            "score": score,
            "validation": validation,
            "promoted": promoted,
        })

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["archive"][-1], indent=2))

if __name__ == "__main__":
    main()
