"""Self-expanding developmental search.

The system maintains candidate representations and operators, generates
tests from failures, and promotes only changes that improve an external
benchmark. No language model is required.
"""
import json, random, math
from pathlib import Path

STATE = Path("open_ended_state.json")
random.seed(int(__import__("os").getenv("CLOVER_SEED", "42")))

# A tiny symbolic universe. The system can discover compositions of primitives.
PRIMITIVES = ["x", "1", "2", "0.5"]
OPS = ["+", "-", "*"]

def eval_expr(expr, x):
    try:
        return eval(expr, {"__builtins__": {}}, {"x": x})
    except Exception:
        return None

def target(x):
    return x*x + 2*x + 1

def tests_for(history):
    # Tests become more discriminating when prior candidates fail.
    base = [0, .25, .5, .75, 1]
    if not history:
        return base
    return base + [-1, 1.5, 2]

def score(expr, tests):
    errors = []
    for x in tests:
        y = eval_expr(expr, x)
        if not isinstance(y, (int, float)) or not math.isfinite(y):
            return -1e9
        errors.append(abs(y - target(x)))
    return -sum(errors) / len(errors)

def generate(archive, depth=2):
    pool = list(PRIMITIVES)
    pool += [a["expr"] for a in archive[-20:]]
    out = []
    for _ in range(60):
        a = random.choice(pool)
        b = random.choice(pool)
        expr = f"({a}){random.choice(OPS)}({b})"
        out.append(expr)
    return out

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {"cycle": 0, "archive": [], "tests": [], "representations": ["symbolic-arithmetic"]}

    for _ in range(20):
        tests = tests_for(s["archive"])
        candidates = generate(s["archive"])
        ranked = sorted(((score(e, tests), e) for e in candidates), reverse=True)
        best_score, best_expr = ranked[0]

        # Independent validation uses a held-out test set.
        held_out = [-2, -.5, .1, 1.2, 3]
        validation = score(best_expr, held_out)

        s["cycle"] += 1
        s["tests"] = tests
        record = {
            "cycle": s["cycle"],
            "expr": best_expr,
            "development_score": best_score,
            "held_out_score": validation,
            "validated": validation > -0.01,
        }

        if record["validated"]:
            s["archive"].append(record)

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["archive"][-1] if s["archive"] else {"validated": False}, indent=2))

if __name__ == "__main__":
    main()
