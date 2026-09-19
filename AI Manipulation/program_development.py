"""Model-free program-development baseline.

Evolves small Python programs against an independently defined benchmark.
No language model, API, or hosted inference is required.
"""
import ast, json, math, random
from pathlib import Path

STATE = Path("program_development_state.json")
SEED = int(__import__("os").getenv("CLOVER_SEED", "42"))
random.seed(SEED)

TARGETS = [0.0, 0.25, 0.5, 0.75, 1.0]
POPULATION_SIZE = 30
ELITE_SIZE = 6
CYCLES = 20

OPS = ["+", "-", "*"]
CONSTS = ["0", "1", "2", "0.5"]

def safe_expr(expr):
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return False
    allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
               ast.Constant, ast.Name, ast.Load)
    return all(isinstance(n, allowed) and
               (not isinstance(n, ast.Name) or n.id == "x")
               for n in ast.walk(tree))

def run_program(expr, x):
    if not safe_expr(expr):
        return None
    try:
        value = eval(compile(ast.parse(expr, mode="eval"), "<candidate>", "eval"),
                     {"__builtins__": {}}, {"x": x})
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    except Exception:
        pass
    return None

def score(expr):
    # Independent benchmark: approximate f(x)=x*2+0.5
    errors = []
    for x in TARGETS:
        y = run_program(expr, x)
        if y is None:
            return -1e9
        errors.append(abs(y - (2*x + 0.5)))
    return -sum(errors) / len(errors)

def seed_population():
    return ["x", "x+1", "x*2", "x*2+1", "x*2+0.5", "x-1",
            "x*0.5", "0"] + [random.choice(["x", "x+1", "x*2", "0.5"])
                              for _ in range(POPULATION_SIZE-8)]

def mutate(expr):
    if not safe_expr(expr):
        return random.choice(["x", "x+1", "x*2", "x*2+0.5"])
    choice = random.randrange(4)
    if choice == 0:
        return f"({expr}){random.choice(OPS)}{random.choice(CONSTS)}"
    if choice == 1:
        return f"({random.choice(['x','2','0.5'])}){random.choice(OPS)}({expr})"
    if choice == 2:
        return expr.replace("1", random.choice(CONSTS), 1)
    return random.choice(["x", "x+1", "x*2", "x*2+0.5", "0.5"])

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cycle": 0, "population": seed_population(), "archive": [],
            "best": None, "validated": []}

def save(s):
    STATE.write_text(json.dumps(s, indent=2))

def run():
    s = load()
    pop = s["population"]
    for _ in range(CYCLES):
        ranked = sorted(((score(p), p) for p in pop), reverse=True)
        elite = [p for _, p in ranked[:ELITE_SIZE]]
        best_score, best_expr = ranked[0]
        s["cycle"] += 1
        record = {"cycle": s["cycle"], "program": best_expr,
                  "score": best_score}
        s["archive"].append(record)
        if s["best"] is None or best_score > s["best"]["score"]:
            s["best"] = record
            if best_score >= -1e-9:
                s["validated"].append(record)
        pop = elite[:]
        while len(pop) < POPULATION_SIZE:
            pop.append(mutate(random.choice(elite)))
    s["population"] = pop
    save(s)
    return s["best"]

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
