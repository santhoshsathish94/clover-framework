"""Model-free developmental baseline.

This is deliberately not an AI model. It tests whether a persistent
computational loop can improve a policy through search, mutation,
evaluation and retained state.
"""
import json, math, random
from pathlib import Path

STATE = Path("model_free_state.json")
SEED = int(__import__("os").getenv("CLOVER_SEED", "42"))
POPULATION_SIZE = 40
ELITE_SIZE = 8
CYCLES = 30

random.seed(SEED)

def objective(x, target=0.731):
    return -(x - target) ** 2

def load():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cycle": 0, "population": [random.uniform(-2, 2)
            for _ in range(POPULATION_SIZE)], "history": []}

def save(state):
    STATE.write_text(json.dumps(state, indent=2))

def run():
    state = load()
    population = state["population"]

    for _ in range(CYCLES):
        ranked = sorted(population, key=objective, reverse=True)
        elite = ranked[:ELITE_SIZE]
        best = elite[0]

        state["cycle"] += 1
        state["history"].append({
            "cycle": state["cycle"],
            "best": best,
            "score": objective(best),
        })

        sigma = max(0.002, 0.4 * (0.94 ** state["cycle"]))
        population = elite[:]
        while len(population) < POPULATION_SIZE:
            parent = random.choice(elite)
            population.append(parent + random.gauss(0, sigma))

    state["population"] = population
    save(state)
    return state["history"][-1]

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
