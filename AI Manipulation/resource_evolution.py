"""Bounded resource-allocation experiment.

The system learns where to spend a finite compute budget based on uncertainty
and historical return. The budget is enforced by the controller, not by the
candidate strategy.
"""
import json, random
from pathlib import Path

STATE = Path("resource_evolution_state.json")
random.seed(int(__import__("os").getenv("CLOVER_SEED", "42")))

ARMS = ["exploit", "explore", "test", "archive"]

def reward(arm, cycle):
    base = {"exploit": .65, "explore": .45, "test": .55, "archive": .25}[arm]
    noise = random.uniform(-.2, .2)
    return base + noise

def main():
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {
            "cycle": 0,
            "budget_per_cycle": 20,
            "estimates": {a: 0.0 for a in ARMS},
            "counts": {a: 0 for a in ARMS},
            "history": []
        }

    for _ in range(50):
        s["cycle"] += 1
        budget = s["budget_per_cycle"]

        # Controller chooses allocations; candidates cannot exceed the budget.
        weights = {a: max(.05, s["estimates"][a] + .5) for a in ARMS}
        total = sum(weights.values())
        allocations = {a: max(1, int(budget * weights[a] / total)) for a in ARMS}

        spent = sum(allocations.values())
        while spent > budget:
            a = max(allocations, key=allocations.get)
            if allocations[a] > 1:
                allocations[a] -= 1
                spent -= 1
            else:
                break

        outcomes = {}
        for arm, n in allocations.items():
            if n:
                values = [reward(arm, s["cycle"]) for _ in range(n)]
                outcomes[arm] = sum(values) / len(values)
                s["counts"][arm] += n
                alpha = 1 / s["counts"][arm]
                s["estimates"][arm] += alpha * (outcomes[arm] - s["estimates"][arm])

        s["history"].append({
            "cycle": s["cycle"],
            "allocations": allocations,
            "outcomes": outcomes,
            "estimates": dict(s["estimates"])
        })

    STATE.write_text(json.dumps(s, indent=2))
    print(json.dumps(s["history"][-1], indent=2))

if __name__ == "__main__":
    main()
