"""Control arm for the developmental hypothesis.

Turns the claim into something that can fail. The same ladder is solved twice:
once carrying memory of which representation last worked, once starting cold at
every task. Both arms get the same shuffled search order and the same budget, so
memory is the only difference between them.

If persistence contributes nothing, the two totals will be identical. That is a
real possible outcome and it is the point of running this.
"""
from __future__ import annotations

import json

import task_ladder
from unified_development_engine import discover, search_order, validate

CEILING = 10_000

def run_arm(carry_memory):
    last_successful = None
    rows = []
    for index in range(task_ladder.count()):
        task_def = task_ladder.task(index)
        order = search_order(index, last_successful if carry_memory else None)
        (train_score, rep, params), spent, _ = discover(task_def, order, CEILING)
        anchor = validate(rep, params, task_def) if rep else -1e9
        solved = anchor >= -0.001
        if solved and carry_memory:
            last_successful = rep
        rows.append({
            "task": task_def["name"],
            "representation": rep,
            "solved": solved,
            "evaluations": spent,
            "first_tried": order[0],
        })
    return rows

def main():
    control = run_arm(carry_memory=False)
    developmental = run_arm(carry_memory=True)

    print(f"{'task':<18}{'control':>10}{'developmental':>16}{'saved':>9}  first tried (dev)")
    print("-" * 74)
    for c, d in zip(control, developmental):
        saved = c["evaluations"] - d["evaluations"]
        print(f"{c['task']:<18}{c['evaluations']:>10}{d['evaluations']:>16}{saved:>9}  {d['first_tried']}")

    c_total = sum(r["evaluations"] for r in control)
    d_total = sum(r["evaluations"] for r in developmental)
    c_solved = sum(r["solved"] for r in control)
    d_solved = sum(r["solved"] for r in developmental)

    print("-" * 74)
    print(f"{'total':<18}{c_total:>10}{d_total:>16}{c_total - d_total:>9}")
    print(f"\nsolved  control={c_solved}/{task_ladder.count()}  developmental={d_solved}/{task_ladder.count()}")

    if d_solved != c_solved:
        verdict = "arms solved different numbers of tasks; compare capability before budget"
    elif d_total < c_total:
        verdict = f"memory reduced total budget by {c_total - d_total} evaluations ({100*(c_total-d_total)/c_total:.1f}%)"
    elif d_total == c_total:
        verdict = "memory made no difference; the developmental claim is not supported here"
    else:
        verdict = f"memory COST {d_total - c_total} extra evaluations; transfer hurt on this ladder"
    print(f"verdict: {verdict}")

    return {"control": control, "developmental": developmental,
            "control_total": c_total, "developmental_total": d_total,
            "verdict": verdict}

if __name__ == "__main__":
    main()
