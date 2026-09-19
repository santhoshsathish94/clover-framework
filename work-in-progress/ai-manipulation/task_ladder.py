"""Task ladder and anchor tests.

The engines read this module and must never write it. The anchor points are the
only evidence that a promoted capability generalizes, so they sit outside the
search that they judge. `integrity_hash` lets state record which version of
these tests a capability was validated against.

The ladder is ordered so that consecutive tasks sometimes share structure. That
is what gives transfer something to demonstrate: an engine that remembers which
representation last worked should reach the next solution for less budget than
one that starts cold, and if it does not, the hypothesis fails.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAIN = [-2, -1, 0, 1, 2]
ANCHOR = [-3, -1.25, -0.5, 0.25, 0.75, 1.5, 2.5, 4]

LADDER = [
    {"name": "linear_a", "structure": "linear", "target": lambda x: 2 * x + 1},
    {"name": "linear_b", "structure": "linear", "target": lambda x: -3 * x + 2},
    {"name": "cubic_a", "structure": "cubic", "target": lambda x: x**3 - x},
    {"name": "cubic_b", "structure": "cubic", "target": lambda x: x**3 + x * x - 1},
    {"name": "quadratic_shift", "structure": "quadratic", "target": lambda x: x * x + 6 * x + 9},
]

def task(index):
    return LADDER[index] if 0 <= index < len(LADDER) else None

def count():
    return len(LADDER)

def integrity_hash():
    return hashlib.sha256(ROOT.joinpath("task_ladder.py").read_bytes()).hexdigest()[:16]

if __name__ == "__main__":
    print(f"{count()} tasks, anchor integrity {integrity_hash()}")
    for i, t in enumerate(LADDER):
        print(f"  {i}. {t['name']:<16} structure={t['structure']}")
