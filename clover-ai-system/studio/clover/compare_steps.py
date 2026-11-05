"""Paired comparison of ladder steps against a base step: per-category pass counts, flips, exact two-sided sign test.

  python compare_steps.py verify/layers/step1.jsonl verify/layers/step2-memory.jsonl [...]
Rust tasks count only when rescored (rescore_rust.py writes passed=True/False; unscored Rust has passed=None).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Tasks whose answer is written into the L5 ontology records (hand-made entries that state the specific numbers or
# option of these test questions). A gain here measures a leaked answer, not the layer.
ONTOLOGY_ANSWER_KEY_TASKS = {
    "mmlu_college_computer_science_1", "mmlu_college_biology_0", "mmlu_college_physics_0", "mmlu_college_physics_1",
    "agent_ha_q5", "agent_mh_q7", "rust_gc_rust_live_1_gcd", "onto_qa_5_thin_film_600nm",
}


def load(p):
    return {r["task_id"]: r for r in (json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip())}


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(base, other, name):
    cats = defaultdict(lambda: [0, 0, 0])
    gains, losses = [], []
    for tid, b in base.items():
        o = other.get(tid)
        if o is None or b.get("passed") is None or o.get("passed") is None:
            continue
        c = cats[b["category"]]
        c[0] += 1
        c[1] += bool(b["passed"])
        c[2] += bool(o["passed"])
        if o["passed"] and not b["passed"]:
            gains.append(tid)
        elif b["passed"] and not o["passed"]:
            losses.append(tid)
    n = sum(c[0] for c in cats.values())
    print(f"\n=== {name}  (paired n={n}, regenerated={sum(1 for r in other.values() if not r.get('derived_from_base'))})")
    for cat in sorted(cats):
        c = cats[cat]
        print(f"  {cat:38s} {c[1]:2d} -> {c[2]:2d} / {c[0]}")
    tb, to = sum(c[1] for c in cats.values()), sum(c[2] for c in cats.values())
    print(f"  {'TOTAL':38s} {tb:2d} -> {to:2d} / {n}   gains {len(gains)}  losses {len(losses)}  sign-test p={sign_p(len(gains), len(losses)):.3f}")
    print("  gains :", gains)
    print("  losses:", losses)
    keyed = [t for t in gains + losses if t in ONTOLOGY_ANSWER_KEY_TASKS]
    if keyed:
        g2 = [t for t in gains if t not in ONTOLOGY_ANSWER_KEY_TASKS]
        l2 = [t for t in losses if t not in ONTOLOGY_ANSWER_KEY_TASKS]
        print(f"  without answer-key tasks {keyed}: gains {len(g2)} losses {len(l2)} p={sign_p(len(g2), len(l2)):.3f}")


if __name__ == "__main__":
    base = load(sys.argv[1])
    for p in sys.argv[2:]:
        compare(base, load(p), Path(p).stem)
