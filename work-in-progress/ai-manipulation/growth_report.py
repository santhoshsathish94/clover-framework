"""Reduce the developmental ledger to an answer.

The state file accumulates and nobody reads it back. This reports whether later
cycles actually differ from earlier ones, using the measurements the experiment
set for itself: capabilities acquired, budget spent per acquisition, whether
transfer occurred, and whether any promotion had to be reverted.
"""
from __future__ import annotations

import json
from pathlib import Path

import task_ladder
from unified_development_engine import discover, search_order

STATE = Path(__file__).resolve().parent / "unified_development_state.json"

def main():
    if not STATE.exists():
        print("no developmental state recorded yet")
        return

    s = json.loads(STATE.read_text())
    caps = [c for c in s.get("capabilities", []) if isinstance(c, dict)]
    history = s.get("history", [])

    print(f"cycles recorded        : {s.get('cycle', 0)}")
    print(f"tasks on the ladder    : {task_ladder.count()}")
    print(f"capabilities acquired  : {len(caps)}")
    print(f"trusted representations: {', '.join(s.get('trusted_representations', [])) or 'none'}")
    print(f"failures               : {len(s.get('failures', []))}")
    print(f"rollbacks              : {len(s.get('rollbacks', []))}")

    if not caps:
        print("\nnothing has been promoted, so no growth can be claimed")
        return

    print("\ncapability                  evals  cold start  saved  transferred from")
    print("-" * 72)
    saved_total = 0
    for c in caps:
        p = c["provenance"]
        task_def = task_ladder.task(c["task_index"])
        # Comparing task 1 against task 5 measures difficulty, not learning. The honest
        # baseline is the same task solved without memory.
        _, cold, _ = discover(task_def, search_order(c["task_index"], None), 10_000)
        saved = cold - p["evaluations"]
        saved_total += saved
        print(f"{c['task']:<22}{p['evaluations']:>9}{cold:>12}{saved:>7}  {p['transferred_from']}")

    transfers = sum(1 for c in caps
                    if c["provenance"]["transferred_from"] == c["representation"])

    print(f"\nacquisitions reusing a representation that already worked: {transfers}/{len(caps)}")
    print(f"total budget saved against cold start: {saved_total}")

    hashes = {c["provenance"]["anchor_integrity"] for c in caps}
    print(f"anchor versions across capabilities: {len(hashes)}"
          f"{' (consistent)' if len(hashes) == 1 else ' — capabilities were validated against different anchors'}")

    if saved_total > 0:
        verdict = f"memory saved {saved_total} evaluations against solving the same tasks cold"
    elif saved_total == 0:
        verdict = "memory saved nothing; the developmental claim is not supported by this run"
    else:
        verdict = f"memory cost {-saved_total} extra evaluations; transfer hurt on this ladder"

    paused = [h for h in history if h.get("status") == "pause"]
    print(f"\nverdict: {verdict}")
    if len(caps) < task_ladder.count():
        print(f"note: {task_ladder.count() - len(caps)} task(s) on the ladder remain unsolved")
    if paused:
        print(f"note: {len(paused)} recorded cycle(s) ended in pause")

if __name__ == "__main__":
    main()
