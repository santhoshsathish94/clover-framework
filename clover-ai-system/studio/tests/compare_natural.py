"""Compares natural-regression runs turn by turn: which turns each run passed, what changed, time per run.
Usage: python tests/compare_natural.py A.jsonl B.jsonl [C.jsonl ...]"""
import json
import sys
from pathlib import Path


def load(p):
    head, turns = {}, {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "head" in r:
            head = r["head"]
        else:
            turns[(r["scenario"], str(r["turn"]))] = r
    return head, turns


def main():
    runs = [(Path(p).stem, *load(p)) for p in sys.argv[1:]]
    keys = list(dict.fromkeys(k for _, _, t in runs for k in t))
    names = [n for n, _, _ in runs]
    w = max(12, *(len(n) for n in names))
    print(f"{'scenario':16} {'turn':>4}  " + "  ".join(f"{n[:w]:>{w}}" for n in names))
    for k in keys:
        cells = []
        for _, _, t in runs:
            r = t.get(k)
            cells.append("-" if r is None else ("PASS" if r["ok"] else "fail") + (f" {r.get('seconds') or 0:>5.1f}s" if r else ""))
        if len({c.split()[0] for c in cells}) > 1:
            mark = "  <- differs"
        else:
            mark = ""
        print(f"{k[0]:16} {k[1]:>4}  " + "  ".join(f"{c:>{w}}" for c in cells) + mark)
    print()
    for n, h, t in runs:
        ok = sum(r["ok"] for r in t.values())
        secs = sum(r.get("seconds") or 0 for r in t.values() if r["scenario"] != "security")
        print(f"{n}: {ok}/{len(t)} passed, {secs:.0f} s in turns; engine {((h.get('engine') or {}).get('engine'))}, "
              f"started {h.get('started')}")


if __name__ == "__main__":
    main()
