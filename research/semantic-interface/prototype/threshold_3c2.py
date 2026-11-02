"""Experiment 3c2 threshold from development data only (EXPERIMENTS.md, pre-registered rule): CSI's closeness threshold
from its dev run in record mode and bge-small's from its dev cosines. Candidates are midpoints between consecutive
distinct values; the highest dev balanced accuracy among candidates keeping every dev control phenomenon at <= 0.05
false equivalence wins (if none does: lowest maximum false equivalence, then balanced accuracy); ties go to the higher
threshold. Usage: python threshold_3c2.py <csi dev json> <bge json> <out json>"""
import hashlib
import json
import sys

INF = float("inf")


def margin(row, align=True):
    """Smallest threshold at which CSI v0.3 calls this pair equivalent: inf = always (no difference), -inf = never,
    None = unsupported (an error at any threshold). align=False drops the role condition (ablation)."""
    if row["pred"] == -1:
        return None
    diff = set(row["differences"]) | {e["slot"] for e in row["lexical"] if e["accepted"]}
    close = {e["slot"]: e["closeness"] for e in row["lexical"]
             if e.get("closeness") is not None and (e["aligned"] or not align)}
    if not diff:
        return INF
    return min(close[s] for s in diff) if diff <= set(close) else -INF


def v02(row):
    """3c's rule (shared synset membership) from the same analysis."""
    if row["pred"] == -1:
        return -1
    diff = set(row["differences"]) | {e["slot"] for e in row["lexical"] if e["accepted"]}
    shared = {e["slot"] for e in row["lexical"] if e.get("shared_meaning")}
    return int(diff <= shared)


def v01(row):
    if row["pred"] == -1:
        return -1
    return int(not row["differences"] and not any(e["accepted"] for e in row["lexical"]))


def decide(margins, tau):
    return [-1 if m is None else int(m >= tau) for m in margins]


def scores(rows, pred):
    """-> (balanced accuracy, {control phenomenon: false-equivalence rate}); unsupported pairs count as errors."""
    y = [r["label"] for r in rows]
    s = [p if p != -1 else 1 - t for p, t in zip(pred, y)]
    pos = [p for p, t in zip(s, y) if t == 1]
    neg = [p for p, t in zip(s, y) if t == 0]
    ba = (sum(pos) / len(pos) + sum(1 - p for p in neg) / len(neg)) / 2
    fe = {}
    for r, p in zip(rows, s):
        if r["label"] == 0:
            fe.setdefault(r["phenomenon"], []).append(p)
    return ba, {k: sum(v) / len(v) for k, v in sorted(fe.items())}


def choose(rows, margins):
    finite = sorted({m for m in margins if m is not None and abs(m) != INF})
    table = []
    for t in [(x + y) / 2 for x, y in zip(finite, finite[1:])]:
        ba, fe = scores(rows, decide(margins, t))
        table.append((t, ba, max(fe.values()), fe))
    ok = [c for c in table if c[2] <= 0.05]
    t, ba, mfe, fe = max(ok, key=lambda c: (c[1], c[0])) if ok else max(table, key=lambda c: (-c[2], c[1], c[0]))
    return {"tau": round(t, 6), "dev_balanced_accuracy": round(ba, 4), "dev_false_equivalence": {k: round(v, 4) for k, v in fe.items()},
            "candidates": len(table), "candidates_within_0.05": len(ok)}


def main():
    csi_path, bge_path, out = sys.argv[1:4]
    csi, bge = json.load(open(csi_path)), json.load(open(bge_path))
    assert csi["CSI_CLOSENESS"] == "record", "the CSI dev run must be in record mode"
    rows = csi["dev"]["rows"]
    res = {"rule": __doc__.split("\n")[1:4], "inputs_sha256": {p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
                                                              for p in (csi_path, bge_path)},
           "csi": choose(rows, [margin(r) for r in rows]),
           "bge": choose(bge["dev"]["rows"], [r["cos"] for r in bge["dev"]["rows"]]),
           "dev_items": len({r["item"] for r in rows}), "dev_pairs": len(rows)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
