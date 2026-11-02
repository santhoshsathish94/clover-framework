"""Show CSI vs gold for the first K errors of selected COGS gen cases (Experiment 6a failure analysis).
Usage: python cogs_examples.py <COGS data dir> <K> case [case ...]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cogs import csi_prop, gold_for, parse_lf  # noqa: E402

d, k, cases = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(d, "gen.tsv"), encoding="utf-8")]
for case in cases:
    print("==", case)
    shown = 0
    for text, lf, c in rows:
        if c != case:
            continue
        events, main = parse_lf(lf)
        prop, unsup = csi_prop(text)
        gold = gold_for(prop, events, main)[0] if prop else events[main]
        ok = prop is not None and gold is not None and prop["predicate"] == gold["verb"] and prop["roles"] == gold["roles"]
        if ok:
            continue
        print(" ", text, "| gold", gold and (gold["verb"], gold["roles"]), "| csi",
              (prop["predicate"], prop["roles"], prop["embedded_under"]) if prop else unsup)
        shown += 1
        if shown >= k:
            break
