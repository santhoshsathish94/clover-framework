"""Experiment 8 analysis: where the LLM with CSI's verdict (B2) and CSI alone (C) disagree, item by item, and the
LLM-alone (A) vs B1 changes per phenomenon. Usage: python abc_disagree.py <e8.json>"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import compare  # noqa: E402
from csi.minimal_pairs import generate  # noqa: E402

res = json.load(open(sys.argv[1]))
out = {}
for lex in ("seen", "unseen"):
    rows = generate(lex)
    sc = {c: res["conditions"][c]["scores"][lex] for c in ("A", "B1", "B2")}
    c_pred = [int(bool(compare(r["a"], r["b"])["equivalent"])) for r in rows]
    dis = collections.Counter()
    items = []
    for i, r in enumerate(rows):
        b2 = int(sc["B2"][i] > 0)
        if b2 != c_pred[i]:
            kind = "LLM corrected CSI" if b2 == r["label"] else "LLM overrode a correct CSI verdict"
            dis[kind] += 1
            items.append({"phenomenon": r["phenomenon"], "a": r["a"], "b": r["b"], "label": r["label"],
                          "csi": c_pred[i], "b2": b2, "b2_score": sc["B2"][i], "kind": kind})
    flips = collections.Counter()
    for i, r in enumerate(rows):
        a_ok, b1_ok = int(sc["A"][i] > 0) == r["label"], int(sc["B1"][i] > 0) == r["label"]
        if a_ok != b1_ok:
            flips[("fixed by B1" if b1_ok else "broken by B1", r["phenomenon"])] += 1
    out[lex] = {"b2_vs_c": dict(dis), "b2_vs_c_items": items,
                "a_to_b1_flips": {f"{k[0]} | {k[1]}": v for k, v in sorted(flips.items())}}
    print(lex, json.dumps(out[lex], indent=1))
json.dump(out, open(sys.argv[1].replace(".json", "_disagree.json"), "w"), indent=1)
