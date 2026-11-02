"""Experiment 2 decision table: dev/test macro-F1 over the convention-neutral CSI relations (the corrected rule) plus
UPOS, XPOS, lemma, UAS/LAS (UD-native only), iobj F1 and double-object counts, for every system.
Usage: python e3_decide.py <e3_pretrained.json> <e3_ours.json>"""
import json
import sys

RELS = ["nsubj", "nsubj:pass", "obj", "obl:agent", "aux:pass", "xcomp", "negator"]
for path in sys.argv[1:]:
    d = json.load(open(path))
    for name, s in d["systems"].items():
        short = name.rstrip("/").split("/")[-2] if "/" in name else name
        for sp in ("dev", "test"):
            x = s[sp]
            m = sum(x["relations"][r]["f1"] for r in RELS) / len(RELS)
            print(f"{short:18s} {sp:4s} rule={m:.4f} upos={x['upos']} xpos={x['xpos']} lemma={x['lemma_ci']} "
                  f"uas={x.get('uas', '-')} las={x.get('las', '-')} iobj={x['relations']['iobj']['f1']} "
                  f"dobj={x['double_object']}")
        feats = s["test"]["features"]
        print(f"{'':18s} test features: " + ", ".join(f"{k} {v['acc']}" for k, v in feats.items() if v["acc"] is not None))
