"""Group the CSI errors of a run_e1.py result by phenomenon and difference pattern, with examples (development data).
Usage: python e1_errors.py <run_e1 json> [lexicon ...]"""
import collections
import json
import sys

r = json.load(open(sys.argv[1]))
for lex in sys.argv[2:] or r["csi"].keys():
    errs = r["csi"][lex]["errors"]
    print(f"== {lex}: {len(errs)} errors; per phenomenon {r['csi'][lex]['per_phenomenon_accuracy']}")
    pat = collections.defaultdict(list)
    for e in errs:
        key = (e["phenomenon"], e["pred"], tuple(sorted(d["type"] for d in e["differences"])) or tuple(e["notes"]))
        pat[key].append(e)
    for key, es in sorted(pat.items(), key=lambda kv: -len(kv[1])):
        e = es[0]
        print(f"  {len(es):3d}  {key[0]} pred={key[1]} {key[2]}")
        print(f"       e.g. {e['a']} | {e['b']} | {e['differences'][:3]} {e['notes'][:1]}")
