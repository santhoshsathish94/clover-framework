"""T2: Learning Concepts reproduction table. Reads the released eval_mteb.py CSVs (lc-results/sts_seed*.csv,
sts_rand_seed*.csv), averages the 9 STS tasks per (condition, seed), then mean and range over seeds, next to the
paper's Table 3 (Falcon 3 1B, C4, embedding synonyms, "Before"): pretrained 53.28, NTP-only 54.35, concept 55.11,
randomized lambda 0.25 54.04, randomized lambda 1.0 50.61.
Usage: python lc_table.py <lc-results dir> <out json>"""
import csv
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

d, out = sys.argv[1], sys.argv[2]
PAPER = {"pretrained": 53.28, "ntp_only (lambda 0)": 54.35, "concept (lambda 1)": 55.11,
         "randomized lambda 0.25": 54.04, "randomized lambda 1.0": 50.61}
per = defaultdict(lambda: defaultdict(dict))  # condition -> seed -> task -> score
for f in sorted(glob.glob(os.path.join(d, "sts_*seed*.csv"))):
    seed = f.split("seed")[-1].split(".")[0]
    rand = "_rand_" in os.path.basename(f)
    for r in csv.DictReader(open(f)):
        w = r["adapter_weight"]
        if w == "base":
            cond = "pretrained"
        elif rand:
            cond = f"randomized lambda {float(w):.2f}".replace("1.00", "1.0")
        else:
            cond = {"0.0": "ntp_only (lambda 0)", "1.0": "concept (lambda 1)"}.get(w, f"lambda {w}")
        per[cond][seed][r["eval_task"]] = 100 * float(r["main_score"])
table = {}
for cond, seeds in per.items():
    means = {s: st.mean(t.values()) for s, t in seeds.items() if len(t) == 9}
    if not means:
        continue
    vals = list(means.values())
    tasks = sorted({k for t in seeds.values() for k in t})
    table[cond] = {"mean": round(st.mean(vals), 2), "min": round(min(vals), 2), "max": round(max(vals), 2),
                   "seeds": {s: round(v, 2) for s, v in sorted(means.items())}, "paper": PAPER.get(cond),
                   "per_task_mean": {k: round(st.mean(t[k] for t in seeds.values() if k in t), 2) for k in tasks}}
json.dump(table, open(out, "w"), indent=1)
order = list(PAPER) + [c for c in table if c not in PAPER]
print(f"{'condition':26} {'ours mean':>9} {'range':>13} {'paper':>6} {'diff':>6}  seeds")
for c in order:
    if c in table:
        t = table[c]
        diff = f"{t['mean'] - t['paper']:+.2f}" if t["paper"] is not None else ""
        print(f"{c:26} {t['mean']:9.2f} {t['min']:6.2f}-{t['max']:<6.2f} {t['paper'] or '':>6} {diff:>6}  {t['seeds']}")
