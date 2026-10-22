"""Cycle 21: the exit rule over every latent span in the existing latent-allowed traces (no new run).
For each span: how it ended (anchor exit or forced at max), P(</latent>) from the LM head at the exit step,
cosine of R(h) to the </latent> embedding, whether the two readouts agree, and how separated the exit step is
from the steps that continued."""
import glob
import json
import os
import re
import statistics as st

J = "/root/runs/hybrid/"
files = [f for f in sorted(glob.glob(J + "trace-q*.json")) if "nolat" not in f and "-one-" not in f]
rows = []
tot = dict(files=0, spans=0, anchor=0, forced=0, lm_only=0, cos_only=0, both=0, neither=0, suppressed=0)
exit_p, exit_cos, cont_p, cont_cos, lens = [], [], [], [], []
for f in files:
    name = os.path.basename(f)[:-5]
    d = json.load(open(f))
    S, ids, r = d["steps"], d["ids"], d["result"]
    if not r.get("latent_spans"):
        continue
    tot["files"] += 1
    tot["forced"] += r.get("forced_exits", 0)
    tot["anchor"] += r.get("anchor_exits", 0)
    cur = None
    n = 0
    for s in S:
        if s["mode_before"] == "DECODE" and s.get("tok") == ids["open"]:
            cur = 0
        elif s["mode_before"] == "LATENT" and cur is not None:
            lm_close = s["lm_top1"] == ids["close"]
            nn_close = s["link_nn_id"] == ids["close"]
            if s.get("tok") == ids["close"]:
                tot["spans"] += 1
                lens.append(cur)
                exit_p.append(s["lm_p_close"])
                exit_cos.append(s["link_cos_close"])
                if lm_close and nn_close:
                    tot["both"] += 1
                elif lm_close:
                    tot["lm_only"] += 1
                elif nn_close:
                    tot["cos_only"] += 1
                else:
                    tot["neither"] += 1
                cur = None
            else:
                if lm_close and cur < (16 if "b16" in name else 4):
                    tot["suppressed"] += 1
                cont_p.append(s["lm_p_close"])
                cont_cos.append(s["link_cos_close"])
                cur += 1
print(f"files with latent spans: {tot['files']}")
print({k: v for k, v in tot.items()})
print(f"span lengths (continuing latent steps before the exit step): n {len(lens)}, median {st.median(lens)}, mean {st.mean(lens):.1f}, min {min(lens)}, max {max(lens)}")
print("lengths histogram (<=8, 9-16, 17-32, 33-64, 65-128, 129+):", [sum(1 for x in lens if a <= x <= b) for a, b in ((0, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**6))])
print(f"at the exit step: LM-head P(</latent>) median {st.median(exit_p):.3f}, min {min(exit_p):.3f}, share > 0.5: {sum(p > 0.5 for p in exit_p) / len(exit_p):.2f}, share > 0.9: {sum(p > 0.9 for p in exit_p) / len(exit_p):.2f}")
print(f"at the exit step: cosine(R(h), Emb(</latent>)) median {st.median(exit_cos):.3f}, min {min(exit_cos):.3f}")
print(f"continued steps: n {len(cont_p)}, max P(</latent>) {max(cont_p):.3f}, share P > 0.1: {sum(p > 0.1 for p in cont_p) / len(cont_p):.4f}, share P > 0.5: {sum(p > 0.5 for p in cont_p) / len(cont_p):.4f}; cosine to </latent> median {st.median(cont_cos):.3f}, max {max(cont_cos):.3f}")
