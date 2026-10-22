"""Step time and stage times against position in the sequence (prompt 529 + step index)."""
import json
import sys

d = json.load(open(sys.argv[1]))
S = d["steps"]
P0 = d["prompt_len"]
close, opn = d["ids"]["close"], d["ids"]["open"]


def net(s):
    return s["t_total"] - s["obs"]


print("bucket of step index | n | net ms mean | layers ms | head ms | text-only net ms | latent-only net ms | position range")
B = 150
for a in range(0, len(S), B):
    seg = S[a:a + B]
    tx = [s for s in seg if s["kind"] == "token"]
    lt = [s for s in seg if s["kind"] == "latent"]
    m = lambda v, f: (sum(f(s) for s in v) / len(v) * 1e3) if v else float("nan")
    print(f"{a:5d}-{a + len(seg) - 1:5d} | {len(seg):3d} | {m(seg, net):6.1f} | {m(seg, lambda s: s['t_layers']):6.1f} | {m(seg, lambda s: s['t_head']):5.1f} | {m(tx, net):6.1f} | {m(lt, net):6.1f} | {P0 + a}-{P0 + a + len(seg)}")

import statistics as st
x = [i for i in range(len(S))]
y = [net(s) * 1e3 for s in S]
mx, my = st.mean(x), st.mean(y)
slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
print("least-squares slope of net step time against step index: %.4f ms per step (about %.1f ms per 1000 positions)" % (slope, slope * 1000))
print("first 50 steps mean net ms", round(sum(y[1:51]) / 50, 1), "| last 50 steps mean", round(sum(y[-51:-1]) / 50, 1))
