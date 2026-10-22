"""Distributions over steps of per-layer values (layer index given by args) from a trace json."""
import json
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))
S = d["steps"]
close, opn = d["ids"]["close"], d["ids"]["open"]
L = [int(x) for x in sys.argv[2:]] or [16, 35]


def q(v):
    v = sorted(v)
    n = len(v)
    return [round(v[min(n - 1, int(p * n))], 3) for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]


tx = [s for s in S if s["kind"] == "token" and s["in_id"] not in (close, opn)]
lt = [s for s in S if s["kind"] == "latent"]
print("quantiles [min, p10, p25, p50, p75, p90, max]; text n=%d latent n=%d" % (len(tx), len(lt)))
for i in L:
    for j, name in ((0, "out_norm"), (1, "rel_update"), (2, "cos_in_out")):
        print(f"layer {i} {name}: text {q([s['layers'][i][j] for s in tx])} | latent {q([s['layers'][i][j] for s in lt])}")

print("\nlayer-16 relative update, text steps: count above 0.9 =", sum(1 for s in tx if s["layers"][16][1] > 0.9), "of", len(tx))
hi = [s for s in tx if s["layers"][16][1] > 0.9]
print("input ids of those steps (top 8):", Counter(s["in_id"] for s in hi).most_common(8))
lo = [s for s in tx if s["layers"][16][1] <= 0.9]
print("mean layer-16 rel update, text steps below/at 0.9:", round(sum(s["layers"][16][1] for s in lo) / max(1, len(lo)), 3))
print("share of text steps whose input id is the most common id overall:", Counter(s["in_id"] for s in tx).most_common(5))

print("\nlayer-35 output norm (pre final norm), by step index in the run (every 25th step; kind)")
for k in range(0, len(S), 25):
    s = S[k]
    print(k, s["kind"], round(s["layers"][35][0], 1), "rel", round(s["layers"][35][1], 3), "final-normed norm", round(s["out_norm"], 1))

print("\nlatent steps by position in span (first 5, middle, last 5 of the longest span):")
spans, cur = [], None
for k, s in enumerate(S):
    if s["mode_before"] == "DECODE" and s.get("tok") == opn:
        cur = []
    elif s["mode_before"] == "LATENT" and cur is not None:
        if s.get("tok") == close:
            spans.append(cur)
            cur = None
        else:
            cur.append(k)
if spans:
    best = max(spans, key=len)
    sel = best[:5] + best[len(best) // 2 - 2:len(best) // 2 + 3] + best[-5:]
    for k in sel:
        s = S[k]
        print("step", k, "pos-in-span", best.index(k), "L16 upd", round(s["layers"][16][1], 3), "L35 norm", round(s["layers"][35][0], 1), "emb_norm", round(s["emb_norm"], 2), "nn_cos", round(s["link_nn_cos"], 3), "p_close", round(s["lm_p_close"], 4))
    for sp in spans:
        print("span len", len(sp), "mean L16 upd", round(sum(S[k]["layers"][16][1] for k in sp) / len(sp), 3), "mean L35 norm", round(sum(S[k]["layers"][35][0] for k in sp) / len(sp), 1), "mean nn_cos", round(sum(S[k]["link_nn_cos"] for k in sp) / len(sp), 3))
