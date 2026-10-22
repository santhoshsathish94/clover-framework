"""Cycle 17: per run, separate generation steps (text tokens + latent steps) from observation-feed forward passes.
Pairs = same question and seed, latent allowed vs `<latent>` blocked. Reads the existing trace-*.json only."""
import collections
import json
import re
import sys

J = "/root/runs/hybrid/"
PAIRS = [  # label, latent-allowed file, latent-blocked file
    ("q4 s0", "trace-q4-search", "trace-q4-nolat-search"),
    ("q4 s1", "trace-q4-s1", "trace-q4-nolat-s1"),
    ("q4 s2", "trace-q4-s2", "trace-q4-nolat-s2"),
    ("q5 s0", "trace-q5-search", "trace-q5-nolat"),
    ("q5 s1", "trace-q5-s1", "trace-q5-nolat-s1"),
    ("q5 s2", "trace-q5-s2", "trace-q5-nolat-s2"),
    ("q1 s0", "trace-q1-search", "trace-q1-nolat-search"),
    ("q2 s0", "trace-q2-search", "trace-q2-nolat-search"),
    ("q3 s0", "trace-q3-search", "trace-q3-nolat-search"),
]


def load(name):
    d = json.load(open(J + name + ".json"))
    S, r = d["steps"], d["result"]
    c = collections.Counter(s["mode_before"] + "/" + s["kind"] for s in S)
    gen_steps = r["generation_steps"]
    lat = r["latent_steps"]
    feed = c.get("FEED/token", 0)
    t_gen = sum(s["t_total"] for s in S if s["mode_before"] != "FEED")
    t_feed = sum(s["t_total"] for s in S if s["mode_before"] == "FEED")
    ans = bool(re.search(r"boxed|<answer>", d["output"]))
    return dict(name=name, total=len(S), gen=gen_steps, lat=lat, text=gen_steps - lat, feed=feed,
                t_gen=t_gen, t_feed=t_feed, wall=d["wall_s"], end=r["end_reason"], ans=ans,
                py=r.get("python_calls", 0), se=r.get("search_calls", 0), spans=r["latent_spans"])


rows = []
for lab, a, b in PAIRS:
    A, B = load(a), load(b)
    rows.append((lab, A, B))
    print(f"{lab}: allowed {a}: total {A['total']} gen {A['gen']} (text {A['text']}, latent {A['lat']}, spans {A['spans']}) feed {A['feed']} "
          f"t_gen {A['t_gen']:.0f}s t_feed {A['t_feed']:.0f}s end {A['end']} ans {A['ans']} py {A['py']} se {A['se']}")
    print(f"      blocked {b}: total {B['total']} gen {B['gen']} (text {B['text']}) feed {B['feed']} "
          f"t_gen {B['t_gen']:.0f}s t_feed {B['t_feed']:.0f}s end {B['end']} ans {B['ans']} py {B['py']} se {B['se']}")
    print(f"      ratios blocked/allowed: text tokens {B['text'] / max(A['text'], 1):.2f}, generation steps {B['gen'] / max(A['gen'], 1):.2f}, "
          f"all forward passes {B['total'] / A['total']:.2f}, generation-step time {B['t_gen'] / A['t_gen']:.2f}")

def tot(k, side):
    return sum((r[1] if side == 0 else r[2])[k] for r in rows)

print()
for k in ("text", "lat", "gen", "feed", "total"):
    print(f"sum {k}: allowed {tot(k, 0)} blocked {tot(k, 1)}")
print(f"sum t_gen: allowed {tot('t_gen', 0):.0f}s blocked {tot('t_gen', 1):.0f}s; t_feed: allowed {tot('t_feed', 0):.0f}s blocked {tot('t_feed', 1):.0f}s")
print("pooled text-token reduction with latent allowed vs blocked: %.1f%%" % (100 * (1 - tot("text", 0) / tot("text", 1))))
print("pooled generation-step reduction: %.1f%%" % (100 * (1 - tot("gen", 0) / tot("gen", 1))))
print("pooled all-forward-pass reduction: %.1f%%" % (100 * (1 - tot("total", 0) / tot("total", 1))))
# pairs where both runs finished (not budget)
fin = [r for r in rows if r[1]["end"] == "eos" and r[2]["end"] == "eos"]
print("pairs both finished:", [r[0] for r in fin])
for k in ("text", "gen", "total"):
    a = sum(r[1][k] for r in fin); b = sum(r[2][k] for r in fin)
    print(f"  finished-only {k}: allowed {a} blocked {b} reduction {100 * (1 - a / b):.1f}%")
