#!/usr/bin/env python3
"""Two questions about layer-level memoisation, answered from recorded vectors.

1. Does the same 7168 input ever arrive at the same layer twice?
   Statistics are a necessary condition for identity: different stats prove
   different vectors, so zero stat collisions proves zero repeats.
2. Is a layer's output a function of its input alone, or does carried state
   also decide it?
"""
import collections
import sys

rows = []
with open("/tmp/lstat.tsv") as handle:
    for line in handle:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 12 and f[0] == "V":
            rows.append(f)

def collisions(name):
    seen = collections.defaultdict(list)
    for f in rows:
        if f[3] != name:
            continue
        # Key on the layer plus every recorded statistic of the vector.
        seen[(f[1],) + tuple(f[5:12])].append(f[2])
    total = sum(len(v) for v in seen.values())
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    return total, len(seen), dup

for name in ("1_resid_in_aggregated", "7_resid_out", "5_x2_postnorm"):
    total, distinct, dup = collisions(name)
    print(f"{name:26s} records {total:5d}  distinct {distinct:5d}  colliding keys {len(dup)}")
    for k, v in list(dup.items())[:3]:
        print(f"    layer {k[0]} seen at positions {v}")

print()
print("KDA state per layer across steps (does carried state change?):")
state = collections.defaultdict(list)
for f in rows:
    if f[3] == "8_kda_state":
        state[f[1]].append(tuple(f[5:12]))
layers = sorted(state, key=int)
changing = sum(1 for L in layers if len(set(state[L])) > 1)
print(f"  layers recorded {len(layers)}, layers whose state changes {changing}")
for L in layers[:3]:
    vals = state[L]
    print(f"  layer {L}: {len(vals)} steps, {len(set(vals))} distinct states, rms "
          + " ".join(v[3] for v in vals[:5]))

print()
print("Same layer, consecutive positions - how far apart are the inputs?")
by_layer = collections.defaultdict(list)
for f in rows:
    if f[3] == "1_resid_in_aggregated":
        by_layer[int(f[1])].append((int(f[2]), float(f[8])))
for L in (0, 46, 92):
    if L in by_layer:
        seq = sorted(by_layer[L])[:6]
        print(f"  layer {L:2d} rms by position: " + " ".join(f"{r:.6g}" for _, r in seq))
