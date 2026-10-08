#!/usr/bin/env python3
"""Is the expert route predictable enough to prefetch?

A layer cannot hide its own expert reads, so the ids have to come from somewhere
earlier than its own router. Three candidate sources, measured here:

  next layer   - does layer L's choice predict layer L+1's?
  next token   - does this token's choice at layer L predict the next token's?
  working set  - how many distinct experts does a layer ever touch?

Only the third needs no prediction at all: if the set is small enough to hold in
RAM, the read disappears instead of being hidden.
"""
import sys
from collections import defaultdict

rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    f = line.split()
    if len(f) < 4:
        continue
    rows.append((int(f[0]), int(f[1]), int(f[2]), [int(x) for x in f[3:]]))

# rows arrive layer-major within an evaluation; a drop back to layer 1 starts a new one
steps, cur, last = [], defaultdict(dict), None
for layer, pos, tok, ids in rows:
    if last is not None and layer < last:
        steps.append(cur); cur = defaultdict(dict)
    cur[layer][pos] = ids
    last = layer
if cur:
    steps.append(cur)

print("evaluations parsed: %d" % len(steps))
for i, s in enumerate(steps):
    widths = {len(v) for v in s.values()}
    print("  step %d: %d layers, %s position(s) each" % (i, len(s), widths.pop()))

decode = [s for s in steps if len(next(iter(s.values()))) == 1]
print("\ndecode steps: %d" % len(decode))
sets = [{L: set(v[0]) for L, v in s.items()} for s in decode]
layers = sorted(sets[0])

def pct(x):
    return 100.0 * x

# 1. next layer, same token
same = [len(sets[k][L] & sets[k][L + 1]) / 16.0
        for k in range(len(sets)) for L in layers if L + 1 in sets[k]]
print("\nlayer L predicts layer L+1   : %.1f%% of 16 (n=%d)" % (pct(sum(same) / len(same)), len(same)))

# 2. same layer, next token
nxt = [len(sets[k][L] & sets[k + 1][L]) / 16.0
       for k in range(len(sets) - 1) for L in layers]
print("token N predicts token N+1   : %.1f%% of 16 (n=%d)" % (pct(sum(nxt) / len(nxt)), len(nxt)))

# 3. union of every earlier token
cov, held = [], {L: set() for L in layers}
for k, s in enumerate(sets):
    if k:
        cov.append(sum(len(s[L] & held[L]) / 16.0 for L in layers) / len(layers))
    for L in layers:
        held[L] |= s[L]
for k, c in enumerate(cov, start=1):
    print("  union of tokens 0..%d covers token %d : %.1f%%" % (k - 1, k, pct(c)))

# 4. working set per layer
distinct = [len(held[L]) for L in layers]
total = sum(distinct)
print("\ndistinct experts touched over %d tokens" % len(sets))
print("  per layer: %.1f mean, %d min, %d max, of 896" % (
    total / len(layers), min(distinct), max(distinct)))
print("  whole model: %d expert slots = %.1f GB at 17.547 MB each" % (total, total * 17.547264 / 1000))
print("  a token reads %d = %.1f GB" % (16 * len(layers), 16 * len(layers) * 17.547264 / 1000))

# 5. how much does the prompt pass already supply?
prompt = [s for s in steps if len(next(iter(s.values()))) > 1]
if prompt:
    pset = {L: set().union(*[set(v) for v in d.values()]) for L, d in prompt[0].items()}
    for k, s in enumerate(sets[:3]):
        c = sum(len(s[L] & pset[L]) / 16.0 for L in layers) / len(layers)
        print("  prompt pass covers decode token %d : %.1f%%" % (k, pct(c)))
    print("  prompt pass touched %d expert slots = %.1f GB" % (
        sum(len(v) for v in pset.values()), sum(len(v) for v in pset.values()) * 17.547264 / 1000))
