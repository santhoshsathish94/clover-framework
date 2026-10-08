#!/usr/bin/env python3
"""A per-layer expert cache, simulated against a real route trace.

The earlier global-LRU result was flat at 23.3% for every capacity. That is the
cyclic-sweep pathology: one token pushes 92x16 fresh keys through the cache, so
layer L's entries are always evicted before the next token comes back to layer L,
92 layers later.

Partitioning the cache per layer removes the competition entirely. Layer L keeps
its own small set and the only traffic through it is layer L's own, one visit per
token. This replays the trace against that structure.
"""
import sys
from collections import defaultdict, OrderedDict

rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    f = line.split()
    if len(f) >= 4:
        rows.append((int(f[0]), int(f[1]), [int(x) for x in f[3:]]))

steps, cur, last = [], defaultdict(dict), None
for layer, pos, ids in rows:
    if last is not None and layer < last:
        steps.append(cur); cur = defaultdict(dict)
    cur[layer][pos] = ids
    last = layer
if cur:
    steps.append(cur)

prompt = steps[0]
decode = steps[1:]
layers = sorted(prompt)
EXPERT_MB = 17.547264

print("trace: %d layers, %d decode tokens" % (len(layers), len(decode)))
print()
print("  cap/layer      RAM      warm hit%   steady hit%   read/token   stall cover")
print("  " + "-" * 74)

for cap in (0, 8, 16, 24, 32, 48, 64):
    cache = {L: OrderedDict() for L in layers}
    # the prompt pass warms it
    for L, bypos in prompt.items():
        for pos in sorted(bypos):
            for e in bypos[pos]:
                if e in cache[L]:
                    cache[L].move_to_end(e)
                else:
                    cache[L][e] = 1
                    while len(cache[L]) > cap:
                        cache[L].popitem(last=False)
    hits = total = 0
    late_hits = late_total = 0
    for k, s in enumerate(decode):
        for L in layers:
            for e in s[L][0]:
                total += 1
                if k >= len(decode) // 2:
                    late_total += 1
                if e in cache[L]:
                    hits += 1
                    if k >= len(decode) // 2:
                        late_hits += 1
                    cache[L].move_to_end(e)
                else:
                    cache[L][e] = 1
                    while len(cache[L]) > cap:
                        cache[L].popitem(last=False)
    ram = len(layers) * cap * EXPERT_MB / 1000.0
    hr = hits / total if total else 0
    lhr = late_hits / late_total if late_total else 0
    read_gb = 16 * len(layers) * EXPERT_MB / 1000.0 * (1 - hr)
    # a layer can hide 14.1 ms of read behind shared expert + projection
    per_layer_ms = read_gb * 1000 / 13.5 / len(layers)
    cover = "yes" if per_layer_ms <= 14.1 else "no  (%.1f ms > 14.1)" % per_layer_ms
    print("  %9d  %6.1f GB   %8.1f%%   %9.1f%%   %7.1f GB   %s" % (
        cap, ram, 100 * hr, 100 * lhr, read_gb, cover))

print()
print("  RAM budget: 124 GB total - 75.9 GB payload = ~48 GB free")
print("  a layer must get its read under 14.1 ms to hide it behind its own compute")
