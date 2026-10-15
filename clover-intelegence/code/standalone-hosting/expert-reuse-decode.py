#!/usr/bin/env python3
"""Expert reuse between consecutive GENERATED tokens, and what cache capacity captures it.

WHY THIS EXISTS, AND HOW IT DIFFERS FROM expert-reuse.py
The Oct 8 script next to this one asked the same question of the PREFILL routes, using a
single global LRU keyed by (layer, expert). Two things make that the wrong instrument for
the question now in front of us:

  - the regime. Prefill is five positions of one prompt. A long answer is hundreds of
    generated tokens, each fed back in, and nothing had measured reuse there.
  - the cache shape. The engine's cache is NOT global. root_pull_begin computes
        held = (root_cache_store && layer < root_cache_layers &&
                count <= root_cache_capacity) ? root_cache_capacity : 0
        base = layer * held
    so every layer gets its own `capacity` slots and evicts within itself. A global LRU
    of the same total size would report a different, more optimistic number.

Reuse is the only lossless lever left on expert I/O. Lowering k is an approximation whose
divergence horizon moves with k: measured, k=12 matched k=16 for all 64 tokens on the
France prompt but diverged at token 19 on "the chemical symbol for gold" and token 16 on
"water boils at a temperature of". So k stays at 16 and the question becomes how often an
expert is already in memory.

INPUT
observe-route64 seeds the five-token France prompt, feeds each generated token back in,
and records every expert actually computed, via the CLOVER_CHECK_EXPERT hook, per layer
per position. 64 positions x 92 layers x 16 experts = 94,208 selections.

TWO CHECKS BEFORE ANY NUMBER BELOW IS WORTH READING
  1. Prefill self-check. THE-FLOW-OBSERVED.md records mean pairwise overlap of 2.30 of 16
     across the five prompt positions. Positions 0-4 here must reproduce that, or the
     capture is not seeing what the engine does.
  2. Simulator validation. The engine's own STEP_JSON for this exact run reported 39,280
     hits and 54,928 misses at K3_EXPERT_CACHE=24 -- 41.7%. A per-layer LRU at capacity 24
     must land near it, or the cache model is wrong and the capacity sweep is fiction.
"""
import json
import re
import sys
from collections import OrderedDict

PATH = sys.argv[1] if len(sys.argv) > 1 else "route64.out"
SEEDED = 5
EXPERT_GB = 17547264 / float(1 << 30)

sel = {}
for line in open(PATH):
    m = re.match(r"^ROUTE (\{.*\})\s*$", line)
    if m:
        r = json.loads(m.group(1))
        sel[(r["position"], r["layer"])] = r["experts"]

positions = sorted({p for p, _ in sel})
layers = sorted({l for _, l in sel})
K = len(next(iter(sel.values())))
print("trace: %d positions, %d layers, k=%d, %d expert selections"
      % (len(positions), len(layers), K, len(sel) * K))


def overlap(lo, hi):
    v = [len(set(sel[(p, l)]) & set(sel[(p + 1, l)]))
         for p in range(lo, hi - 1) for l in layers
         if (p, l) in sel and (p + 1, l) in sel]
    return (sum(v) / float(len(v)), min(v), max(v), len(v)) if v else (0, 0, 0, 0)


print()
print("CHECK 1 -- overlap between consecutive positions, out of %d" % K)
for label, lo, hi in [("prefill  (0-4)", 0, SEEDED),
                      ("decode   (5-63)", SEEDED, len(positions))]:
    mean, mn, mx, n = overlap(lo, hi)
    print("  %-16s mean %5.2f   min %2d   max %2d   over %d layer-pairs" % (label, mean, mn, mx, n))

# The recorded 2.30 is a different statistic: mean over ALL TEN pairs of the five prompt
# positions, reported as a median ACROSS pods. Consecutive pairs are the most similar, so
# comparing consecutive-only against all-pairs overstates agreement. Computed both ways.
per_pod = []
for l in layers:
    v = [len(set(sel[(a, l)]) & set(sel[(b, l)]))
         for a in range(SEEDED) for b in range(a + 1, SEEDED)
         if (a, l) in sel and (b, l) in sel]
    if v:
        per_pod.append(sum(v) / float(len(v)))
per_pod.sort()
print("  prefill, ALL pairs, per pod: min %.2f  median %.2f  max %.2f"
      % (per_pod[0], per_pod[len(per_pod) // 2], per_pod[-1]))
print("  THE-FLOW-OBSERVED.md records min 0.70 / median 2.30 / max 4.50 for exactly this")


def simulate(cap, lo, hi):
    """Per-layer LRU, hits resolved for the whole batch before any insertion, which is
       what root_pull_begin does so a miss cannot evict an expert the same batch needs."""
    hits = misses = 0
    caches = {l: OrderedDict() for l in layers}
    for p in range(lo, hi):
        for l in layers:
            experts = sel.get((p, l))
            if not experts:
                continue
            c = caches[l]
            fresh = []
            for e in experts:
                if cap and e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    misses += 1
                    fresh.append(e)
            for e in fresh:
                if cap:
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
    return hits, misses


print()
print("CHECK 2 and the capacity sweep -- per-layer LRU, all %d positions" % len(positions))
print("  %8s %10s %11s %14s %10s" % ("capacity", "hit rate", "GB/token", "cache RAM", "vs none"))
base = None
for cap in (0, 8, 16, 24, 32, 48, 64, 96, 128):
    h, m = simulate(cap, 0, len(positions))
    rate = 100.0 * h / (h + m)
    gb = m * EXPERT_GB / len(positions)
    if cap == 0:
        base = gb
    ram = cap * len(layers) * EXPERT_GB
    print("  %8d %9.1f%% %11.2f %11.1f GB %10s"
          % (cap, rate, gb, ram, "-" if cap == 0 else "%.2fx" % (base / gb)))
print("  engine measured 39280 hits / 54928 misses at capacity 24 on this run = 41.7%")

print()
print("decode only (positions %d-%d), which is what a long answer pays" % (SEEDED, len(positions) - 1))
print("  %8s %10s %11s" % ("capacity", "hit rate", "GB/token"))
for cap in (0, 16, 24, 32, 48, 64, 96, 128):
    h, m = simulate(cap, SEEDED, len(positions))
    rate = 100.0 * h / (h + m)
    gb = m * EXPERT_GB / (len(positions) - SEEDED)
    print("  %8d %9.1f%% %11.2f" % (cap, rate, gb))

print()
sizes = sorted(len({e for p in positions for e in sel.get((p, l), [])}) for l in layers)
print("working set: distinct experts a layer touches over %d positions" % len(positions))
print("  min %d   median %d   max %d   of 896" % (sizes[0], sizes[len(sizes) // 2], sizes[-1]))
print("  holding every expert ever touched would need %.0f GB" % (sum(sizes) * EXPERT_GB))
