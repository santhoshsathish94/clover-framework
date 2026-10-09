#!/usr/bin/env python3
"""If the experts were spread across pods, would N pods give N times the speed?

Expert parallelism puts a slice of each layer's 896 experts on each pod. A token
selects 16, they land wherever they land, the pods compute in parallel, and the
layer is not finished until the slowest pod is. So the layer's cost is not
16/N, it is the largest number of selected experts that any one pod holds.

  speedup against one node doing all sixteen = 16 / max_over_pods(count)

Two structural facts this has to expose. Sixteen experts cannot occupy more than
sixteen pods, so beyond N=16 the critical path cannot shrink further however
many pods are added. And sixteen balls into N bins collide, so even at N=16 the
deepest bin holds more than one.

Three assignments: round robin, contiguous blocks, and a greedy balance using
each expert's observed frequency, which is the best a static layout can do.

Reads a route trace. Changes nothing.
"""
import sys
from collections import defaultdict, Counter

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
decode = [s for s in steps[1:] if len(next(iter(s.values()))) == 1]
layers = sorted(decode[0])
TOPK, NEXP = 16, 896
print("trace: %d decode tokens, %d layers, top-%d of %d" % (
    len(decode), len(layers), TOPK, NEXP))

# observed frequency per (layer, expert), for the balanced layout
freq = defaultdict(Counter)
for s in decode:
    for L in layers:
        for e in s[L][0]:
            freq[L][e] += 1

def layout(scheme, N, L):
    if scheme == "round":
        return lambda e: e % N
    if scheme == "block":
        return lambda e: min(e * N // NEXP, N - 1)
    load = [0] * N
    where = {}
    for e, c in sorted(freq[L].items(), key=lambda kv: -kv[1]):   # LPT
        p = min(range(N), key=lambda i: load[i])
        where[e] = p; load[p] += c
    nxt = 0
    for e in range(NEXP):
        if e not in where:
            where[e] = nxt % N; nxt += 1
    return lambda e: where[e]

print()
print("  %-7s %5s %9s %9s %9s %9s %9s" % (
    "scheme", "pods", "mean max", "p95 max", "speedup", "ideal", "efficiency"))
for scheme in ("round", "block", "balanced"):
    for N in (2, 4, 8, 14, 16, 28, 56):
        worst = []
        for L in layers:
            pod = layout(scheme, N, L)
            for s in decode:
                c = Counter(pod(e) for e in s[L][0])
                worst.append(max(c.values()))
        mean = sum(worst) / len(worst)
        p95 = sorted(worst)[int(0.95 * len(worst)) - 1]
        speed = TOPK / mean
        ideal = min(N, TOPK)
        print("  %-7s %5d %9.2f %9d %9.2f %9d %9.0f%%" % (
            scheme, N, mean, p95, speed, ideal, 100 * speed / ideal))
    print()

# Concurrent requests give a layer 16B experts to place instead of 16, so the
# same pods should balance better. Unlike the per-layer cache, which batching
# defeats, this should compose.
print("concurrent requests, round-robin placement")
print("  %-6s %7s %10s %10s %9s %9s" % (
    "pods", "batch", "distinct", "mean max", "speedup", "efficiency"))
for N in (14, 28):
    for B in (1, 2, 4, 8):
        worst, total = [], []
        for L in layers:
            pod = layout("round", N, L)
            for start in range(0, len(decode) - B + 1, B):
                union = set()
                for s in decode[start:start + B]:
                    union |= set(s[L][0])
                c = Counter(pod(e) for e in union)
                worst.append(max(c.values()))
                total.append(len(union))
        mean = sum(worst) / len(worst)
        dist = sum(total) / len(total)
        speed = dist / mean
        ideal = min(N, dist)
        print("  %-6d %7d %10.1f %10.2f %9.2f %9.0f%%" % (
            N, B, dist, mean, speed, 100 * speed / ideal))
    print()
