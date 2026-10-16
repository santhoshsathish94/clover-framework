#!/usr/bin/env python3
"""Does allocating cache capacity per layer beat giving every layer the same amount?

THE OBSERVATION THAT PROMPTS THIS
Across 10 prompts and 64 generated tokens each, adjacent-token expert overlap varies by
5x between layers: layer 90 repeats 9.300 of its 16 experts, layer 6 only 1.863. The
engine gives every layer the identical K3_EXPERT_CACHE slots regardless. A layer whose
experts barely repeat is holding 24 slots that mostly miss, while a layer whose experts
repeat heavily could use more.

THE METHOD
Hit rate as a function of capacity is measured per layer from the real routing trace,
then the fixed slot budget is allocated greedily: repeatedly give the next slot to
whichever layer gains the most hits from it. Because each layer's hit curve is concave
-- the first slots capture the most reuse -- greedy on marginal gain is the optimal
allocation, not an approximation.

The budget is held at exactly 92 x 24 = 2208 slots, the same 36.1 GB the engine uses
today, so any improvement is free in RAM terms.

A caveat that decides whether this is worth building: the allocation is derived from
these ten prompts. If the per-layer ordering is a property of the model it will hold on
new prompts; if it is a property of this prompt set it will not. That is testable by
fitting on half the prompts and scoring on the other half, which is done below.

usage: cache-allocation.py DIR
"""
import glob
import json
import os
import sys
from collections import OrderedDict

EXPERT_GB = 17547264 / float(1 << 30)
MAXCAP = 96


def load(path):
    seeded, sel = None, {}
    for line in open(path):
        if line.startswith("RUN "):
            seeded = json.loads(line[4:])["seeded"]
        elif line.startswith("ROUTE "):
            r = json.loads(line[6:])
            sel[(r["position"], r["layer"])] = r["experts"]
    return seeded, sel


def hits_for(seqs, cap):
    """hits and total for one layer's access sequence list, per-layer LRU at cap."""
    hits = total = 0
    for seq in seqs:
        c = OrderedDict()
        for experts in seq:
            fresh = []
            for e in experts:
                total += 1
                if cap and e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    fresh.append(e)
            for e in fresh:
                if cap:
                    c[e] = 1
                    if len(c) > cap:
                        c.popitem(last=False)
    return hits, total


def build(files):
    """per layer, a list of per-prompt access sequences over generated positions"""
    per_layer = {}
    for path in files:
        seeded, sel = load(path)
        if seeded is None:
            continue
        positions = sorted({p for p, _ in sel})
        for l in sorted({l for _, l in sel}):
            seq = [sel[(p, l)] for p in positions if p >= seeded and (p, l) in sel]
            if seq:
                per_layer.setdefault(l, []).append(seq)
    return per_layer


def curve(per_layer):
    return {l: [hits_for(seqs, c)[0] for c in range(MAXCAP + 1)]
            for l, seqs in per_layer.items()}


def allocate(curves, budget):
    """greedy on marginal gain; optimal because each curve is concave"""
    caps = {l: 0 for l in curves}
    # marginal gain of the next slot for each layer
    import heapq
    heap = []
    for l, cv in curves.items():
        if MAXCAP >= 1:
            heapq.heappush(heap, (-(cv[1] - cv[0]), l))
    left = budget
    while left > 0 and heap:
        gain, l = heapq.heappop(heap)
        if caps[l] >= MAXCAP:
            continue
        caps[l] += 1
        left -= 1
        if caps[l] < MAXCAP:
            cv = curves[l]
            heapq.heappush(heap, (-(cv[caps[l] + 1] - cv[caps[l]]), l))
    return caps


def score(per_layer, caps):
    hits = total = 0
    for l, seqs in per_layer.items():
        h, t = hits_for(seqs, caps.get(l, 0))
        hits += h
        total += t
    return hits, total


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep"
    files = sorted(glob.glob(os.path.join(root, "route-*.jsonl")))
    print("%d prompts" % len(files))

    per_layer = build(files)
    n_layers = len(per_layer)
    budget = n_layers * 24
    print("budget %d slots = %.1f GB, the same as uniform capacity 24 over %d layers"
          % (budget, budget * EXPERT_GB, n_layers))

    uni = {l: 24 for l in per_layer}
    h, t = score(per_layer, uni)
    print("uniform 24      hit %.2f%%   misses %d" % (100.0 * h / t, t - h))

    curves = curve(per_layer)
    caps = allocate(curves, budget)
    h2, t2 = score(per_layer, caps)
    print("allocated       hit %.2f%%   misses %d" % (100.0 * h2 / t2, t2 - h2))
    print("change          %+.2f points, %.1f%% fewer misses"
          % (100.0 * h2 / t2 - 100.0 * h / t, 100.0 * (1 - float(t2 - h2) / (t - h))))

    order = sorted(caps.items(), key=lambda kv: -kv[1])
    print("  most slots: %s" % ", ".join("L%d=%d" % kv for kv in order[:8]))
    print("  least:      %s" % ", ".join("L%d=%d" % kv for kv in order[-8:]))

    # does the allocation transfer, or is it fitted to these prompts?
    if len(files) >= 4:
        half = len(files) // 2
        fit = build(files[:half])
        test = build(files[half:])
        common = set(fit) & set(test)
        fit = {l: fit[l] for l in common}
        test = {l: test[l] for l in common}
        b = len(common) * 24
        caps_fit = allocate(curve(fit), b)
        hu, tu = score(test, {l: 24 for l in common})
        ha, ta = score(test, caps_fit)
        print()
        print("held out: allocation fitted on %d prompts, scored on the other %d"
              % (half, len(files) - half))
        print("  uniform 24 on held-out   %.2f%%" % (100.0 * hu / tu))
        print("  allocated  on held-out   %.2f%%" % (100.0 * ha / ta))
        print("  transfer                 %+.2f points" % (100.0 * ha / ta - 100.0 * hu / tu))


if __name__ == "__main__":
    main()
