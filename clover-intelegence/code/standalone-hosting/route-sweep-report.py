#!/usr/bin/env python3
"""Routing observed across many prompts and 64 generated tokens each.

Repeats the single-prompt observation at scale so the numbers stop resting on one
prompt. Reads every route-*.jsonl written by route-sweep.sh. Each file seeds a prompt
and feeds every emitted token back, so within one file the context grows from the seed
length to seed+64, and across files the seed length spans 5 to 24.

Reports, counting GENERATED positions only unless stated:

  adjacent overlap     |E(p) & E(p+1)| out of k, per prompt and pooled
  chance               k*k/896, what two independent draws would give
  per layer            which layers, if any, route more stably than others
  versus position      whether overlap drifts as the context grows
  per-layer LRU        hit rate by cache capacity, which is the only form of this
                       that turns into bytes not read

Nothing is scored against any answer token.

usage: route-sweep-report.py DIR
"""
import glob
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

EXPERT_GB = 17547264 / float(1 << 30)


def load(path):
    seeded = None
    sel = {}
    for line in open(path):
        if line.startswith("RUN "):
            seeded = json.loads(line[4:])["seeded"]
        elif line.startswith("ROUTE "):
            r = json.loads(line[6:])
            sel[(r["position"], r["layer"])] = r["experts"]
    return seeded, sel


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep"
    files = sorted(glob.glob(os.path.join(root, "route-*.jsonl")))
    if not files:
        print("no route files in %s" % root)
        return

    pooled_adj = []
    per_layer = defaultdict(list)
    per_distance = defaultdict(list)
    all_sel = []
    k = None

    print("%-26s %5s %7s %9s %9s" % ("prompt file", "seed", "gen pos", "adj mean", "vs chance"))
    for path in files:
        seeded, sel = load(path)
        if not sel or seeded is None:
            print("%-26s  unreadable or empty" % os.path.basename(path))
            continue
        positions = sorted({p for p, _ in sel})
        layers = sorted({l for _, l in sel})
        k = len(next(iter(sel.values())))
        gen = [p for p in positions if p >= seeded]
        vals = []
        for p in gen:
            if (p + 1) not in positions:
                continue
            for l in layers:
                a, b = sel.get((p, l)), sel.get((p + 1, l))
                if a and b:
                    n = len(set(a) & set(b))
                    vals.append(n)
                    per_layer[l].append(n)
        chance = k * k / 896.0
        if vals:
            m = sum(vals) / float(len(vals))
            pooled_adj.extend(vals)
            print("%-26s %5d %7d %9.3f %8.1fx"
                  % (os.path.basename(path), seeded, len(gen), m, m / chance))
        all_sel.append((seeded, sel, positions, layers))

        # Overlap at increasing distance, generated only. Odd distances are included
        # one by one because the first pass showed distance 2 scoring HIGHER than
        # distance 1, which is the opposite of a decay and suggests a parity effect
        # rather than a trend. Only every distance can tell them apart.
        for d in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32):
            for p in gen:
                if (p + d) in positions:
                    for l in layers:
                        a, b = sel.get((p, l)), sel.get((p + d, l))
                        if a and b:
                            per_distance[d].append(len(set(a) & set(b)))

    if not pooled_adj:
        print("nothing to pool")
        return
    chance = k * k / 896.0
    print()
    print("pooled over %d prompts: adjacent overlap %.3f of %d, chance %.3f, %.1fx chance"
          % (len(files), sum(pooled_adj) / float(len(pooled_adj)), k, chance,
             (sum(pooled_adj) / float(len(pooled_adj))) / chance))

    print()
    print("overlap by distance between generated positions")
    for d in sorted(per_distance):
        v = per_distance[d]
        print("  %3d apart   %9.3f   (%d pairs)" % (d, sum(v) / float(len(v)), len(v)))

    print()
    ranked = sorted(((sum(v) / float(len(v)), l, len(v)) for l, v in per_layer.items()),
                    reverse=True)
    print("layers with the MOST stable routing")
    for m, l, n in ranked[:8]:
        print("  layer %2d   %6.3f of %d" % (l, m, k))
    print("layers with the LEAST")
    for m, l, n in ranked[-5:]:
        print("  layer %2d   %6.3f of %d" % (l, m, k))

    print()
    print("per-layer LRU over generated positions, pooled across prompts")
    print("  %8s %10s %12s %12s" % ("capacity", "hit rate", "GB/token", "cache RAM"))
    for cap in (0, 16, 24, 32, 48, 64, 96, 128):
        hits = misses = tokens = 0
        for seeded, sel, positions, layers in all_sel:
            caches = {l: OrderedDict() for l in layers}
            for p in positions:
                if p < seeded:
                    continue
                tokens += 1
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
            tokens -= 0
        total = hits + misses
        per_tok = misses * EXPERT_GB / max(1, tokens / float(len(layers)) * len(layers) / len(layers))
        n_tok = sum(len([p for p in ps if p >= sd]) for sd, _, ps, _ in all_sel)
        print("  %8d %9.1f%% %12.2f %9.1f GB"
              % (cap, 100.0 * hits / total if total else 0,
                 misses * EXPERT_GB / max(1, n_tok),
                 cap * 92 * EXPERT_GB))


if __name__ == "__main__":
    main()
