#!/usr/bin/env python3
"""Do consecutive DECODE tokens select the same experts?

WHY THIS QUESTION AND NOT THE ONE ALREADY ANSWERED

THE-FLOW-OBSERVED.md measured expert overlap across the five PREFILL positions of one
prompt and found it negligible: mean pairwise overlap 2.3 of 16, zero experts common to
all five positions at 91 of 92 layers, 50-74 distinct experts touched per layer. I read
that as evidence against reuse generally. That was the wrong generalisation. Prefill
positions are five different words of a sentence evaluated together; consecutive decode
tokens are successive states of one continuing generation, and nobody has measured them.

It matters because of where the cost is. A token costs 16 experts x 92 layers x 17.55 MB
= 25.8 GB, and today's runs read 18.6-27.8 GB per token with no sign of reuse. The
residual, snapshots and folds are ~29 KB vectors -- six orders of magnitude smaller. If
there is a per-prompt expert working set, it is the only observation that could move the
number. If consecutive decode tokens overlap like prefill positions do, the idea is dead
and that is worth knowing cheaply.

SELF-CHECK BUILT IN
The same dump contains the prefill positions, whose overlap is already established at
~2.3 of 16. This script reports that separately. If it does not reproduce the known
figure, the parsing is wrong and the decode numbers must not be believed.

Input: the K3_DUMPSEL file, one line per (layer, position):
    layer <TAB> position <TAB> token_id <TAB> e1 <TAB> ... <TAB> e16
"""
import collections
import sys


def main(path, prompt_len):
    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        try:
            rows.append((int(f[0]), int(f[1]), int(f[2]), tuple(int(x) for x in f[3:])))
        except ValueError:
            continue
    if not rows:
        print("no rows parsed -- wrong file or format")
        return

    widths = collections.Counter(len(r[3]) for r in rows)
    print("rows %d   layers %d   positions %d   experts/row %s"
          % (len(rows), len({r[0] for r in rows}), len({r[1] for r in rows}),
             dict(widths)))

    # A position may be re-evaluated; keep the LAST selection seen for each (layer, pos).
    sel = {}
    seen_order = collections.defaultdict(list)
    for layer, pos, tok, experts in rows:
        if pos not in seen_order[layer]:
            seen_order[layer].append(pos)
        sel[(layer, pos)] = (tok, experts)
    repeats = len(rows) - len(sel)
    if repeats:
        print("note: %d rows were repeats of a (layer,position) already seen -- the engine"
              % repeats)
        print("      re-evaluates positions, so only the last selection per pair is used")

    def overlaps(layer, positions):
        out = []
        for a, b in zip(positions, positions[1:]):
            if (layer, a) in sel and (layer, b) in sel:
                out.append(len(set(sel[(layer, a)][1]) & set(sel[(layer, b)][1])))
        return out

    layers = sorted({r[0] for r in rows})
    allpos = sorted({r[1] for r in rows})
    prefill = [p for p in allpos if p < prompt_len]
    decode = [p for p in allpos if p >= prompt_len - 1]   # include the join

    def report(name, positions, k):
        vals = []
        for layer in layers:
            vals += overlaps(layer, positions)
        if not vals:
            print("  %-34s no pairs" % name)
            return
        vals.sort()
        n = len(vals)
        print("  %-34s n=%5d  mean %5.2f of %d  median %2d  min %2d  max %2d"
              % (name, n, sum(vals) / n, k, vals[n // 2], vals[0], vals[-1]))

    width = widths.most_common(1)[0][0]
    print()
    print("expert overlap between CONSECUTIVE positions, per layer:")
    report("prefill positions (known ~2.3/16)", prefill, width)
    report("decode positions", decode, width)

    # A per-prompt working set only helps if it is bounded. Count distinct experts a
    # layer ever uses across the whole generation, against the 896 it could.
    print()
    print("distinct experts used per layer across the whole generation:")
    counts = []
    for layer in layers:
        used = set()
        for p in allpos:
            if (layer, p) in sel:
                used |= set(sel[(layer, p)][1])
        counts.append(len(used))
    counts.sort()
    n = len(counts)
    print("  over %d positions: median %d of 896   min %d   max %d"
          % (len(allpos), counts[n // 2], counts[0], counts[-1]))
    print("  a layer reading its whole used set once would cost %.1f GB against"
          % (counts[n // 2] * 92 * 17547264 / float(1 << 30)))
    print("  %.1f GB for reading 16 fresh experts at every one of %d positions"
          % (width * 92 * len(allpos) * 17547264 / float(1 << 30), len(allpos)))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
