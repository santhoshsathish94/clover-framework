#!/usr/bin/env python3
"""Does a wider prefill batch amortise expert reads, or only the trunk?

I claimed expert bytes scale with positions, so widening the batch past 8 would
only help the trunk - about 16% of a prefill pass. That was an assumption, and
it contradicts data already measured: adjacent positions share 5.879 of their
16 experts. If positions overlap, W positions touch far fewer than 16*W
distinct experts, and the reads amortise exactly as the trunk read does.

This counts, per layer, the distinct experts a window of W consecutive
positions actually selects, from the ten captured routing files. Expert bytes a
token is then distinct/W * 17.5 MB, and the ratio against W=1 is what a wider
batch would save.

usage: batch-width-experts.py [dir]
"""
import glob
import json
import os
import sys

EXPERT_MB = 17547264 / 1e6


def load(path):
    sel = {}
    for line in open(path):
        if line.startswith("ROUTE "):
            r = json.loads(line[6:])
            sel.setdefault(r["layer"], {})[r["position"]] = r["experts"]
    return sel


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep"
    files = sorted(glob.glob(os.path.join(root, "route-*.jsonl")))
    if not files:
        print("no routing files in %s" % root)
        return 1

    widths = [1, 2, 4, 8, 16, 32, 64]
    totals = {w: [0, 0] for w in widths}   # distinct sum, windows counted

    for path in files:
        sel = load(path)
        for layer, byposition in sel.items():
            positions = sorted(byposition)
            for w in widths:
                for start in range(0, len(positions) - w + 1, w):
                    window = positions[start:start + w]
                    distinct = set()
                    for p in window:
                        distinct.update(byposition[p])
                    totals[w][0] += len(distinct)
                    totals[w][1] += 1

    print("%d routing files, %d layers" % (len(files), len(load(files[0]))))
    print()
    print("%-7s %12s %14s %14s %12s"
          % ("batch W", "distinct/win", "distinct/token", "expert MB/token", "vs W=1"))
    base = None
    for w in widths:
        s, n = totals[w]
        if not n:
            continue
        per_win = s / float(n)
        per_token = per_win / w
        mb = per_token * EXPERT_MB
        if base is None:
            base = mb
        print("%-7d %12.1f %14.2f %14.1f %11.2fx"
              % (w, per_win, per_token, mb, base / mb))

    print()
    print("If distinct/token falls with W, a wider batch amortises expert reads")
    print("as well as the trunk, and the 16%% figure I gave was wrong.")
    print("Ceiling is 16*W if positions shared nothing at all.")


if __name__ == "__main__":
    sys.exit(main())
