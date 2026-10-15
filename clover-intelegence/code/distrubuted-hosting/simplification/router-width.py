#!/usr/bin/env python3
"""How many experts does a pod actually route to? Read it, do not assume it.

The distributed capture was taken before checking k. Earlier in this work I inferred
"8 experts per layer" for this side from 13 GB/token of measured disk reads, which was
wrong: measured disk reads are demand minus whatever the page cache already held, so
they are a floor on demand, not a count of it. The router's own records settle it --
local 3000 is the weight per chosen expert and local 5103 is the chosen ids, so their
width IS k.

usage: router-width.py FILE [FILE ...]
"""
import struct
import sys
from collections import Counter

WANT = {3000: "router weights on the chosen", 5103: "chosen expert ids",
        2003: "fold 37 scores", 2021: "fold 38 scores"}


def main():
    for path in sys.argv[1:]:
        widths = {k: Counter() for k in WANT}
        sums = {}
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                _, local, pos, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 22:
                    break
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    break
                if local in WANT:
                    widths[local][count] += 1
                    if local == 3000 and pos not in sums:
                        v = struct.unpack("<%df" % count, body)
                        sums[pos] = sum(v)
        print(path)
        for local, label in sorted(WANT.items()):
            if widths[local]:
                print("  local %-5d %-28s widths %s"
                      % (local, label, dict(widths[local])))
        if sums:
            print("  router weights sum to: %s"
                  % " ".join("%.4f" % sums[p] for p in sorted(sums)))
            print("  (a sum of 1.0000 means the width above is the full normalised set)")


if __name__ == "__main__":
    main()
