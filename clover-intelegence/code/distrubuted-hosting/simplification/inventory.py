#!/usr/bin/env python3
"""Inventory a vectors capture: which owners, which locals, how many records, what width.

Written before looking at any value. The point is to find out what the engine recorded,
not to confirm what I think it records. Every local that appears is listed, including ones
no patch of mine put there.

usage: inventory.py DUMP-PREFIX
"""
import collections
import glob
import struct
import sys


def scan(path):
    seen = collections.Counter()
    widths = collections.defaultdict(set)
    positions = collections.defaultdict(set)
    order = []
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            owner, local, position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                break
            if handle.seek(4 * count, 1) is None:
                break
            seen[local] += 1
            widths[local].add(count)
            positions[local].add(position)
            if local not in order:
                order.append(local)
    return seen, widths, positions, order


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 1
    files = sorted(glob.glob(sys.argv[1] + ".*.bin"),
                   key=lambda p: int(p.rsplit(".", 2)[-2]))
    print("%d owner files\n" % len(files))

    shapes = collections.defaultdict(list)
    for path in files:
        owner = int(path.rsplit(".", 2)[-2])
        seen, widths, positions, order = scan(path)
        key = tuple(sorted(seen.items()))
        shapes[key].append(owner)
        if owner in (0, 1, 2, 3, 93):
            print("owner %d, first-appearance order: %s" % (owner, order))
            for local in sorted(seen):
                print("   local %-5d %5d records  width %-8s positions %s"
                      % (local, seen[local],
                         ",".join(str(w) for w in sorted(widths[local])),
                         ",".join(str(p) for p in sorted(positions[local]))))
            print()

    print("=== distinct record shapes across owners ===")
    for key, owners in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        total = sum(c for _, c in key)
        print("\n%d owners (%s%s): %d records each"
              % (len(owners), ",".join(str(o) for o in owners[:12]),
                 " ..." if len(owners) > 12 else "", total))
        print("   " + "  ".join("%d x%d" % (local, count) for local, count in key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
