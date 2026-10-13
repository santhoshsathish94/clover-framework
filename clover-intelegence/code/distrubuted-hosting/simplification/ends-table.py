#!/usr/bin/env python3
"""Read a vectors dump and print one row per position for the locals asked for.

The walker prints one position at a time. Reading the ends of the chain means asking
whether something holds across every position, so this lays the positions out as rows
and the locals as columns. Norms only, plus the raw values for short records such as
fold scores, because a nine float record is the whole story and a 7168 float one is not.

usage: ends-table.py DUMP [local[:label] ...]
"""
import math
import struct
import sys


def records(path):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return
            owner, local, position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return
            yield owner, local, position, struct.unpack("<%df" % count, body)


def norm(values):
    return math.sqrt(sum(float(v) * float(v) for v in values))


def softmax(scores):
    top = max(scores)
    weights = [math.exp(float(s) - top) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip())
        return 1
    path = sys.argv[1]
    wanted = []
    for item in sys.argv[2:]:
        local, _, label = item.partition(":")
        wanted.append((int(local), label or local))

    table = {}
    owners = set()
    for owner, local, position, values in records(path):
        owners.add(owner)
        table.setdefault(position, {})[local] = values

    print("owner(s) %s   %d positions" % (sorted(owners), len(table)))
    header = "  pos |" + "".join(" %12s" % label for _, label in wanted)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for position in sorted(table):
        row = "  %3d |" % position
        for local, _ in wanted:
            values = table[position].get(local)
            if values is None:
                row += " %12s" % "-"
            elif len(values) <= 9:
                row += " %12s" % ("/".join("%.4f" % v for v in values))
            else:
                row += " %12.4f" % norm(values)
        print(row)

    # Short records are scores going into a softmax everywhere they appear in this
    # engine, so print what the fold actually weighted rather than leaving it implied.
    for local, label in wanted:
        sample = next((table[p][local] for p in sorted(table) if local in table[p]), None)
        if sample is None or len(sample) > 9:
            continue
        print("\n  %s as softmax weights" % label)
        for position in sorted(table):
            values = table[position].get(local)
            if values is None:
                continue
            print("    pos %3d  %s" % (position,
                "  ".join("%.6f" % w for w in softmax(values))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
