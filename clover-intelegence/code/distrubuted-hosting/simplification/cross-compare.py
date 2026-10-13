#!/usr/bin/env python3
"""Compare one record in one dump against another record in another dump.

vector-compare takes the same local in both files, which is right for asking whether a
layer changed between two prompts and wrong for asking whether the thing arriving at
the tail is the thing the server was handed. Different owners use different locals for
the same vector, so the comparison has to name both.

usage: cross-compare.py A.bin LOCAL POS B.bin LOCAL POS
"""
import math
import struct
import sys


def find(path, local, position):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return None
            _, rec_local, rec_position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return None
            if rec_local == local and rec_position == position:
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    return None
                return struct.unpack("<%df" % count, body)
            handle.seek(4 * count, 1)


def main():
    if len(sys.argv) != 7:
        print(__doc__.strip())
        return 1
    a = find(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    b = find(sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
    if a is None or b is None:
        print("not found: a=%s b=%s" % (a is not None, b is not None))
        return 1
    if len(a) != len(b):
        print("different widths %d vs %d" % (len(a), len(b)))
        return 1
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    worst = max(abs(float(x) - float(y)) for x, y in zip(a, b))
    identical = all(x == y for x, y in zip(a, b))
    print("count %d" % len(a))
    print("norm a %.6f  norm b %.6f" % (na, nb))
    print("cos %.9f" % (dot / (na * nb) if na and nb else 0.0))
    print("max abs difference %.9g" % worst)
    print("bit identical: %s" % ("yes" if identical else "no"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
