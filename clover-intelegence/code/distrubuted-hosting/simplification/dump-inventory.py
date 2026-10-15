#!/usr/bin/env python3
"""What is actually inside a vector dump: locals, positions, widths.

Written because answer-trajectory.py found zero records for (position 4, local 119) in
/tmp/vec-all.bin.*.bin and the honest response to that is to look rather than to guess
which of the three assumptions -- header layout, position numbering, or local id -- was
wrong.

usage: dump-inventory.py FILE [FILE ...]
"""
import struct
import sys
from collections import Counter


def main():
    for path in sys.argv[1:]:
        locals_seen, positions, widths = Counter(), Counter(), Counter()
        total = 0
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                a, local, pos, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 22:
                    print("%s: header stops making sense at record %d: %r"
                          % (path, total, (a, local, pos, count)))
                    break
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    break
                total += 1
                locals_seen[local] += 1
                positions[pos] += 1
                widths[count] += 1
                if total == 1:
                    print("%s: first header fields = (%d, local=%d, pos=%d, count=%d)"
                          % (path, a, local, pos, count))
        print("  records %d" % total)
        print("  positions : %s" % sorted(positions))
        print("  widths    : %s" % sorted(widths))
        ls = sorted(locals_seen)
        print("  locals    : %d distinct, %s%s"
              % (len(ls), ls[:24], " ..." if len(ls) > 24 else ""))
        print("  has 116/118/119/120: %s"
              % {k: locals_seen.get(k, 0) for k in (116, 118, 119, 120)})


if __name__ == "__main__":
    main()
