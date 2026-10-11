#!/usr/bin/env python3
"""Does a donor capture actually contain the snapshot slots the injection needs?

If slots 2 and above are absent, CLOVER_FREEZE_FROM=2 silently injects nothing and
the run looks like a null result when in fact nothing was tested.
"""
import collections
import glob
import struct
import sys


def census(path):
    slots = collections.Counter()
    records = 0
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            owner, local, position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 20:
                break
            handle.seek(4 * count, 1)
            records += 1
            if 1000 <= local < 1008:
                slots[local - 1000] += 1
    return records, slots


def main():
    for prefix in sys.argv[1:]:
        files = sorted(glob.glob(f"{prefix}.*.bin"))
        print(f"=== {prefix} : {len(files)} files ===")
        have_two_plus = 0
        for path in files:
            owner = int(path.split(".")[-2])
            records, slots = census(path)
            if any(s >= 2 for s in slots):
                have_two_plus += 1
            if owner in (2, 24, 25, 46, 60, 92):
                print(f"  layer {owner:3d}: {records:4d} records, slots present {sorted(slots)}")
        print(f"  layers holding a slot 2 or above: {have_two_plus} of {len(files)}")
        print()


if __name__ == "__main__":
    main()
