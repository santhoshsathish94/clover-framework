#!/usr/bin/env python3
"""Pull one CLOVER_VECTORS record out to a raw float32 file.

Used to cache pod 12's attention output so it can be substituted for the computed one.
"""
import struct
import sys

if len(sys.argv) != 5:
    sys.exit("usage: extract-record.py <dump.bin> <local> <position> <out.f32>")

path, local, position, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

with open(path, "rb") as handle:
    while True:
        header = handle.read(16)
        if len(header) < 16:
            sys.exit("record not found")
        owner, loc, pos, count = struct.unpack("<4i", header)
        if count < 0 or count > 65536:
            sys.exit("implausible count %d" % count)
        payload = handle.read(count * 4)
        if len(payload) < count * 4:
            sys.exit("truncated")
        if loc == local and pos == position:
            with open(out, "wb") as sink:
                sink.write(payload)
            print("wrote %s: owner %d local %d position %d count %d" % (out, owner, loc, pos, count))
            break
