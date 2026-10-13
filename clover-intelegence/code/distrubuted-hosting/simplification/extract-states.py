#!/usr/bin/env python3
"""Pull the carried vector's states out of a dump and keep only those.

A dump is 1.6 GB; the 93 states per position are 13 MB at five tokens. The saturation
sweep needs many prompts and the disk holds one dump at a time, so this writes the
states out and the dump is deleted behind it.

Rows come out ordered by position then by pod, which is the order the chain runs in.

usage: extract-states.py DUMP-PREFIX OUT.npy
"""
import glob
import os
import re
import struct
import sys

import numpy


def main():
    prefix, out = sys.argv[1], sys.argv[2]
    found = {}
    for path in sorted(glob.glob(prefix + ".*.bin"),
                       key=lambda p: int(re.search(r"(\d+)\.bin$", p).group(1))):
        pod = int(re.search(r"(\d+)\.bin$", os.path.basename(path)).group(1))
        if not 1 <= pod <= 92:
            continue
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                _, local, at, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 22:
                    break
                if not (local == 0 or (local == 119 and pod == 92)):
                    handle.seek(4 * count, 1)
                    continue
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    break
                found.setdefault(at, []).append(numpy.frombuffer(body, dtype="<f4"))

    states = numpy.array([v for at in sorted(found) for v in found[at]], dtype="<f4")
    numpy.save(out, states)
    sys.stderr.write("%s: %d positions, %d states, %d wide\n"
                     % (out, len(found), states.shape[0], states.shape[1]))


if __name__ == "__main__":
    main()
