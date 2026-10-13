#!/usr/bin/env python3
"""Pull a chosen record out of a dump and keep only that.

A dump is 1.6 GB; one record per pod per position is 13 MB at five tokens. The sweeps
need many prompts and the disk holds one dump at a time, so this writes the records out
and the dump is deleted behind it.

Local 0 is the residual arriving at a pod, which is what the chain carries. Locals 4003
and 4021 are the two fold outputs, which after a normalize are what the attention and
the MoE actually read. The weights never see the residual, so which record is pulled
decides which question the sweep answers.

Rows come out ordered by position then by pod, which is the order the chain runs in.

usage: extract-states.py DUMP-PREFIX OUT.npy [LOCAL]
"""
import glob
import os
import re
import struct
import sys

import numpy


def main():
    prefix, out = sys.argv[1], sys.argv[2]
    local_wanted = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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
                keep = (local == local_wanted
                        or (local_wanted == 0 and local == 119 and pod == 92))
                if not keep:
                    handle.seek(4 * count, 1)
                    continue
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    break
                found.setdefault(at, []).append(numpy.frombuffer(body, dtype="<f4"))

    states = numpy.array([v for at in sorted(found) for v in found[at]], dtype="<f4")
    numpy.save(out, states)
    sys.stderr.write("%s: local %d, %d positions, %d rows, %d wide\n"
                     % (out, local_wanted, len(found), states.shape[0], states.shape[1]))


if __name__ == "__main__":
    main()
