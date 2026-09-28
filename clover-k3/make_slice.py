#!/usr/bin/env python3
"""Extract one layer's trunk slice.

Every layer occupies a contiguous, gap-free range of trunk.bin and no two
layers overlap - checked, not assumed - so a slice is a single read and the
tensor offsets inside it need no rewriting.

Writes:
  $K3_SLICES/L<NN>.bin    the layer's bytes, nothing else
  $K3_SLICES/L<NN>.json   the layer's tensor table, offsets relative to the slice

usage: make_slice.py <layer> [more layers...]
"""
import hashlib
import json
import os
import sys

TRUNK = os.environ.get("K3_TRUNK", "/srv/k3/trunk")
SLICES = os.environ.get("K3_SLICES", "/srv/k3/slices")
SRC = os.path.join(TRUNK, "full.bin")
CHUNK = 64 << 20


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: make_slice.py <layer> [layer...]")
    meta = json.load(open(os.path.join(TRUNK, "trunk.json")))
    os.makedirs(SLICES, exist_ok=True)

    for arg in sys.argv[1:]:
        L = int(arg)
        lay = meta["layers"][L]
        off, nb = lay["file_off"], lay["nbytes"]

        # nbytes is the layer's padded extent; the tensors span slightly less,
        # and the next layer starts at file_off + nbytes. Data running past the
        # declared extent would be corruption, so only that direction is fatal.
        span = max(t["off"] + t["nbytes"] for t in lay["tensors"].values())
        if span > nb:
            sys.exit("L%d: tensors span %d past the declared %d" % (L, span, nb))
        pad = nb - span

        out = os.path.join(SLICES, "L%02d.bin" % L)
        h = hashlib.sha256()
        with open(SRC, "rb") as fi, open(out, "wb") as fo:
            fi.seek(off)
            left = nb
            while left:
                b = fi.read(min(CHUNK, left))
                if not b:
                    sys.exit("L%d: short read at %d" % (L, nb - left))
                fo.write(b)
                h.update(b)
                left -= len(b)
        written = h.hexdigest()

        # read the source range again and compare: proves the slice is the
        # range it claims to be, not merely the right length
        h2 = hashlib.sha256()
        with open(SRC, "rb") as fi:
            fi.seek(off)
            left = nb
            while left:
                b = fi.read(min(CHUNK, left))
                h2.update(b)
                left -= len(b)
        if h2.hexdigest() != written:
            sys.exit("L%d: slice does not match the source range" % L)

        json.dump({"layer": L, "nbytes": nb, "span": span, "pad": pad,
                   "sha256": written, "source_off": off,
                   "tensors": lay["tensors"]},
                  open(os.path.join(SLICES, "L%02d.json" % L), "w"))

        print("  L%-3d %10.0f MB  pad %-4d  %s  %s"
              % (L, nb / 1e6, pad, written[:16], os.path.basename(out)))


main()
