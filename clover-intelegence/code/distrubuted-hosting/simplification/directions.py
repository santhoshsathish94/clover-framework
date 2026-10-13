#!/usr/bin/env python3
"""How many directions does the chain actually move in?

The residual turns 20-40 degrees at every step for eighty steps yet stays at a right
angle to what entered the chain. That is what confined motion looks like, so this
measures the confinement instead of asserting it.

Takes the 93 states the carried vector passes through at one position, stacks them, and
reports the singular value spectrum. Two readings, because they answer different
questions:

  raw     the states as they are. Norms span 0.46 to 133, so this is dominated by the
          late large ones and says where the energy is.
  unit    every state scaled to length 1. This says how many directions are visited at
          all, regardless of how far along them the chain had got.

A reference row of the same shape filled with random normal values is printed beside it,
because a spectrum means nothing without knowing what no structure looks like.

The eight fold sources are measured the same way, and then projected onto the residual
states to see whether they lie inside the space the chain moves in or outside it.

usage: directions.py DUMP-PREFIX POSITION
"""
import glob
import os
import re
import struct
import sys

import numpy

MLA_PODS = set(range(3, 92, 4)) | {92}
WANTED = {0, 119} | set(range(1000, 1008))


def read(path, position, ahead):
    out = {}
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return out
            _, local, at, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return out
            want = ahead if 1000 <= local <= 1007 else position
            if local not in WANTED or at != want:
                handle.seek(4 * count, 1)
                continue
            body = handle.read(4 * count)
            if len(body) < 4 * count:
                return out
            out[local] = numpy.frombuffer(body, dtype="<f4").astype("f8")


def spectrum(matrix, label):
    values = numpy.linalg.svd(matrix, compute_uv=False)
    energy = numpy.cumsum(values ** 2) / float((values ** 2).sum())
    reach = [int(numpy.searchsorted(energy, f) + 1) for f in (0.50, 0.90, 0.99, 0.999)]
    print("%-22s %4d x %d   50%% in %d   90%% in %d   99%% in %d   99.9%% in %d"
          % (label, matrix.shape[0], matrix.shape[1], *reach))
    return values, energy


def main():
    prefix, position = sys.argv[1], int(sys.argv[2])
    states, sources = [], None
    for path in sorted(glob.glob(prefix + ".*.bin"),
                       key=lambda p: int(re.search(r"(\d+)\.bin$", p).group(1))):
        pod = int(re.search(r"(\d+)\.bin$", os.path.basename(path)).group(1))
        if not 1 <= pod <= 92:
            continue
        frame = read(path, position, position + 1)
        if 0 in frame:
            states.append(frame[0])
        if pod == 92:
            if 119 in frame:
                states.append(frame[119])
            sources = numpy.array([frame[k] for k in range(1000, 1008) if k in frame])

    states = numpy.array(states)
    unit = states / numpy.linalg.norm(states, axis=1, keepdims=True)
    noise = numpy.random.default_rng(0).standard_normal(states.shape)
    noise /= numpy.linalg.norm(noise, axis=1, keepdims=True)

    print("position %d\n" % position)
    values, energy = spectrum(states, "residual states, raw")
    spectrum(unit, "residual states, unit")
    spectrum(noise, "random, unit (control)")
    print()
    spectrum(sources, "fold sources, raw")
    spectrum(sources / numpy.linalg.norm(sources, axis=1, keepdims=True),
             "fold sources, unit")

    print("\nsingular values of the unit residual states, largest 12:")
    unit_values = numpy.linalg.svd(unit, compute_uv=False)
    print("  " + "  ".join("%.3f" % v for v in unit_values[:12]))
    print("the same for the random control:")
    print("  " + "  ".join("%.3f" % v for v in numpy.linalg.svd(noise, compute_uv=False)[:12]))

    basis = numpy.linalg.svd(unit, full_matrices=False)[2]
    print("\nhow much of each fold source lies in the span of the residual states:")
    for index, source in enumerate(sources):
        inside = float(numpy.linalg.norm(basis @ source)) / float(numpy.linalg.norm(source))
        print("  slot %d  size %10.4f   %.4f of it" % (index, numpy.linalg.norm(source), inside))


if __name__ == "__main__":
    main()
