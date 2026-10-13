#!/usr/bin/env python3
"""Do the positions of one prompt move in the same directions, or different ones?

Each position's carried vector passes through 93 states. If the five positions share
their directions, the five sets stacked together span about as many directions as one
set does. If they share nothing, the stack spans five times as many. The answer sits
between those two and the point is to find where.

Three readings:

  per position   the spectrum of one position's 93 states, so the stack has something
                 to be read against
  stacked        all 465 states together
  pairwise       how much of one position's energy lies inside another's span

A random control of the same shape is run through all three, because two subspaces of
93 directions drawn at random from 7168 barely meet, and without that number an overlap
cannot be read as large or small.

The five positions are not independent: the chain is causal, so position 4 was computed
with positions 0 to 3 in its attention. That is a fact about the thing, not a confound
to remove, but it is why this says nothing about two unrelated prompts.

usage: positions.py DUMP-PREFIX
"""
import glob
import os
import re
import struct
import sys

import numpy


def read_states(prefix, position):
    states = []
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
                keep = (local == 0 or (local == 119 and pod == 92)) and at == position
                if not keep:
                    handle.seek(4 * count, 1)
                    continue
                body = handle.read(4 * count)
                if len(body) < 4 * count:
                    break
                states.append(numpy.frombuffer(body, dtype="<f4").astype("f8"))
    return numpy.array(states)


def reach(matrix):
    values = numpy.linalg.svd(matrix, compute_uv=False)
    energy = numpy.cumsum(values ** 2) / float((values ** 2).sum())
    return [int(numpy.searchsorted(energy, f) + 1) for f in (0.50, 0.90, 0.99)]


def unit(matrix):
    return matrix / numpy.linalg.norm(matrix, axis=1, keepdims=True)


def inside(target, basis):
    """Fraction of target's energy that lies in the row space of basis."""
    projected = (target @ basis.T) @ basis
    return float((projected ** 2).sum() / (target ** 2).sum())


def main():
    prefix = sys.argv[1]
    by_position = {}
    for position in range(16):
        states = read_states(prefix, position)
        if len(states):
            by_position[position] = states
    positions = sorted(by_position)
    rows = len(by_position[positions[0]])

    rng = numpy.random.default_rng(0)
    control = {p: unit(rng.standard_normal((rows, by_position[p].shape[1])))
               for p in positions}

    print("%d positions, %d states each, width %d\n"
          % (len(positions), rows, by_position[positions[0]].shape[1]))

    print("%-28s %8s %8s %8s" % ("", "50%", "90%", "99%"))
    for position in positions:
        print("%-28s %8d %8d %8d"
              % ("position %d alone" % position, *reach(unit(by_position[position]))))
    stacked = numpy.vstack([unit(by_position[p]) for p in positions])
    print("%-28s %8d %8d %8d" % ("all %d stacked" % len(positions), *reach(stacked)))
    print("%-28s %8d %8d %8d"
          % ("random, stacked (control)", *reach(numpy.vstack([control[p] for p in positions]))))
    print()
    print("if the positions shared every direction the stack would read like one position;")
    print("if they shared none it would read like %d separate sets." % len(positions))

    bases = {p: numpy.linalg.svd(unit(by_position[p]), full_matrices=False)[2]
             for p in positions}
    noise_bases = {p: numpy.linalg.svd(control[p], full_matrices=False)[2] for p in positions}

    print("\nfraction of the row position's energy lying inside the column position's span")
    print("%-6s %s" % ("", "  ".join("%8d" % p for p in positions)))
    for a in positions:
        cells = "  ".join("%8.4f" % inside(by_position[a], bases[b]) for b in positions)
        print("%-6d %s" % (a, cells))
    print("%-6s %s" % ("random",
                       "  ".join("%8.4f" % inside(control[positions[0]], noise_bases[b])
                                 for b in positions)))


if __name__ == "__main__":
    main()
