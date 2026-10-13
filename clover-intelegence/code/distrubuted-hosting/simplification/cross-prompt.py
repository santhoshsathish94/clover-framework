#!/usr/bin/env python3
"""Do two different prompts move in the same directions?

Within one prompt the five positions share part of their directions. That only buys
anything if the directions are a property of the model rather than of the prompt, so
this asks the same question across two captures that share no tokens.

Three readings, each against a random control of the same shape:

  alone        each prompt's 465 states
  stacked      both prompts together. Shared directions make the stack read like one
               prompt; unshared ones make it read like the sum.
  pairwise     how much of one prompt's energy lies inside the other's span

The control matters more here than anywhere else: two subspaces of a few hundred
directions drawn at random from 7168 barely meet, so a small overlap can still be far
above chance and a large one can still be an artefact of dimension.

usage: cross-prompt.py PREFIX-A NAME-A PREFIX-B NAME-B
"""
import glob
import os
import re
import struct
import sys

import numpy


def states(prefix):
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
                found.setdefault(at, []).append(
                    numpy.frombuffer(body, dtype="<f4").astype("f8"))
    return numpy.array([v for at in sorted(found) for v in found[at]])


def reach(matrix):
    values = numpy.linalg.svd(matrix, compute_uv=False)
    energy = numpy.cumsum(values ** 2) / float((values ** 2).sum())
    return [int(numpy.searchsorted(energy, f) + 1) for f in (0.50, 0.90, 0.99)]


def unit(matrix):
    return matrix / numpy.linalg.norm(matrix, axis=1, keepdims=True)


def inside(target, basis, keep):
    kept = basis[:keep]
    projected = (target @ kept.T) @ kept
    return float((projected ** 2).sum() / (target ** 2).sum())


def main():
    pa, na, pb, nb = sys.argv[1:5]
    a, b = unit(states(pa)), unit(states(pb))
    rng = numpy.random.default_rng(0)
    ra = unit(rng.standard_normal(a.shape))
    rb = unit(rng.standard_normal(b.shape))

    print("%s %d states, %s %d states, width %d\n" % (na, len(a), nb, len(b), a.shape[1]))
    print("%-30s %8s %8s %8s" % ("", "50%", "90%", "99%"))
    for label, matrix in ((na, a), (nb, b),
                          ("%s + %s stacked" % (na, nb), numpy.vstack([a, b])),
                          ("random stacked (control)", numpy.vstack([ra, rb]))):
        print("%-30s %8d %8d %8d" % (label, *reach(matrix)))

    basis_a = numpy.linalg.svd(a, full_matrices=False)[2]
    basis_b = numpy.linalg.svd(b, full_matrices=False)[2]
    basis_ra = numpy.linalg.svd(ra, full_matrices=False)[2]

    print("\nfraction of one prompt's energy inside the other's leading directions")
    print("%-8s %10s %10s %10s %10s" % ("keep", "%s in %s" % (nb, na),
                                        "%s in %s" % (na, nb), "random", ""))
    for keep in (10, 33, 110, 303, len(basis_a)):
        print("%-8d %10.4f %10.4f %10.4f"
              % (keep, inside(b, basis_a, keep), inside(a, basis_b, keep),
                 inside(rb, basis_ra, keep)))


if __name__ == "__main__":
    main()
