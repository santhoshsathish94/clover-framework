#!/usr/bin/env python3
"""Does the space the residual moves in stop growing as prompts are added?

Two prompts overlapping proves nothing on its own: any two trajectories of the same
length overlap somewhat, and N states can only span N directions. What separates "the
model works in a fixed region" from "that is just what two samples look like" is whether
the region stops growing.

For each prompt in turn:

  inside before   how much of this prompt's energy already lies in the span of every
                  prompt before it. Climbing towards 1 is saturation; staying flat is not
  adds            directions the 90% energy ball gains by including this prompt
  running         where the 90% ball sits after including it

A random control of the same shapes runs beside it, because the same two numbers on
random data say what no shared structure looks like.

The first prompts are deliberately near-duplicates of each other, differing in one token
of five. If the measurement works they must add almost nothing while an unrelated prompt
adds a lot. That is the positive control and it is checked before anything else is read.

usage: saturate.py STATES-DIR NAME [NAME ...]
"""
import os
import sys

import numpy


def unit(matrix):
    return matrix / numpy.linalg.norm(matrix, axis=1, keepdims=True)


def orthonormal(matrix):
    """Rows spanning the row space, from the n x n Gram matrix.

    The direct route is an SVD of an n x 7168 matrix asking for the 7168-wide basis,
    which dominates the run. n is a few thousand and always smaller, so the Gram matrix
    gives the same thing for far less.
    """
    gram = matrix @ matrix.T
    values, vectors = numpy.linalg.eigh(gram)
    keep = values > values.max() * 1e-12
    values, vectors = values[keep], vectors[:, keep]
    return (vectors.T @ matrix) / numpy.sqrt(values)[:, None], values[::-1]


def reach(values, fraction=0.90):
    energy = numpy.cumsum(values) / float(values.sum())
    return int(numpy.searchsorted(energy, fraction) + 1)


def inside(target, basis):
    if basis is None:
        return float("nan")
    return float((((target @ basis.T) @ basis) ** 2).sum() / (target ** 2).sum())


def run(label, sets, names):
    print("\n%s" % label, flush=True)
    print("%-10s %8s %14s %8s %10s" % ("prompt", "states", "inside before", "adds", "running"),
          flush=True)
    seen, basis, before = None, None, 0
    for name, block in zip(names, sets):
        share = inside(block, basis)
        seen = block if seen is None else numpy.vstack([seen, block])
        basis, values = orthonormal(seen)
        now = reach(values)
        print("%-10s %8d %14s %8s %10d"
              % (name, len(block),
                 "-" if share != share else "%.4f" % share,
                 "-" if before == 0 else "+%d" % (now - before), now), flush=True)
        before = now


def main():
    directory, names = sys.argv[1], sys.argv[2:]
    sets, kept = [], []
    for name in names:
        path = os.path.join(directory, name + ".npy")
        if not os.path.exists(path):
            sys.stderr.write("missing %s, skipping\n" % path)
            continue
        sets.append(unit(numpy.load(path).astype("f8")))
        kept.append(name)

    run("measured", sets, kept)
    rng = numpy.random.default_rng(0)
    run("random, same shapes (control)",
        [unit(rng.standard_normal(block.shape)) for block in sets], kept)


if __name__ == "__main__":
    main()
