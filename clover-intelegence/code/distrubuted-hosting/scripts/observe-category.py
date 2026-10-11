#!/usr/bin/env python3
"""Phases 1 and 2 of vector-transformation-observer.md, applied to one stage-120 block.

Observation only. Nothing here assumes the map is linear, affine, constant or
invertible; every such property is tested and the test result is reported either way.

Inputs:
  /tmp/vec46.bin   records of (owner,local,position,count) int32 then count float32
  /tmp/pairs46.bin six (input,output) float32 pairs, in probe order
"""
import struct
import sys

WIDTH = 7168
PROBE_NAMES = ["zero", "u", "v", "u+v", "2u", "w(held out)"]


def read_records(path):
    runs, current = [], None
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            owner, local, position, count = struct.unpack("<4i", header)
            payload = handle.read(4 * count)
            values = struct.unpack(f"<{count}f", payload)
            if local == 0:
                current = []
                runs.append(current)
            if current is not None:
                current.append((local, values))
    return runs


def read_pairs(path, n):
    pairs = []
    with open(path, "rb") as handle:
        for _ in range(n):
            a = struct.unpack(f"<{WIDTH}f", handle.read(4 * WIDTH))
            b = struct.unpack(f"<{WIDTH}f", handle.read(4 * WIDTH))
            pairs.append((a, b))
    return pairs


def norm(v):
    return sum(x * x for x in v) ** 0.5


def diff_stats(a, b):
    worst, total, changed = 0.0, 0.0, 0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d:
            changed += 1
        worst = max(worst, d)
        total += d * d
    return worst, total ** 0.5, changed


def main():
    runs = read_records("/tmp/vec46.bin")
    pairs = read_pairs("/tmp/pairs46.bin", 6)
    print(f"runs captured: {len(runs)}, records in first run: {len(runs[0])}")

    print()
    print("=== PHASE 1. the trajectory of one probe ===")
    run = runs[1]
    print(f"probe 'u': {len(run)} recorded vectors, stages {run[0][0]} .. {run[-1][0]}")
    previous = run[0][1]
    moved = []
    for local, values in run[1:]:
        worst, euclid, changed = diff_stats(previous, values)
        if changed:
            moved.append((local, changed, worst, euclid))
        previous = values
    print(f"stages that altered the vector: {len(moved)} of {len(run) - 1}")
    print("  stage  coords changed   max |delta|     ||delta||")
    for local, changed, worst, euclid in moved:
        print(f"  {local:5d}  {changed:13d}  {worst:12.6g}  {euclid:12.6g}")

    print()
    print("=== PHASE 1. is the set of moving stages the same for every probe? ===")
    signatures = {}
    for index, run in enumerate(runs):
        previous = run[0][1]
        stages = []
        for local, values in run[1:]:
            if any(x != y for x, y in zip(previous, values)):
                stages.append(local)
            previous = values
        signatures.setdefault(tuple(stages), []).append(PROBE_NAMES[index])
    for stages, names in signatures.items():
        print(f"  {list(stages)}  <- {names}")

    print()
    print("=== PHASE 1. norms along the trajectory ===")
    for index, run in enumerate(runs):
        marks = [(local, norm(values)) for local, values in run
                 if local in (0, 1, 2, 9, 20, 116, 120)]
        text = "  ".join(f"{l}:{n:.4g}" for l, n in marks)
        print(f"  {PROBE_NAMES[index]:12s} {text}")

    print()
    print("=== PHASE 2. transformation class ===")
    zero_in, zero_out = pairs[0]
    u_in, u_out = pairs[1]
    v_in, v_out = pairs[2]
    uv_in, uv_out = pairs[3]
    two_u_in, two_u_out = pairs[4]

    print(f"f(0) norm = {norm(zero_out):.6g}   (a zero here would mean no constant offset)")

    # additivity about the offset: f(u+v) - f(0) =? (f(u)-f(0)) + (f(v)-f(0))
    predicted = [u_out[i] + v_out[i] - zero_out[i] for i in range(WIDTH)]
    worst, euclid, changed = diff_stats(predicted, uv_out)
    scale = norm(uv_out)
    print(f"additivity  f(u+v) vs f(u)+f(v)-f(0):  max abs {worst:.6g}  "
          f"||err|| {euclid:.6g}  relative {euclid/scale if scale else 0:.6g}  differing coords {changed}/{WIDTH}")

    # homogeneity about the offset: f(2u) - f(0) =? 2(f(u) - f(0))
    predicted = [2.0 * (u_out[i] - zero_out[i]) + zero_out[i] for i in range(WIDTH)]
    worst, euclid, changed = diff_stats(predicted, two_u_out)
    scale = norm(two_u_out)
    print(f"homogeneity f(2u) vs 2(f(u)-f(0))+f(0): max abs {worst:.6g}  "
          f"||err|| {euclid:.6g}  relative {euclid/scale if scale else 0:.6g}  differing coords {changed}/{WIDTH}")

    print()
    print("=== PHASE 2. how much of the output is simply the input carried through? ===")
    for index in (1, 2, 5):
        a, b = pairs[index]
        worst, euclid, changed = diff_stats(a, b)
        print(f"  {PROBE_NAMES[index]:12s} ||in|| {norm(a):.5g}  ||out|| {norm(b):.5g}  "
              f"||out-in|| {euclid:.5g}  relative change {euclid/norm(a) if norm(a) else 0:.5g}")


if __name__ == "__main__":
    main()
