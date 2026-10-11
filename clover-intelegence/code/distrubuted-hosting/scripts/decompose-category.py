#!/usr/bin/env python3
"""Phase 1 continued: what the four writing stages actually do to the vector.

Only stages 2, 9, 20 and 116 alter the residual. This tests, bit-exactly, whether the
block's output decomposes as a sum of the terms those stages deposit. Nothing is
assumed about the terms themselves; the arithmetic identity is either exact or it is
not, and the result is reported either way.
"""
import struct

WIDTH = 7168
NAMES = ["zero", "u", "v", "u+v", "2u", "w(held out)"]


def read_runs(path):
    runs, current = [], None
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            _owner, local, _position, count = struct.unpack("<4i", header)
            values = struct.unpack(f"<{count}f", handle.read(4 * count))
            if local == 0:
                current = {}
                runs.append(current)
            if current is not None:
                current[local] = values
    return runs


def norm(v):
    return sum(x * x for x in v) ** 0.5


def f32(x):
    """Round to single precision. The layer computes in float32; a float64 sum here
    would disagree with it by rounding alone and hide whether the identity holds."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def compare(a, b):
    worst, total, differing = 0.0, 0.0, 0
    for x, y in zip(a, b):
        d = x - y
        if d:
            differing += 1
        worst = max(worst, abs(d))
        total += d * d
    return worst, total ** 0.5, differing


def main():
    runs = read_runs("/tmp/vec46.bin")
    print("stage 0 holds the input; stage 1 holds the residual before anything was")
    print("written to it, which for a fresh sequence is zero.")
    print()

    print("=== norms of the four writing stages ===")
    print(f"{'probe':12s} {'||input||':>10s} {'r1 (pre)':>10s} {'r2':>10s} {'r9 attn':>10s} {'r20':>10s} {'r116 out':>10s}")
    for index, run in enumerate(runs):
        print(f"{NAMES[index]:12s} {norm(run[0]):10.5g} {norm(run[1]):10.5g} {norm(run[2]):10.5g} "
              f"{norm(run[9]):10.5g} {norm(run[20]):10.5g} {norm(run[116]):10.5g}")

    print()
    print("=== is r2 exactly the input? ===")
    for index, run in enumerate(runs):
        worst, euclid, differing = compare(run[2], run[0])
        print(f"  {NAMES[index]:12s} max abs {worst:.6g}  differing coords {differing}/{WIDTH}")

    print()
    print("=== is r20 exactly r2 + r9, computed in single precision? ===")
    for index, run in enumerate(runs):
        predicted = [f32(run[2][i] + run[9][i]) for i in range(WIDTH)]
        worst, euclid, differing = compare(predicted, run[20])
        print(f"  {NAMES[index]:12s} max abs {worst:.6g}  ||err|| {euclid:.6g}  differing coords {differing}/{WIDTH}")

    print()
    print("=== the term stage 116 adds, and its size next to the rest ===")
    print(f"{'probe':12s} {'||input||':>10s} {'||attn||':>10s} {'||moe||':>10s} {'||out||':>10s} {'moe/out':>8s}")
    for index, run in enumerate(runs):
        moe = [run[116][i] - run[20][i] for i in range(WIDTH)]
        out = norm(run[116])
        print(f"{NAMES[index]:12s} {norm(run[0]):10.5g} {norm(run[9]):10.5g} {norm(moe):10.5g} "
              f"{out:10.5g} {norm(moe)/out if out else 0:8.4f}")

    print()
    print("=== so is the whole block exactly input + attn + moe, in single precision? ===")
    for index, run in enumerate(runs):
        moe = [f32(run[116][i] - run[20][i]) for i in range(WIDTH)]
        predicted = [f32(f32(run[0][i] + run[9][i]) + moe[i]) for i in range(WIDTH)]
        worst, euclid, differing = compare(predicted, run[120])
        print(f"  {NAMES[index]:12s} max abs {worst:.6g}  ||err|| {euclid:.6g}  differing coords {differing}/{WIDTH}")

    print()
    print("=== does the output equal the residual at stage 116? ===")
    for index, run in enumerate(runs):
        worst, euclid, differing = compare(run[116], run[120])
        print(f"  {NAMES[index]:12s} max abs {worst:.6g}  differing coords {differing}/{WIDTH}")


if __name__ == "__main__":
    main()
