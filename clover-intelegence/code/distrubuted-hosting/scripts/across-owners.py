#!/usr/bin/env python3
"""The sequence the protocol should actually be applied to.

x0 is what the server handed to layer 1; x1..x92 are the outputs of each layer in
turn. One step per owner, not one step per stage. Observation only: norms, directions,
differences, and whether anything about the step looks constant across owners.
"""
import glob
import struct

WIDTH = 7168
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}


def read(prefix="/tmp/vec-all.bin"):
    """Final residual per owner, plus the vector handed to the first captured owner.

    One file per owner. Only the wanted records are unpacked; the rest are skipped by
    seeking, because each file holds a hundred-odd vectors and two are needed.
    """
    finals, inputs = {}, {}
    for path in sorted(glob.glob(f"{prefix}.*.bin")):
        with open(path, "rb") as handle:
            while True:
                header = handle.read(16)
                if len(header) < 16:
                    break
                owner, local, position, count = struct.unpack("<4i", header)
                if count <= 0 or count > 1 << 20:
                    break
                # Stage 120 increments the position counter before the record is
                # written, so it reports one position later than it belongs to.
                if local == 120 and position > 0:
                    position -= 1
                if position != 0 or local not in (0, 120):
                    handle.seek(4 * count, 1)
                    continue
                payload = handle.read(4 * count)
                if len(payload) < 4 * count:
                    break
                values = struct.unpack(f"<{count}f", payload)
                if local == 0:
                    inputs[owner] = values
                else:
                    finals[owner] = values
    first = min(inputs) if inputs else None
    return (inputs[first] if first is not None else None), finals


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return dot(v, v) ** 0.5


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else float("nan")


def main():
    x0, finals = read()
    if x0 is None:
        print("no input record found")
        return
    owners = sorted(finals)
    print(f"owners with a final residual: {len(owners)}  range {owners[0]}..{owners[-1]}")
    print(f"x0 is the vector handed to owner {owners[0]}, i.e. the output of owner {owners[0]-1}")

    seq = [x0] + [finals[o] for o in owners]
    labels = ["x0 (server)"] + [f"x{o}" for o in owners]

    print()
    print("=== norms along the sequence ===")
    print(f"{'step':>12s} {'kind':>5s} {'||x||':>12s} {'||x-prev||':>12s} "
          f"{'cos(x,prev)':>12s} {'cos(x,x0)':>11s} {'||x||/||x0||':>13s}")
    base = norm(x0)
    for i, v in enumerate(seq):
        kind = "" if i == 0 else ("MLA" if owners[i - 1] in MLA else "KDA")
        if i == 0:
            print(f"{labels[i]:>12s} {kind:>5s} {norm(v):12.5g} {'':>12s} {'':>12s} "
                  f"{1.0:11.4f} {1.0:13.4f}")
            continue
        prev = seq[i - 1]
        d = [v[k] - prev[k] for k in range(WIDTH)]
        print(f"{labels[i]:>12s} {kind:>5s} {norm(v):12.5g} {norm(d):12.5g} "
              f"{cosine(v, prev):12.5f} {cosine(v, x0):11.4f} {norm(v)/base if base else 0:13.4f}")

    print()
    print("=== growth ratio per step ===")
    ratios = []
    for i in range(1, len(seq)):
        a, b = norm(seq[i - 1]), norm(seq[i])
        if a:
            ratios.append(b / a)
    if ratios:
        print(f"steps {len(ratios)}  min {min(ratios):.5f}  max {max(ratios):.5f}  "
              f"mean {sum(ratios)/len(ratios):.5f}")
        kda = [r for r, o in zip(ratios, owners) if o not in MLA]
        mla = [r for r, o in zip(ratios, owners) if o in MLA]
        if kda:
            print(f"  KDA steps {len(kda)}: mean {sum(kda)/len(kda):.5f} min {min(kda):.5f} max {max(kda):.5f}")
        if mla:
            print(f"  MLA steps {len(mla)}: mean {sum(mla)/len(mla):.5f} min {min(mla):.5f} max {max(mla):.5f}")

    print()
    print("=== is the step direction similar from one owner to the next? ===")
    deltas = []
    for i in range(1, len(seq)):
        deltas.append([seq[i][k] - seq[i - 1][k] for k in range(WIDTH)])
    consecutive = [cosine(deltas[i], deltas[i + 1]) for i in range(len(deltas) - 1)]
    if consecutive:
        print(f"cos(delta_n, delta_n+1): min {min(consecutive):.4f} max {max(consecutive):.4f} "
              f"mean {sum(consecutive)/len(consecutive):.4f}")
    with_first = [cosine(deltas[0], d) for d in deltas]
    print(f"cos(delta_1, delta_n):   min {min(with_first):.4f} max {max(with_first):.4f} "
          f"mean {sum(with_first)/len(with_first):.4f}")

    print()
    print("=== does it converge? last ten steps ===")
    for i in range(max(1, len(seq) - 10), len(seq)):
        prev = seq[i - 1]
        d = [seq[i][k] - prev[k] for k in range(WIDTH)]
        print(f"  {labels[i]:>6s} ||x|| {norm(seq[i]):10.5g}  ||x-prev|| {norm(d):10.5g}  "
              f"relative {norm(d)/norm(seq[i]) if norm(seq[i]) else 0:8.5f}")


if __name__ == "__main__":
    main()
