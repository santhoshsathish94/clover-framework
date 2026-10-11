#!/usr/bin/env python3
"""Are attn and moe actually functions of the input, or near-constant?

Their norms barely moved while the input norm spanned 0 to 97. Norm is a weak test:
two vectors of equal length can point anywhere. This measures direction.

If either term were constant, a precomputed vector would reproduce it. The question is
decided by the cosines, not by the norms.
"""
import struct

WIDTH = 7168
NAMES = ["zero", "u", "v", "u+v", "2u", "w(held)"]


def read_runs(path):
    runs, current = [], None
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                break
            _o, local, _p, count = struct.unpack("<4i", header)
            values = struct.unpack(f"<{count}f", handle.read(4 * count))
            if local == 0:
                current = {}
                runs.append(current)
            if current is not None:
                current[local] = values
    return runs


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(v):
    return dot(v, v) ** 0.5


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else float("nan")


def table(title, vectors):
    print(f"=== {title} ===")
    print("            " + "".join(f"{n:>10s}" for n in NAMES))
    for i, a in enumerate(vectors):
        row = "".join(f"{cosine(a, b):10.4f}" for b in vectors)
        print(f"{NAMES[i]:12s}{row}")
    print()


def main():
    runs = read_runs("/tmp/vec46.bin")
    attn = [r[9] for r in runs]
    moe = [[r[116][i] - r[20][i] for i in range(WIDTH)] for r in runs]
    delta = [[r[120][i] - r[0][i] for i in range(WIDTH)] for r in runs]

    table("cosine between attn vectors", attn)
    table("cosine between moe vectors", moe)
    table("cosine between (output - input)", delta)

    print("=== if moe were simply its zero-input value, how wrong would that be? ===")
    base = moe[0]
    print(f"{'probe':12s} {'||moe||':>10s} {'||moe-base||':>14s} {'relative':>10s} {'cosine':>8s}")
    for i, m in enumerate(moe):
        d = [m[k] - base[k] for k in range(WIDTH)]
        print(f"{NAMES[i]:12s} {norm(m):10.5g} {norm(d):14.5g} "
              f"{norm(d)/norm(m) if norm(m) else 0:10.5g} {cosine(m, base):8.4f}")

    print()
    print("=== same question for attn ===")
    base = attn[0]
    print(f"{'probe':12s} {'||attn||':>10s} {'||attn-base||':>14s} {'relative':>10s} {'cosine':>8s}")
    for i, a in enumerate(attn):
        d = [a[k] - base[k] for k in range(WIDTH)]
        print(f"{NAMES[i]:12s} {norm(a):10.5g} {norm(d):14.5g} "
              f"{norm(d)/norm(a) if norm(a) else 0:10.5g} {cosine(a, base):8.4f}")

    print()
    print("=== how much of the output is the input carried straight through? ===")
    for i, r in enumerate(runs):
        out, inp = r[120], r[0]
        c = cosine(out, inp)
        print(f"{NAMES[i]:12s} cosine(out, in) {c:7.4f}   ||out||/||in|| "
              f"{norm(out)/norm(inp) if norm(inp) else float('inf'):7.4f}")


if __name__ == "__main__":
    main()
