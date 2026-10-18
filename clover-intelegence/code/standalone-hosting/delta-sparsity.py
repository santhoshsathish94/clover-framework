#!/usr/bin/env python3
"""Is the change between consecutive layer inputs sparse enough to be worth exploiting?

W x_{t+1} = W x_t + W d.  Computing W d needs only the columns of W where d is
non-zero, so a d concentrated in k of 7168 dims would cut the sweep to k/7168.
This measures how concentrated d actually is.
"""
import struct
import sys

E = 7168
NLAY = 93
POS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

raw = open("/tmp/trace55/layers.bin", "rb").read()
vec = E * 4
total = len(raw) // vec
print(f"vectors dumped: {total}  ({len(raw)/1e6:.1f} MB)")

# The dump is written layer by layer, all positions of a pass together.
def get(layer, pos):
    idx = layer * POS + pos
    off = idx * vec
    return struct.unpack_from(f"{E}f", raw, off)

def norm(v):
    return sum(x * x for x in v) ** 0.5

print()
print(f"{'layer':>5} {'||x||':>10} {'||d||':>10} {'||d||/||x||':>12} "
      f"{'dims for 90%':>13} {'99%':>7} {'99.9%':>8}")

for layer in (0, 1, 23, 46, 69, 92):
    if (layer + 1) * POS > total:
        continue
    a = get(layer, 0)
    b = get(layer, 1)
    d = [b[i] - a[i] for i in range(E)]
    na, nd = norm(a), norm(d)
    sq = sorted((x * x for x in d), reverse=True)
    tot = sum(sq)
    cum, marks, need = 0.0, {}, [0.90, 0.99, 0.999]
    for i, s in enumerate(sq, 1):
        cum += s
        while need and cum >= need[0] * tot:
            marks[need.pop(0)] = i
    print(f"{layer:5d} {na:10.4f} {nd:10.4f} {nd/na:12.4f} "
          f"{marks.get(0.90, E):13d} {marks.get(0.99, E):7d} {marks.get(0.999, E):8d}")

print()
print(f"for reference a full sweep uses all {E} dims;")
print("k dims means the trunk read shrinks to k/7168 only if d is exactly that sparse.")
