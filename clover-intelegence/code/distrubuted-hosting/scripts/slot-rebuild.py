#!/usr/bin/env python3
# Step 3. Could slot k be constructed instead of carried?
#
# Two different claims hide behind "construct":
#
#   recompute it   -- run layers 12k..n backwards. Attention and the expert mixture
#                     are not invertible, so this is not on the table and is not
#                     tested here.
#   rebuild it     -- express s_k as a linear combination of the vectors the layer
#                     already holds: the current residual and the other snapshots.
#                     If that works, s_k need not be stored, because the layer can
#                     reconstitute it from its remaining sources.
#
# The second is testable right now. For each fold, project s_k onto the span of the
# other sources and report the relative error. Small error means the slot is
# redundant with what is already present; error near 1 means it carries a direction
# nothing else has.
#
# Solved by normal equations with a small ridge, Gram matrix at most 8x8.
import glob, struct, math, collections, statistics

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def solve(gram, rhs, ridge=1e-6):
    n = len(rhs)
    m = [row[:] + [rhs[i]] for i, row in enumerate(gram)]
    for i in range(n):
        m[i][i] += ridge * (abs(m[i][i]) + 1.0)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-30:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] * inv
            if factor:
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        total = m[r][n] - sum(m[r][c] * out[c] for c in range(r + 1, n))
        out[r] = total / m[r][r]
    return out

rows = collections.defaultdict(dict)
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local == 3 or 1000 <= local < 1008:
            if 1000 <= local < 1008 and position > 0:
                position -= 1
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

per_slot = collections.defaultdict(list)
alone = collections.defaultdict(list)

for (owner, position), got in rows.items():
    r = got.get(3)
    slots = sorted(l for l in got if 1000 <= l < 1008)
    if not r or len(slots) < 2:
        continue
    if owner % 12 == 0:
        continue                      # the layer that made a slot trivially has it
    vectors = [got[l] for l in slots] + [r]
    names = [l - 1000 for l in slots] + [-1]
    for index, k in enumerate(names):
        if k < 0:
            continue
        target = vectors[index]
        basis = [v for i, v in enumerate(vectors) if i != index]
        gram = [[dot(a, b) for b in basis] for a in basis]
        rhs = [dot(v, target) for v in basis]
        coefficients = solve(gram, rhs)
        if coefficients is None:
            continue
        tt = dot(target, target)
        if tt <= 0.0:
            continue
        fitted = sum(coefficients[i] * rhs[i] for i in range(len(basis)))
        quad = sum(coefficients[i] * sum(gram[i][j] * coefficients[j] for j in range(len(basis)))
                   for i in range(len(basis)))
        err2 = max(tt - 2.0 * fitted + quad, 0.0)
        per_slot[k].append(math.sqrt(err2 / tt))
        # the same fit using the residual alone, which is the simplest possible rebuild
        lam = dot(r, target) / dot(r, r) if dot(r, r) > 0 else 0.0
        e2 = max(tt - 2.0 * lam * dot(r, target) + lam * lam * dot(r, r), 0.0)
        alone[k].append(math.sqrt(e2 / tt))

print("relative error rebuilding slot k from the OTHER sources that layer already holds")
print("0.0 = perfectly reconstructable and need not be stored")
print("1.0 = nothing else in the layer points that way")
print()
print("slot     n     best     median      mean      worst")
for k in sorted(per_slot):
    xs = sorted(per_slot[k])
    print("  %2d %5d  %7.4f  %9.4f %9.4f  %9.4f"
          % (k, len(xs), xs[0], statistics.median(xs), sum(xs) / len(xs), xs[-1]))

print()
print("same, but rebuilding from the current residual alone (scalar multiple)")
print("slot     n   median     mean")
for k in sorted(alone):
    xs = sorted(alone[k])
    print("  %2d %5d %8.4f %8.4f" % (k, len(xs), statistics.median(xs), sum(xs) / len(xs)))
