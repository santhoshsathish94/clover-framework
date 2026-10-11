#!/usr/bin/env python3
# The fold written as algebra, checked against the bytes.
#
#   a = sum_j alpha_j s_j,  s_{c-1} = r
#   s_j = rho_j r + s_j_perp,  rho_j = <s_j,r>/||r||^2
#   a   = A r + p,  A = alpha_{c-1} + sum_{j<c-1} alpha_j rho_j,  p = sum_{j<c-1} alpha_j s_j_perp
#   cos(a,r) = 1 / sqrt(1 + tau^2),  tau = ||p|| / (A ||r||)
#
# Two things are checked rather than asserted:
#   1. the aggregate really is the weighted sum of the raw sources
#   2. the closed form for cos reproduces the measured cosine
# Then the per-snapshot contribution alpha_j * k_j is printed, k_j = ||s_j||/||r||,
# because that product is what decides the tilt, not the weight on its own.
#
# Stage-loop records (3, 21, 2003, 2021, 4003, 4021) carry the correct position.
# Snapshot records are dumped after stage 120 advanced the counter and need -1.
import glob, struct, math, collections, statistics

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(dot(v, v))

rows = collections.defaultdict(dict)
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        keep = local in (3, 21, 2003, 2021, 4003, 4021) or 1000 <= local < 1008
        if keep:
            if 1000 <= local < 1008 and position > 0:
                position -= 1
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

recon_err, cos_err = [], []
contributions = collections.defaultdict(list)
examples = {}

for (owner, position), got in sorted(rows.items()):
    slots = sorted(l for l in got if 1000 <= l < 1008)
    for stage, w_code, a_code in ((3, 2003, 4003), (21, 2021, 4021)):
        r = got.get(stage)
        weights = got.get(w_code)
        a = got.get(a_code)
        if not r or not weights or not a:
            continue
        sources = [got[l] for l in slots] + [r]
        if len(weights) != len(sources):
            continue                      # residual-1 folds store a single 1.0; not a blend
        alpha = list(weights)

        rebuilt = [sum(alpha[j] * sources[j][i] for j in range(len(sources)))
                   for i in range(len(r))]
        scale = max(abs(x) for x in a) or 1.0
        recon_err.append(max(abs(x - y) for x, y in zip(rebuilt, a)) / scale)

        nr = norm(r)
        if nr == 0.0:
            continue
        A = alpha[-1] + sum(alpha[j] * dot(sources[j], r) / (nr * nr) for j in range(len(sources) - 1))
        p = [sum(alpha[j] * (sources[j][i] - dot(sources[j], r) / (nr * nr) * r[i])
                 for j in range(len(sources) - 1)) for i in range(len(r))]
        tau = norm(p) / (A * nr) if A != 0.0 else float("inf")
        predicted = 1.0 / math.sqrt(1.0 + tau * tau) * (1.0 if A > 0 else -1.0)
        measured = dot(a, r) / (norm(a) * nr)
        cos_err.append(abs(predicted - measured))

        for j in range(len(sources) - 1):
            k = norm(sources[j]) / nr
            contributions[(owner, stage)].append((j, alpha[j], k, alpha[j] * k))
        if (owner, stage) in ((6, 3), (12, 3), (13, 3), (46, 3), (84, 21)):
            examples[(owner, stage, position)] = (alpha[-1], tau, predicted, measured,
                                                  [(j, alpha[j], norm(sources[j]) / nr)
                                                   for j in range(len(sources) - 1)])

print("check 1: is the aggregate the weighted sum of the raw sources?")
print("  folds checked %d, worst relative error %.3e, median %.3e"
      % (len(recon_err), max(recon_err), statistics.median(recon_err)))
print()
print("check 2: does 1/sqrt(1+tau^2) reproduce the measured cosine?")
print("  folds checked %d, worst absolute error %.3e, median %.3e"
      % (len(cos_err), max(cos_err), statistics.median(cos_err)))
print()

print("worked examples: alpha_r is the residual's weight, k_j = ||s_j||/||r||")
for key in sorted(examples):
    owner, stage, position = key
    if position != 0:
        continue
    alpha_r, tau, predicted, measured, per = examples[key]
    print("  owner %2d stage %2d  alpha_r %.5f  tau %9.4f  cos predicted %.6f  measured %.6f"
          % (owner, stage, alpha_r, tau, predicted, measured))
    for j, aj, k in per:
        print("      slot %d  alpha %.5f  k %10.3f  alpha*k %10.4f" % (j, aj, k, aj * k))

print()
print("the product that decides the tilt, summed over snapshots, by owner (stage 3, position 0)")
print("owner   sum alpha_j*k_j   alpha_r")
for (owner, stage), items in sorted(contributions.items()):
    if stage != 3:
        continue
    per_position = len(items) // 5 if len(items) >= 5 else len(items)
    first = items[:per_position] if per_position else items
    total = sum(c for _, _, _, c in first)
    if owner in (2, 6, 11, 12, 13, 24, 46, 48, 84, 92):
        print("  %3d        %10.4f" % (owner, total))
