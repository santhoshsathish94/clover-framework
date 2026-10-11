#!/usr/bin/env python3
# The snapshots against the residual, which is the actual subject.
#
# A snapshot is pushed at stage 4 of every twelfth layer and is that layer's input.
# The fold then reads all of them alongside the residual. Slot k therefore belongs to
# layer 12k, and the number of slots a layer can see grows with its depth.
#
# Three questions, all answerable from the capture already on disk:
#
#   1. How different is each snapshot from the residual the fold saw? If a snapshot is
#      already the residual, it is not a second channel, it is a copy.
#   2. How different is each snapshot from the one before it? The cross-prompt table
#      said slots 1 upward barely move between prompts; this asks whether they move
#      between each other at all.
#   3. Is snapshot n equal to snapshot n-1 plus the residual? That is the scheme that
#      was asked for at the start. It has been run and read as text, but the vectors
#      it claims have never been checked against what the model actually holds.
#
# The two folds see different residuals: stage 3 reads it before the attention add at
# stage 20, stage 21 reads it after. Both are reported.
import glob, struct, math, collections, statistics

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def describe(label, xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        print("%-34s no data" % label)
        return
    print("%-34s n %4d  min %8.5f  median %8.5f  mean %8.5f  max %8.5f"
          % (label, len(xs), xs[0], statistics.median(xs), sum(xs) / len(xs), xs[-1]))

rows = collections.defaultdict(dict)
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        # Snapshots are dumped after stage 120 has already advanced the counter, so they
        # are labelled one position late. Omitting this compares different tokens.
        if 1000 <= local < 1008 and position > 0:
            position -= 1
        if local in (3, 21) or 1000 <= local < 1008:
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

print("records: %d owner/position pairs over %d owners"
      % (len(rows), len({o for o, _ in rows})))

slots_by_owner = collections.defaultdict(int)
for (owner, position), got in rows.items():
    count = sum(1 for local in got if 1000 <= local < 1008)
    slots_by_owner[owner] = max(slots_by_owner[owner], count)
first_owner_with = {}
for owner in sorted(slots_by_owner):
    first_owner_with.setdefault(slots_by_owner[owner], owner)
print("slot count -> first owner that has it: %s"
      % ", ".join("%d slots from owner %d" % (c, o) for c, o in sorted(first_owner_with.items())))
print()

to_r3 = collections.defaultdict(list)
to_r21 = collections.defaultdict(list)
ratio = collections.defaultdict(list)
to_prev = collections.defaultdict(list)
delta_cos = collections.defaultdict(list)
delta_ratio = collections.defaultdict(list)

for (owner, position), got in rows.items():
    r3, r21 = got.get(3), got.get(21)
    slots = sorted(local for local in got if 1000 <= local < 1008)
    for local in slots:
        k = local - 1000
        s = got[local]
        if r3:
            to_r3[k].append(cosine(s, r3))
            n3 = norm(r3)
            if n3 > 0:
                ratio[k].append(norm(s) / n3)
        if r21:
            to_r21[k].append(cosine(s, r21))
        if local - 1 in got and r3:
            previous = got[local - 1]
            to_prev[k].append(cosine(s, previous))
            delta = [a - b for a, b in zip(s, previous)]
            delta_cos[k].append(cosine(delta, r3))
            n3 = norm(r3)
            if n3 > 0:
                delta_ratio[k].append(norm(delta) / n3)

print("1. each snapshot against the residual the fold saw")
print("slot  n     cos to residual@3   cos to residual@21   ||snapshot|| / ||residual@3||")
for k in sorted(to_r3):
    print("  %2d %4d          %8.5f             %8.5f                     %8.3f"
          % (k, len(to_r3[k]), mean(to_r3[k]), mean(to_r21[k]), mean(ratio[k])))
print()

print("2. each snapshot against the one before it")
print("slot  n     cos to previous slot")
for k in sorted(to_prev):
    print("  %2d %4d          %8.5f" % (k, len(to_prev[k]), mean(to_prev[k])))
print()

print("3. is snapshot n equal to snapshot n-1 plus the residual?")
print("   testing the claim directly: cos(snapshot[n] - snapshot[n-1], residual)")
print("   and the size of that difference relative to the residual")
print("slot  n     cos(delta, residual)   ||delta|| / ||residual||")
for k in sorted(delta_cos):
    print("  %2d %4d           %8.5f                %8.3f"
          % (k, len(delta_cos[k]), mean(delta_cos[k]), mean(delta_ratio[k])))
print()

describe("all slots: cos to residual@3", [v for k in to_r3 for v in to_r3[k]])
describe("slots 1+: cos to residual@3", [v for k in to_r3 if k >= 1 for v in to_r3[k]])
describe("slots 1+: cos to previous", [v for k in to_prev for v in to_prev[k]])
describe("slots 1+: cos(delta, residual)", [v for k in delta_cos for v in delta_cos[k]])
describe("slots 1+: ||delta||/||residual||", [v for k in delta_ratio for v in delta_ratio[k]])
