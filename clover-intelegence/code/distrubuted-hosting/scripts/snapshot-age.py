#!/usr/bin/env python3
# A snapshot is stage 4's copy of the layer's input, and the residual at stage 3 is
# also the layer's input. At the owner that creates a slot those two are the same
# object, so the cosine there must be 1. The previous pass averaged every owner
# holding a slot and reported 0.04, which hid that completely.
#
# This splits the same measurement by age: the owner that created the slot, against
# every owner that merely inherited it.
import glob, struct, math, collections, statistics

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine(a, b):
    na, nb = norm(a), norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else float("nan")

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
        if local == 3 or 1000 <= local < 1008:
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

at_birth, inherited = collections.defaultdict(list), collections.defaultdict(list)
ratio_birth, ratio_inherited = collections.defaultdict(list), collections.defaultdict(list)
age_cos = collections.defaultdict(list)

for (owner, position), got in rows.items():
    r3 = got.get(3)
    if not r3:
        continue
    for local in sorted(l for l in got if 1000 <= l < 1008):
        k = local - 1000
        s = got[local]
        c = cosine(s, r3)
        r = norm(s) / norm(r3) if norm(r3) else float("nan")
        born = 12 * k if k >= 1 else 0
        if owner == born:
            at_birth[k].append(c); ratio_birth[k].append(r)
        else:
            inherited[k].append(c); ratio_inherited[k].append(r)
            age_cos[owner - born].append(c)

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

print("slot  created at   n at birth  cos at birth  ratio at birth | n inherited  cos  ratio")
for k in sorted(set(at_birth) | set(inherited)):
    print("  %2d  layer %3s    %9d    %10.6f    %10.3f | %10d  %8.5f %8.3f"
          % (k, 12 * k if k else "0/server", len(at_birth[k]), mean(at_birth[k]), mean(ratio_birth[k]),
             len(inherited[k]), mean(inherited[k]), mean(ratio_inherited[k])))

print()
print("how fast a snapshot stops resembling the residual, by layers since it was made")
print("layers since created   n     cos to residual@3")
buckets = [(1, 1), (2, 3), (4, 6), (7, 11), (12, 23), (24, 47), (48, 95)]
for low, high in buckets:
    xs = [c for age, cs in age_cos.items() if low <= age <= high for c in cs]
    xs = [x for x in xs if not math.isnan(x)]
    if xs:
        print("  %3d to %-3d        %5d        %8.5f" % (low, high, len(xs), sum(xs) / len(xs)))
