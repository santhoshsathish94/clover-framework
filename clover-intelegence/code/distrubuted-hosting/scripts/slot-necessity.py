#!/usr/bin/env python3
# Must slot k be stored at all?
#
# A slot is carried from layer to layer at a cost of 22.0 MB of memcpy per token
# (399 vectors in and 406 out across the 92 layers, 7168 floats each). It earns that
# only if some fold's output would differ had it been absent.
#
# Dropping slot j from a fold and renormalising the remaining weights has a closed
# form, so this needs no new run and no vector reconstruction:
#
#   a       = sum_i alpha_i s_i
#   a_drop  = sum_{i != j} (alpha_i / (1 - alpha_j)) s_i
#   a - a_drop = (alpha_j / (1 - alpha_j)) (s_j - a)
#
# So the damage from dropping slot j depends only on ||a||, ||s_j||, <a, s_j> and
# alpha_j. Reported as the cosine between the fold's real output and what it would
# have produced without the slot: 1.0 means the slot was doing nothing.
#
# Stage-loop records carry the correct position; snapshot dumps run after stage 120
# advanced the counter and need -1.
import glob, struct, math, collections, statistics

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

rows = collections.defaultdict(dict)
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local in (2003, 2021, 4003, 4021) or 1000 <= local < 1008:
            if 1000 <= local < 1008 and position > 0:
                position -= 1
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

per_slot = collections.defaultdict(list)
per_owner_slot = collections.defaultdict(list)

for (owner, position), got in rows.items():
    slots = sorted(l for l in got if 1000 <= l < 1008)
    for stage, w_code, a_code in ((3, 2003, 4003), (21, 2021, 4021)):
        weights, a = got.get(w_code), got.get(a_code)
        if not weights or not a or len(weights) != len(slots) + 1:
            continue
        aa = dot(a, a)
        if aa <= 0.0:
            continue
        for index, local in enumerate(slots):
            k = local - 1000
            alpha = weights[index]
            if alpha >= 1.0:
                continue
            s = got[local]
            beta = alpha / (1.0 - alpha)
            asj = dot(a, s)
            ss = dot(s, s)
            num = (1.0 + beta) * aa - beta * asj
            den2 = (1.0 + beta) ** 2 * aa - 2.0 * (1.0 + beta) * beta * asj + beta * beta * ss
            if den2 <= 0.0:
                continue
            cos = num / math.sqrt(aa * den2)
            per_slot[k].append(cos)
            per_owner_slot[(owner, k, stage)].append(cos)

def describe(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    return (len(xs), xs[0], statistics.median(xs), sum(xs) / len(xs))

print("cos(fold output, fold output with slot k removed and weights renormalised)")
print("1.0 means removing the slot would not have changed that fold at all")
print()
print("slot     n      worst     median       mean    folds below 0.99   below 0.9")
for k in sorted(per_slot):
    xs = per_slot[k]
    n, worst, med, mean = describe(xs)
    below99 = sum(1 for x in xs if x < 0.99)
    below90 = sum(1 for x in xs if x < 0.90)
    print("  %2d %5d  %9.6f  %9.6f  %9.6f   %6d (%4.1f%%)  %5d (%4.1f%%)"
          % (k, n, worst, med, mean, below99, 100.0 * below99 / n, below90, 100.0 * below90 / n))

print()
print("the worst single fold for each slot, which is where dropping it would hurt most")
print("slot   owner  stage      cos")
for k in sorted(per_slot):
    worst_key, worst_val = None, 2.0
    for (owner, slot, stage), xs in per_owner_slot.items():
        if slot != k:
            continue
        m = min(xs)
        if m < worst_val:
            worst_val, worst_key = m, (owner, stage)
    if worst_key:
        print("  %2d   %5d  %5d  %9.6f" % (k, worst_key[0], worst_key[1], worst_val))

print()
print("owners where some slot is load-bearing, cos below 0.9 (stage, slot, cos)")
hits = sorted((min(xs), owner, k, stage) for (owner, k, stage), xs in per_owner_slot.items()
              if min(xs) < 0.9)
for cos, owner, k, stage in hits[:25]:
    print("  owner %2d stage %2d slot %d  cos %9.6f" % (owner, stage, k, cos))
print("  ... %d owner/slot/stage combinations below 0.9 in total" % len(hits))
