#!/bin/bash
# Where is the fold already doing nothing?
#
# The fold blends snapshots and the residual, residual last. The blend uses the raw
# vectors; stages 4 and 22 then push the result through transformer_normalize, which
# divides the scale back out. So only direction survives, and the question "can this
# fold be replaced by the residual at weight 1" is answered by the cosine between
# what the fold produced and the residual that fed it. At 1.0 the replacement is
# free. Below that it costs something, and how much is what we are looking for.
#
# No new capture code is needed. The residual is already written at every stage and
# the fold does not modify it, so local 3 and local 21 are exactly the vectors the
# two folds consumed. The keep filter was hiding them; this runs unfiltered, about
# 1.7 GB for one prompt against 120 GB free.
#
# Captured under the configuration that produced the output being protected:
# hardmax at layer 46, both folds, which gave [ Paris. The Eiffel Tower is located].
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/fold-cosine.log
rm -f /tmp/foldcos.done
: > "$LOG"
exec >>"$LOG" 2>&1

pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/fc.in /tmp/fc.out /tmp/fc.err /tmp/fcv.*
mkfifo /tmp/fc.in
(
    exec 9<>/tmp/fc.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    export CLOVER_HARDMAX_LAYERS=46 CLOVER_HARDMAX_STAGES=3,21
    export CLOVER_VECTORS=/tmp/fcv
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/fc.in >/tmp/fc.out 2>/tmp/fc.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/fc.out 2>/dev/null && break; sleep 1; done
grep -q READY /tmp/fc.out 2>/dev/null || { echo "never became ready"; tail -5 /tmp/fc.err; touch /tmp/foldcos.done; exit 1; }

printf '1 1008 10484 318 15383 387\n' > /tmp/fc.in
for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/fc.out 2>/dev/null && break; sleep 2; done
sleep 3
pkill -f '[b]in/pipeline-serve'; sleep 1
rm -f /tmp/fc.in

grep '^REQUEST' /tmp/fc.out | head -1
echo "capture size: $(du -sh /tmp/fcv.* 2>/dev/null | awk '{s+=$1} END {print NR" files"}') $(du -ch /tmp/fcv.* 2>/dev/null | tail -1 | cut -f1)"
echo

python3 - <<'PY'
import glob, struct, math, collections, re

WANT = {3, 21, 2003, 2021, 4003, 4021}

def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na > 0.0 and nb > 0.0 else float("nan")

rows = collections.defaultdict(dict)   # (owner, position) -> {local: values}
for path in glob.glob("/tmp/fcv.*"):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        if local in WANT:
            rows[(owner, position)][local] = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n

per_owner = collections.defaultdict(lambda: {3: [], 21: [], "w3": [], "w21": [], "win3": 0, "win21": 0, "n": 0})
for (owner, position), got in rows.items():
    entry = per_owner[owner]
    entry["n"] += 1
    for stage, aggregate_code, weight_code in ((3, 4003, 2003), (21, 4021, 2021)):
        residual = got.get(stage)
        aggregate = got.get(aggregate_code)
        if residual and aggregate:
            entry[stage].append(cosine(aggregate, residual))
        weights = got.get(weight_code)
        if weights:
            entry["w%d" % stage].append(weights[-1])          # residual is the LAST source
            if max(range(len(weights)), key=lambda i: weights[i]) == len(weights) - 1:
                entry["win%d" % stage] += 1

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

print("owner  n  cos(agg,resid) s3   s21      residual weight s3   s21     residual wins s3/s21")
safe90, safe99 = [], []
for owner in sorted(per_owner):
    e = per_owner[owner]
    c3, c21 = mean(e[3]), mean(e[21])
    w3, w21 = mean(e["w3"]), mean(e["w21"])
    print("  %3d %3d        %8.5f %8.5f           %8.5f %8.5f          %3d/%3d"
          % (owner, e["n"], c3, c21, w3, w21, e["win3"], e["win21"]))
    worst = min(c3, c21)
    if not math.isnan(worst):
        if worst > 0.99: safe99.append(owner)
        if worst > 0.90: safe90.append(owner)

def ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None: start = prev = n; continue
        if n == prev + 1: prev = n; continue
        out.append((start, prev)); start = prev = n
    if start is not None: out.append((start, prev))
    return ",".join(str(a) if a == b else "%d-%d" % (a, b) for a, b in out)

print()
print("owners where BOTH folds have cos > 0.99 : %d" % len(safe99))
print("  %s" % ranges(safe99))
print("owners where BOTH folds have cos > 0.90 : %d" % len(safe90))
print("  %s" % ranges(safe90))
PY

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/foldcos.done
