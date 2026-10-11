#!/bin/bash
# Does the aggregate capture actually reach disk? The writer was added to all 92
# layers and built, but CLOVER_VECTORS_KEEP=weights kept only 2000-2999, so the
# new 4000+stage records were being dropped by the filter that was supposed to
# make the capture affordable. The filter now admits 4000-4999 as well. One token
# is enough to answer whether anything lands.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/aggregate-capture.log
rm -f /tmp/aggregate.done
: > "$LOG"
exec >>"$LOG" 2>&1

pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/ag.in /tmp/ag.out /tmp/ag.err /tmp/agv.*
mkfifo /tmp/ag.in
(
    exec 9<>/tmp/ag.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    export CLOVER_VECTORS=/tmp/agv CLOVER_VECTORS_KEEP=weights
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/ag.in >/tmp/ag.out 2>/tmp/ag.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/ag.out 2>/dev/null && break; sleep 1; done
grep -q READY /tmp/ag.out 2>/dev/null || { echo "never became ready"; tail -5 /tmp/ag.err; touch /tmp/aggregate.done; exit 1; }

printf '1 1008 10484 318 15383 387\n' > /tmp/ag.in
for _ in $(seq 1 600); do grep -q '^REQUEST' /tmp/ag.out 2>/dev/null && break; sleep 2; done
sleep 2
pkill -f '[b]in/pipeline-serve'; sleep 1
rm -f /tmp/ag.in

grep '^REQUEST' /tmp/ag.out | head -1
echo "capture files: $(ls /tmp/agv.* 2>/dev/null | wc -l)"
echo

python3 - <<'PY'
import glob, struct, collections, math
counts = collections.Counter()
owners = collections.defaultdict(set)
vectors = {}
for path in sorted(glob.glob("/tmp/agv.*")):
    raw = open(path, "rb").read()
    at = 0
    while at + 16 <= len(raw):
        owner, local, position, n = struct.unpack_from("<4i", raw, at)
        at += 16
        values = struct.unpack_from("<%df" % n, raw, at)
        at += 4 * n
        counts[local] += 1
        owners[local].add(owner)
        vectors.setdefault((owner, local), values)

print("local code    records  owners  width")
for local in sorted(counts):
    width = len(next(v for (o, l), v in vectors.items() if l == local))
    print("  %-10d %6d  %6d  %5d" % (local, counts[local], len(owners[local]), width))

print()
for local in (4003, 4021):
    sample = [(o, v) for (o, l), v in vectors.items() if l == local]
    if not sample:
        print("local %d: nothing captured" % local)
        continue
    sample.sort()
    owner, values = sample[0]
    norm = math.sqrt(sum(x * x for x in values))
    nonzero = sum(1 for x in values if x != 0.0)
    print("local %d owner %d: norm %.4f, %d/%d non-zero, first 4 %s"
          % (local, owner, norm, nonzero, len(values), ["%.5f" % x for x in values[:4]]))
PY

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/aggregate.done
