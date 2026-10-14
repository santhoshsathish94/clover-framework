#!/usr/bin/env bash
# A/B two configurations of the pipeline, with the controls the earlier ad-hoc runs lacked.
#
# The prefetch-touch result was judged on n=2 with the same arm running first in every
# round, which is exactly the shape that lets ordering masquerade as an effect. This
# fixes three things:
#
#   alternating order   A,B on odd rounds and B,A on even, so position in the round
#                       cancels. ROUNDS MUST BE EVEN or it does not cancel: at three
#                       rounds A runs first twice and B once. A null A/B caught exactly
#                       that, reporting a 0.95% difference between two identical arms
#                       with non-overlapping ranges.
#   discarded warm-up   one full round is run and thrown away before any value is kept
#   position reported   the median of every first-in-round run against every
#                       second-in-round run, so the size of the ordering effect is
#                       visible rather than assumed away
#   overlap reported    if the two arms' ranges overlap the summary says so, because a
#                       difference of medians across overlapping ranges is not a result
#
# It also runs at a chosen lane count, since this pipeline's answer changes sign between
# one lane and four, and reports per-request time rather than wall so startup is out.
#
# usage: ab-pipeline.sh LANES EVEN-ROUNDS 'labelA|ENV=v ...' 'labelB|ENV=v ...'
#   ab-pipeline.sh 4 4 'arena|CLOVER_ARENA=1' 'mmap|CLOVER_ARENA=0'
set -u
C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LANES=${1:?lanes}; ROUNDS=${2:?rounds}; A=${3:?arm A}; B=${4:?arm B}
if [ $((ROUNDS % 2)) -ne 0 ]; then
    echo "rounds must be even so each arm runs first the same number of times" >&2
    exit 1
fi
NAME_A=${A%%|*}; ENV_A=${A#*|}
NAME_B=${B%%|*}; ENV_B=${B#*|}
OUT=${AB_OUT:-/tmp/ab-pipeline.tsv}

cd "$C/pipeline" || exit 1
: > /tmp/ab-in.txt
i=0; while [ $i -lt "$LANES" ]; do echo "1 1008 10484 318 15383 387" >> /tmp/ab-in.txt; i=$((i+1)); done

run_one() {
    env $1 OMP_NUM_THREADS=${AB_THREADS:-16} OMP_PROC_BIND=close OMP_PLACES=cores \
        ./bin/pipeline-serve "$C" serve < /tmp/ab-in.txt 2>&1 \
        | awk '/^REQUEST 0 OK/ { sub(/s$/, "", $4); print $4; exit }'
}

: > "$OUT"
echo "warm-up round, discarded"
run_one "$ENV_A" > /dev/null; run_one "$ENV_B" > /dev/null

round=1
while [ $round -le "$ROUNDS" ]; do
    if [ $((round % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
    slot=1
    for which in $order; do
        if [ "$which" = A ]; then name=$NAME_A; settings=$ENV_A; else name=$NAME_B; settings=$ENV_B; fi
        value=$(run_one "$settings")
        printf "%s\t%s\t%s\t%s\n" "$name" "$round" "$slot" "$value" | tee -a "$OUT"
        slot=$((slot + 1))
    done
    round=$((round + 1))
done

python3 - "$OUT" "$NAME_A" "$NAME_B" <<'EOF'
import statistics, sys
rows, byslot = {}, {}
for line in open(sys.argv[1]):
    name, _, slot, value = line.split("\t")
    rows.setdefault(name, []).append(float(value))
    byslot.setdefault(int(slot), []).append(float(value))
print()
for name in sys.argv[2:4]:
    v = sorted(rows.get(name, []))
    if not v:
        continue
    print("%-10s n=%d  median %8.3f   min %8.3f   max %8.3f   %s"
          % (name, len(v), statistics.median(v), v[0], v[-1],
             " ".join("%.3f" % x for x in v)))

first, second = byslot.get(1, []), byslot.get(2, [])
if first and second:
    print("\nposition in round: first %8.3f   second %8.3f   effect %+.3f s"
          % (statistics.median(first), statistics.median(second),
             statistics.median(first) - statistics.median(second)))

a, b = sorted(rows.get(sys.argv[2], [])), sorted(rows.get(sys.argv[3], []))
if a and b:
    overlap = not (a[-1] < b[0] or b[-1] < a[0])
    gap = abs(statistics.median(b) - statistics.median(a))
    print("ratio of medians %.4f, difference %.3f s" % (statistics.median(b) / statistics.median(a), gap))
    print("ranges overlap: %s%s" % (overlap,
          "  -- a difference of medians across overlapping ranges is not a result"
          if overlap else ""))
    if first and second:
        position = abs(statistics.median(first) - statistics.median(second))
        if gap < position:
            print("the arm difference is smaller than the position effect, so it is not a result")
EOF
touch "${OUT%.tsv}.done"
