#!/usr/bin/env bash
# What does a LONG answer cost? Every number measured in this repo so far is for a
# five-token prompt producing ONE output token.
#
# WHY THIS MATTERS AND IS NOT INFERABLE
# The source shows the shape but not the cost. In pipeline.c the generation loop is
#
#     for (produced = 0; produced < request.wanted; produced++) {
#         pipeline_advance(fleet, &session, index, tokens, filled, count, &timing);
#         filled = count; ... count++;
#     }
#
# so the first pass processes all five prompt positions and every later pass processes
# exactly one new position -- but each pass still walks all 93 layers and selects
# experts afresh for that token. Whether token two re-reads the ~97 GB that token one
# just pulled through, or finds it still resident, decides whether a hundred-token
# answer takes two minutes or two hours. That is a property of the running system, not
# of the source, so it has to be measured.
#
# Already known and relevant: a SECOND BATCH of the identical prompt re-read all 104 GB,
# because the working set is larger than RAM and the sweep evicts its own early layers.
# If the same holds between consecutive tokens, cost is linear in answer length. If
# consecutive tokens within one request reuse what is resident, it is not. Both are
# plausible; neither is established.
#
# Reports wall time and bytes read per request, so the MARGINAL cost of token N can be
# taken as the difference between successive runs rather than assumed.
set -u

C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LANES=${LANES:-1}
THREADS=${THREADS:-16}
BIND=${BIND:-close}
WANTED_LIST=${WANTED_LIST:-"1 2 4 8 16"}
PROMPT_IDS="1008 10484 318 15383 387"

cd "$C/pipeline" || exit 1

echo "lanes=$LANES threads=$THREADS bind=$BIND"
echo "prompt is 5 tokens; first field of a request line is request.wanted, the number"
echo "of output tokens. Confirmed empirically below by the length of the answer text."
echo
printf "%7s %12s %12s %14s %14s %s\n" "wanted" "wall s" "marginal s" "GB read" "marginal GB" "answer"

prev_t=""; prev_g=""
for wanted in $WANTED_LIST; do
    : > /tmp/long-in.txt
    i=0; while [ $i -lt "$LANES" ]; do echo "$wanted $PROMPT_IDS" >> /tmp/long-in.txt; i=$((i+1)); done

    r0=$(awk '/^nvme0n1|^ *259 *0 nvme0n1/ {print}' /dev/null; grep -E ' (nvme0n1|nvme1n1) ' /proc/diskstats | awk '{s+=$6} END{print s}')
    env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND="$BIND" OMP_PLACES=cores \
        ./bin/pipeline-serve "$C" serve < /tmp/long-in.txt > /tmp/long-$wanted.out 2>/tmp/long-$wanted.err
    rc=$?
    r1=$(grep -E ' (nvme0n1|nvme1n1) ' /proc/diskstats | awk '{s+=$6} END{print s}')

    line=$(grep -m1 '^REQUEST 0 ' /tmp/long-$wanted.out)
    if [ -z "$line" ]; then
        printf "%7s  *** FAILED (exit %s): %s\n" "$wanted" "$rc" "$(tail -1 /tmp/long-$wanted.out)"
        continue
    fi
    t=$(echo "$line" | awk '{sub(/s$/,"",$4); print $4}')
    answer=$(echo "$line" | sed 's/^[^[]*\[//; s/\]$//')
    gb=$(echo "scale=2; ($r1 - $r0) * 512 / 1073741824" | bc)

    if [ -n "$prev_t" ]; then
        dt=$(echo "scale=3; $t - $prev_t" | bc)
        dg=$(echo "scale=2; $gb - $prev_g" | bc)
    else
        dt="-"; dg="-"
    fi
    printf "%7s %12s %12s %14s %14s [%s]\n" "$wanted" "$t" "$dt" "$gb" "$dg" "$answer"
    prev_t=$t; prev_g=$gb
done

echo
echo "marginal columns are the difference from the previous row, so they cover the"
echo "tokens added between the two. Divide by the number of tokens added to get a"
echo "per-token cost: the step from 8 to 16 adds eight tokens, not one."
touch /tmp/long-answer.done
