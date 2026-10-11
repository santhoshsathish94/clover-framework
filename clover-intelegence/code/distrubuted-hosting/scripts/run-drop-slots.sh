#!/bin/bash
# Which slots earn the 22.0 MB of memcpy per token that carrying them costs?
#
# A slot cannot be rebuilt from anything the consuming layer holds: least squares
# against the other snapshots and the residual leaves 0.745 relative error at best.
# So any slot that is read must be stored. The question is only which are read.
#
# The closed-form drop test over the existing capture predicts:
#   slot 0  removable  median 0.999995, worst 0.72, never inverts a fold
#   slot 4  harmful    worst -0.069 at owner 58 stage 3, 39% of folds below 0.99
# Slot 4 is the falsification case. If dropping it is as harmless as dropping slot 0,
# the prediction is worthless and so is the closed form behind it.
#
# Slot 0 is the prompt-bearing channel, so run 2 is also a test of the design intent:
# it is the only slot the folds can spare, and the only one that varies by prompt.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/drop.log
WANT=${1:-8}
rm -f /tmp/drop.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; drop=$2; shift 2
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/dr.in /tmp/dr.out /tmp/dr.err
    mkfifo /tmp/dr.in
    (
        exec 9<>/tmp/dr.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS=46 CLOVER_HARDMAX_STAGES=3,21
        export CLOVER_DROP_SLOTS="$drop"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/dr.in >/tmp/dr.out 2>/tmp/dr.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/dr.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/dr.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/dr.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-26s %-8s %s\n' "$label" "${drop:-none}" \
        "$(grep -A4 '^REQUEST' /tmp/dr.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/dr.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "hardmax at layer 46 throughout; slots removed from the fold sources only,"
echo "still created and still carried, because the per-layer count is a compile-time contract"
echo

run "anchor, nothing dropped"  ""   $FR
run "drop slot 0 (predicted ok)" 0  $FR
run "drop slot 4 (FALSIFY)"      4  $FR
run "drop slots 1-7, keep 0"   1-7  $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/drop.done
