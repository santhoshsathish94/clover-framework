#!/bin/bash
# The original direction, tested directly: slot 0 keeps following the prompt, and
# every twelfth layer takes the residual at weight 1 instead of folding.
#
# Those seven layers are the ones that push a snapshot, at stage 4, which lands
# between the two folds. So at layer 12 the stage 3 fold still sees only slot 0 and
# the stage 21 fold sees slot 0 and the new slot 1. The folds are run separately for
# that reason.
#
# Prediction, written before the run. cos(aggregate, residual) at these seven:
#   owner  12  0.9998 / 0.9944      owner  48  0.6535 / 0.5615
#   owner  24  0.6678 / 0.7216      owner  60  0.8322 / 0.7429
#   owner  36  0.8768 / 0.7103      owner  72  0.6927 / 0.8954
#                                   owner  84  0.6259 / 0.7950
# Only owner 12 is above the 0.89 that stage 3 tolerated, and all seven are below the
# 0.99 that stage 21 needed. So this should degrade, and degrade worse at stage 21.
# If it holds anyway the cosine model is wrong, which is the more useful outcome.
#
# Run 5 is the counterpart: the layers that first SEE each new snapshot rather than
# the ones that make it. Owner 13 is where slot 1 first appears in a stage 3 fold, and
# owner 13 alone already moved the output once.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/twelfth.log
WANT=${1:-8}
rm -f /tmp/twelfth.done
: > "$LOG"
exec >>"$LOG" 2>&1

PUSHERS='12,24,36,48,60,72,84'
CONSUMERS='13,25,37,49,61,73,85'

run() {
    label=$1; layers=$2; stages=$3; shift 3
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/tw.in /tmp/tw.out /tmp/tw.err
    mkfifo /tmp/tw.in
    (
        exec 9<>/tmp/tw.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS=46 CLOVER_HARDMAX_STAGES=3,21
        export CLOVER_FOLD_LAYERS="$layers" CLOVER_FOLD_STAGES="$stages"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/tw.in >/tmp/tw.out 2>/tmp/tw.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/tw.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/tw.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/tw.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-32s %-24s %-5s %s\n' "$label" "${layers:-none}" "$stages" \
        "$(grep -A4 '^REQUEST' /tmp/tw.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/tw.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "every run keeps hardmax at layer 46, both folds; slot 0 is never touched"
echo

run "anchor, nothing forced"      ""          3,21  $FR
run "every 12th layer, both"      $PUSHERS    3,21  $FR
run "every 12th layer, stage 3"   $PUSHERS    3     $FR
run "every 12th layer, stage 21"  $PUSHERS    21    $FR
run "first consumers, both"       $CONSUMERS  3,21  $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/twelfth.done
