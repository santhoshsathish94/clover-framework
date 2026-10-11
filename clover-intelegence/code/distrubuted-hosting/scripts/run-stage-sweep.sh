#!/bin/bash
# Two questions, one sweep, now that the switch addresses owner AND stage.
#
# 1. The residual's cross-prompt cosine is flat at 1.000 for owners ~15-81 and
#    falls at both ends. If that map is the whole story, hardmaxing 15-81 should
#    be safe at full width. The width trend from the previous sweep predicts it
#    will not be. Either answer is informative; run 2 is the falsification test.
#
# 2. A layer folds twice, at stage 3 before attention and stage 21 before the MLP.
#    Every sweep so far moved both together. Runs 3-6 separate them.
#
# Run 1 is an anchor: same build, same prompt, must reproduce
# "[ Paris. The Eiffel Tower is located]" or the sweep is not comparable.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/stage-sweep.log
WANT=${1:-8}
rm -f /tmp/stage.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; layers=$2; stages=$3; shift 3
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/ss.in /tmp/ss.out /tmp/ss.err
    mkfifo /tmp/ss.in
    (
        exec 9<>/tmp/ss.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS="$layers" CLOVER_HARDMAX_STAGES="$stages"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/ss.in >/tmp/ss.out 2>/tmp/ss.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/ss.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/ss.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/ss.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-34s %s\n' "$label" "$(grep -A4 '^REQUEST' /tmp/ss.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/ss.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "residual cross-prompt cosine: owners 2-11 low, 15-81 flat at 1.000, 82-92 falls to 0.52"
echo "previously measured, both folds together:"
echo "  46          [ Paris. The Eiffel Tower is located]"
echo "  40-52       [ Paris. The city is located in the]"
echo "  36-56       [ Paris. The capital of the United States]"
echo "  30-60       [ Paris, and the capital of the world]"
echo

run "anchor 46, both folds"          46     3,21  $FR
run "flat 15-81, both folds"         15-81  3,21  $FR
run "flat 15-81, stage 3 only"       15-81  3     $FR
run "flat 15-81, stage 21 only"      15-81  21    $FR
run "narrow 40-52, stage 3 only"     40-52  3     $FR
run "narrow 40-52, stage 21 only"    40-52  21    $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/stage.done
