#!/bin/bash
# Can the fold be replaced by the residual where it was already producing it?
#
# cos(aggregate, residual) was measured per owner under the configuration being
# protected, hardmax at layer 46 both folds, which gives
# [ Paris. The Eiffel Tower is located]. Where that cosine is ~1 the fold is
# already a copy of the residual and forcing the copy should cost nothing. Where
# it is low the snapshots are carrying something and it should cost a lot.
#
# Both directions are tested, because a map that only ever predicts "safe" is not
# being tested. Run 6 forces residual-1 at owner 13 stage 3 alone, measured at
# cos 0.00012, near perpendicular. One fold. If the map means anything, that one
# fold should do more damage than the ten near-parallel ones together.
#
# Layer 46 is in every set on purpose. Hardmax already reduced its fold to the
# residual at weight 1, so adding it must change nothing; if a set containing it
# differs from the anchor, the cause is elsewhere in that set.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/residual-fold.log
WANT=${1:-8}
rm -f /tmp/residual.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; layers=$2; stages=$3; shift 3
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/rf.in /tmp/rf.out /tmp/rf.err
    mkfifo /tmp/rf.in
    (
        exec 9<>/tmp/rf.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS=46 CLOVER_HARDMAX_STAGES=3,21
        export CLOVER_FOLD_LAYERS="$layers" CLOVER_FOLD_STAGES="$stages"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/rf.in >/tmp/rf.out 2>/tmp/rf.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/rf.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/rf.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/rf.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-30s %-12s %-5s %s\n' "$label" "${layers:-none}" "$stages" \
        "$(grep -A4 '^REQUEST' /tmp/rf.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/rf.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "every run keeps hardmax at layer 46, both folds"
echo "cos(aggregate,residual) > 0.99 both folds: 9,10,12,46"
echo "cos(aggregate,residual) > 0.90 both folds: 3-5,7-12,46"
echo "cos > 0.99 at stage 3 only:                3,6-12,46"
echo

run "anchor, no residual fold"   ""          3,21  $FR
run "strict 0.99, both folds"    9-10,12,46  3,21  $FR
run "loose 0.90, both folds"     3-5,7-12,46 3,21  $FR
run "stage 3 only, cos > 0.99"   3,6-12,46   3     $FR
run "stage 3 only, down to 0.89" 2-12,46     3     $FR
run "FALSIFY: owner 13 stage 3"  13          3     $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/residual.done
