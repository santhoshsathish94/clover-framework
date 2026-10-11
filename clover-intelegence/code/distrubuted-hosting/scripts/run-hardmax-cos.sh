#!/bin/bash
# Hardmax at every fold whose cos(aggregate, residual) exceeds 0.9.
#
# The owner lists differ between the two folds, which is why CLOVER_HARDMAX_LAYERS_3
# and CLOVER_HARDMAX_LAYERS_21 exist. Sixteen owners clear 0.9 at stage 3 and
# twenty-five at stage 21, overlapping only in part.
#
# This is not the same change as the residual fold just measured. Hardmax gives the
# whole weight to whichever source wins, and at some of these owners the winner is
# not the residual: owner 5 clears 0.94 and 0.93 yet the residual wins 0 of 5 at
# both folds, so hardmax there hands everything to a snapshot. A high cosine says
# the fold's output is close to the residual, not that the residual is winning.
#
# Immediately prior result, same prompt and build, for the residual fold rather than
# hardmax: stage 3 took eleven owners down to cos 0.89 with the output unchanged,
# stage 21 broke at cos 0.93, and owner 13 stage 3 alone at cos 0.0001 changed it.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/hardmax-cos.log
WANT=${1:-8}
rm -f /tmp/hardmaxcos.done
: > "$LOG"
exec >>"$LOG" 2>&1

S3_90='3-12,35,43,46,78,80-81'
S21_90='1-5,7-12,34,42,46-47,53,56-59,83,88-91'
S3_99='3,6-12,46'
S21_99='9-10,12,46'

run() {
    label=$1; at3=$2; at21=$3; stages=$4; shift 4
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/hc.in /tmp/hc.out /tmp/hc.err
    mkfifo /tmp/hc.in
    (
        exec 9<>/tmp/hc.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_STAGES="$stages"
        export CLOVER_HARDMAX_LAYERS_3="$at3" CLOVER_HARDMAX_LAYERS_21="$at21"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/hc.in >/tmp/hc.out 2>/tmp/hc.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/hc.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/hc.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/hc.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-28s %-6s %s\n' "$label" "$stages" \
        "$(grep -A4 '^REQUEST' /tmp/hc.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/hc.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "hardmax owner lists by fold, from cos(aggregate,residual):"
echo "  stage 3  > 0.9  : $S3_90   (16 owners)"
echo "  stage 21 > 0.9  : $S21_90  (25 owners)"
echo "  stage 3  > 0.99 : $S3_99"
echo "  stage 21 > 0.99 : $S21_99"
echo

run "anchor 46 both folds"     46      46      3,21  $FR
run "cos > 0.9  both folds"    $S3_90  $S21_90 3,21  $FR
run "cos > 0.9  stage 3 only"  $S3_90  $S21_90 3     $FR
run "cos > 0.9  stage 21 only" $S3_90  $S21_90 21    $FR
run "cos > 0.99 both folds"    $S3_99  $S21_99 3,21  $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/hardmaxcos.done
