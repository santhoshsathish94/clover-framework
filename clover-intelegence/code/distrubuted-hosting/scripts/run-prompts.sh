#!/bin/bash
# One configuration, six prompts: every twelfth layer takes the residual at weight 1,
# both folds, slot 0 untouched. Hardmax stays at layer 46 as in every other run.
#
# The question is whether the ANSWER holds, not whether the continuation is tidy. The
# first run of this produced [ Paris."} {"role": "user] and was reported as a failure;
# the answer in it was correct and the rest was chat-template scaffolding.
#
# No variants and no per-prompt anchor. If a prompt answers wrong, the anchor for that
# prompt alone is the next run, because a wrong answer cannot be attributed without it.
#
# Token ids came from encode-prompts.py, which reproduces the two prompts whose ids are
# in the repository's own test-reference.sh and round-trips every prompt below.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/prompts.log
WANT=${1:-8}
rm -f /tmp/prompts.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; shift
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/pr.in /tmp/pr.out /tmp/pr.err
    mkfifo /tmp/pr.in
    (
        exec 9<>/tmp/pr.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS=46 CLOVER_HARDMAX_STAGES=3,21
        export CLOVER_FOLD_LAYERS=12,24,36,48,60,72,84 CLOVER_FOLD_STAGES=3,21
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/pr.in >/tmp/pr.out 2>/tmp/pr.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/pr.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/pr.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/pr.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-28s %s\n' "$label" \
        "$(grep -A4 '^REQUEST' /tmp/pr.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/pr.in
}

echo "started $(date -u +%FT%TZ), $WANT tokens"
echo "configuration: CLOVER_FOLD_LAYERS=12,24,36,48,60,72,84 CLOVER_FOLD_STAGES=3,21"
echo "               hardmax at layer 46, both folds; slot 0 never touched"
echo "expected answers: Paris, Tokyo, Rome, Jupiter, four, blue"
echo

run "capital of France"   1008 10484 318 15383 387
run "capital of Japan"    1008 10484 318 10417 387
run "capital of Italy"    1008 10484 318 19509 387
run "largest planet"      1008 10604 19645 387
run "two plus two"        17285 9620 2069 28542
run "colour of the sky"   1008 12955 318 276 18985 387

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/prompts.done
