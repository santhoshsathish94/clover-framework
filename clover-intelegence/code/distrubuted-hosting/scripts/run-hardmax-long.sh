#!/bin/bash
# Sixteen tokens per prompt, blended against hard-maxed. One token tells you almost
# nothing; a continuation shows whether the two paths stay together or separate.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/hardmax-long.log
WANT=16
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; hard=$2; shift 2
    ids="$*"
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/hl.in /tmp/hl.out /tmp/hl.err
    mkfifo /tmp/hl.in
    (
        exec 9<>/tmp/hl.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX="$hard"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/hl.in >/tmp/hl.out 2>/tmp/hl.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/hl.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$ids" > /tmp/hl.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/hl.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    echo "--- $label ---"
    sed -n 's/^REQUEST [0-9]* OK /time /p' /tmp/hl.out | head -1
    # cat -v so a non printing token is visible rather than silently lost
    grep -A99 '^REQUEST' /tmp/hl.out | cat -v | head -8
    echo
    rm -f /tmp/hl.in
}

echo "started $(date -u +%FT%TZ), $WANT tokens per run"
echo
echo "########## the capital of france is ##########"
run "france  softmax" 0 1008 10484 318 15383 387
run "france  hardmax" 1 1008 10484 318 15383 387
echo "########## the capital of japan is ##########"
run "japan   softmax" 0 1008 10484 318 10417 387
run "japan   hardmax" 1 1008 10484 318 10417 387
echo "########## two tokens ##########"
run "onthe   softmax" 0 91019 25528
run "onthe   hardmax" 1 91019 25528
echo "finished $(date -u +%FT%TZ)"
