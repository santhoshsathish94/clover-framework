#!/bin/bash
# The same prompts with the aggregate blended, then with the winning source taking
# all the weight. Recording what comes out. No judgement, just both outputs.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/hardmax.log
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; hard=$2; shift 2
    ids="$*"
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/h.in /tmp/h.out /tmp/h.err
    mkfifo /tmp/h.in
    (
        exec 9<>/tmp/h.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX="$hard"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/h.in >/tmp/h.out 2>/tmp/h.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/h.out 2>/dev/null && break; sleep 1; done
    printf '2 %s\n' "$ids" > /tmp/h.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/h.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-26s %s\n' "$label" "$(grep '^REQUEST' /tmp/h.out | head -1 | tr -d '\n')"
    rm -f /tmp/h.in
}

echo "started $(date -u +%FT%TZ)"
echo "prompt: the capital of france is"
run "france  softmax" 0 1008 10484 318 15383 387
run "france  hardmax" 1 1008 10484 318 15383 387
echo
echo "prompt: the capital of japan is"
run "japan   softmax" 0 1008 10484 318 10417 387
run "japan   hardmax" 1 1008 10484 318 10417 387
echo
echo "prompt: two tokens"
run "onthe   softmax" 0 91019 25528
run "onthe   hardmax" 1 91019 25528
echo "finished $(date -u +%FT%TZ)"
