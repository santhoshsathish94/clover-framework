#!/bin/bash
# The new default against the old behaviour, to confirm both paths still run and to
# keep a record of what each produces. Eight tokens is enough to see a continuation
# without a half hour per cell.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/default-check.log
: > "$LOG"
exec >>"$LOG" 2>&1

serve() {
    label=$1; soft=$2; want=$3; shift 3
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/dc.in /tmp/dc.out /tmp/dc.err
    mkfifo /tmp/dc.in
    (
        exec 9<>/tmp/dc.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        [ -n "$soft" ] && export CLOVER_SOFTMAX="$soft"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/dc.in >/tmp/dc.out 2>/tmp/dc.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/dc.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$want" "$*" > /tmp/dc.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/dc.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    echo "--- $label ---"
    grep -A20 '^REQUEST' /tmp/dc.out | cat -v | head -5
    echo
    rm -f /tmp/dc.in
}

echo "started $(date -u +%FT%TZ)"
echo "the capital of france is"
serve "default (weight 1)" ""  8 1008 10484 318 15383 387
serve "CLOVER_SOFTMAX=1"   1   8 1008 10484 318 15383 387
echo "the capital of japan is"
serve "default (weight 1)" ""  8 1008 10484 318 10417 387
serve "CLOVER_SOFTMAX=1"   1   8 1008 10484 318 10417 387
echo "finished $(date -u +%FT%TZ)"
