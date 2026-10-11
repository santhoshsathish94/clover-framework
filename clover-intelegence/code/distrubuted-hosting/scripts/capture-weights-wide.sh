#!/bin/bash
# Aggregate weights only, across several positions and several prompts, under the new
# default. Weights are nine floats, so many positions cost almost nothing to record.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/wide-capture.log
WANT=${1:-6}
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    name=$1; shift
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/wc.in /tmp/wc.out /tmp/wc.err "/tmp/wt-$name".*.bin
    mkfifo /tmp/wc.in
    (
        exec 9<>/tmp/wc.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_VECTORS="/tmp/wt-$name" CLOVER_VECTORS_WEIGHTS=1
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/wc.in >/tmp/wc.out 2>/tmp/wc.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/wc.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/wc.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/wc.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-10s %s\n' "$name" "$(grep '^REQUEST' /tmp/wc.out | cat -v | head -1)"
    printf '%-10s bytes %s\n' "$name" "$(du -cb /tmp/wt-$name.*.bin 2>/dev/null | tail -1 | cut -f1)"
    rm -f /tmp/wc.in
}

echo "started $(date -u +%FT%TZ), $WANT tokens"
run france 1008 10484 318 15383 387
run japan  1008 10484 318 10417 387
run rain   91019 25528
run misc   500 1200 9000 33000 120000
echo "finished $(date -u +%FT%TZ)"
