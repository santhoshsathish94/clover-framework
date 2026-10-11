#!/bin/bash
# Capture the distributed fold weights with the build that is on the box right now,
# so the comparison against the standalone is current against current rather than
# current against a file written hours ago.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/now-capture.log
: > "$LOG"
exec >>"$LOG" 2>&1

grab() {
    name=$1; shift
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/nc.in /tmp/nc.out /tmp/nc.err "/tmp/now-$name".*.bin
    mkfifo /tmp/nc.in
    (
        exec 9<>/tmp/nc.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_VECTORS="/tmp/now-$name" CLOVER_VECTORS_KEEP=weights
        "$@"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/nc.in >/tmp/nc.out 2>/tmp/nc.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/nc.out 2>/dev/null && break; sleep 1; done
    printf '1 1008 10484 318 15383 387\n' > /tmp/nc.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/nc.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-10s %s\n' "$name" "$(grep '^REQUEST' /tmp/nc.out | cat -v | head -1)"
    rm -f /tmp/nc.in
}

echo "started $(date -u +%FT%TZ)"
grab hard true
grab soft export CLOVER_SOFTMAX=1
echo "finished $(date -u +%FT%TZ)"
