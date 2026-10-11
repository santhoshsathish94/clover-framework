#!/bin/bash
# The same all-owner vector dump for several different prompts, one fleet per prompt
# so the per-owner files cannot mix. One position each: the question is whether the
# shape of the across-owner sequence recurs, not how it evolves with position.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/multi-capture.log
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    name=$1; token=$2
    echo "=== $name : token $token ==="
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/m.in /tmp/m.out /tmp/m.err "/tmp/vp-$name".*.bin
    mkfifo /tmp/m.in
    (
        exec 9<>/tmp/m.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_VECTORS="/tmp/vp-$name"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/m.in >/tmp/m.out 2>/tmp/m.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/m.out 2>/dev/null && break; sleep 1; done
    printf '1 %s\n' "$token" > /tmp/m.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/m.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    echo "  $(grep '^REQUEST' /tmp/m.out)"
    echo "  files $(ls /tmp/vp-$name.*.bin 2>/dev/null | wc -l)"
    rm -f /tmp/m.in
}

echo "started $(date -u +%FT%TZ)"
run p1 91019
run p2 1008
run p3 15383
run p4 70000
run p5 500
echo "finished $(date -u +%FT%TZ)"
