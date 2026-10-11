#!/bin/bash
# Capture the executed stage path for several prompts, one fleet per prompt so the
# traces cannot interleave. Recording only; nothing here decides what the data means.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/trace-capture.log
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    name=$1; shift
    ids="$*"
    echo "=== $name : $ids ==="
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/tr.in /tmp/tr.out /tmp/tr.err "/tmp/trace-$name.csv"
    mkfifo /tmp/tr.in
    (
        exec 9<>/tmp/tr.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_TRACE="/tmp/trace-$name.csv"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/tr.in >/tmp/tr.out 2>/tmp/tr.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/tr.out 2>/dev/null && break; sleep 1; done
    printf '2 %s\n' "$ids" > /tmp/tr.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/tr.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 2
    echo "output : $(grep '^REQUEST' /tmp/tr.out)"
    echo "rows   : $(wc -l < "/tmp/trace-$name.csv")"
    rm -f /tmp/tr.in
}

echo "started $(date -u +%FT%TZ)"
run france  1008 10484 318 15383 387
run japan   1008 10484 318 10417 387
run onthe   91019 25528
run alpha   500 1200 9000 33000 120000
run beta    70000 80 4096 150000 2
echo "finished $(date -u +%FT%TZ)"
ls -l /tmp/trace-*.csv
