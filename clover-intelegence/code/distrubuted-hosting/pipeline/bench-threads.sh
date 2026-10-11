#!/usr/bin/env bash
# Thread count is the only variable: same binary, same prompt, back to back on one
# resident fleet, so page-cache state cannot be mistaken for a parallelism win.
# OMP_NUM_THREADS=1 is the control; if 4 and 16 do not beat it, the parallel regions
# are not contributing.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
RESULT=/tmp/threads.log
: > "$RESULT"; rm -f /tmp/threads.flag

run_at() {
    local threads=$1 bind=$2
    pkill -f "bin/pipeline-serve" 2>/dev/null; sleep 3
    rm -f /tmp/n.in /tmp/n.out /tmp/n.err; mkfifo /tmp/n.in
    (
        exec 9<>/tmp/n.in
        ulimit -l unlimited
        export OMP_NUM_THREADS="$threads" CLOVER_LOCK_TRUNK=1 CLOVER_MEMORY_BUDGET_MB=1
        if [ "$bind" = close ]; then export OMP_PROC_BIND=close OMP_PLACES=cores; else export OMP_PROC_BIND=false; fi
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/n.in >/tmp/n.out 2>/tmp/n.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/n.out 2>/dev/null && break; sleep 1; done
    # First request warms the cache for this configuration and is discarded.
    printf '2 91019 25528\n' > /tmp/n.in
    for _ in $(seq 1 900); do [ "$(grep -c '^REQUEST' /tmp/n.out)" -ge 1 ] && break; sleep 2; done
    printf '2 91019 25528\n' > /tmp/n.in
    for _ in $(seq 1 900); do [ "$(grep -c '^REQUEST' /tmp/n.out)" -ge 2 ] && break; sleep 2; done
    local pid; pid=$(pgrep -f "bin/pipeline-serve" | head -1)
    {
        printf '%-18s warm=%s  cold=%s  peak_cpu=%s%%\n' "threads=$threads bind=$bind" \
            "$(grep '^REQUEST' /tmp/n.out | sed -n 2p | awk '{print $4}')" \
            "$(grep '^REQUEST' /tmp/n.out | sed -n 1p | awk '{print $4}')" \
            "$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')"
    } >> "$RESULT"
}

run_at 1 false
run_at 4 false
run_at 16 false
run_at 16 close
pkill -f "bin/pipeline-serve" 2>/dev/null
echo DONE > /tmp/threads.flag
