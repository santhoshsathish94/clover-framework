#!/usr/bin/env bash
# Runs the same four prompts against the resident fleet with the trunk locked and
# unlocked, so the trade between pinned weights and page cache is measured, not assumed.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
RESULT=/tmp/bench.log
: > "$RESULT"
rm -f /tmp/bench.flag

PROMPTS='2 91019 25528
2 29349 147043 623
2 1008 10484 318 15383 387
2 16 11 220 17 11 220 18 11'

run_config() {
    local label=$1 lock=$2
    pkill -f "bin/pipeline-serve" 2>/dev/null
    sleep 3
    rm -f /tmp/clover.in /tmp/clover.out /tmp/clover.err
    mkfifo /tmp/clover.in
    (
        exec 9<>/tmp/clover.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=4 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK="$lock"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve \
            </tmp/clover.in >/tmp/clover.out 2>/tmp/clover.err
    ) &
    # wait for READY
    for _ in $(seq 1 120); do grep -q READY /tmp/clover.out 2>/dev/null && break; sleep 1; done

    local open_line swap_before
    open_line=$(grep -m1 "Fleet resident" /tmp/clover.err)
    swap_before=$(free -m | awk '/^Swap:/{print $3}')

    local started
    started=$(date +%s.%N)
    printf '%s\n' "$PROMPTS" > /tmp/clover.in
    for _ in $(seq 1 900); do grep -q "^END" /tmp/clover.out 2>/dev/null && break; sleep 1; done
    local finished
    finished=$(date +%s.%N)

    {
        echo "===== $label (CLOVER_LOCK_TRUNK=$lock) ====="
        echo "$open_line"
        grep "^REQUEST" /tmp/clover.out
        echo "batch wall: $(echo "$finished - $started" | bc) s"
        free -m | awk -v b="$swap_before" '/^Mem:/{printf "mem used=%sM cache=%sM avail=%sM\n",$3,$6,$7}
                                           /^Swap:/{printf "swap before=%sM after=%sM\n",b,$3}'
        echo
    } >> "$RESULT"
}

run_config "LOCKED" 1
run_config "UNLOCKED" 0
pkill -f "bin/pipeline-serve" 2>/dev/null
echo DONE > /tmp/bench.flag
