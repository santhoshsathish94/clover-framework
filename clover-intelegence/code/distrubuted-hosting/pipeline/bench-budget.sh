#!/usr/bin/env bash
# Concurrency is no longer configured, so it cannot be swept directly: vary the memory
# budget and let the coordinator decide how many fit. It reports the figure it derived,
# which is recorded here alongside the throughput it produced.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
RESULT=/tmp/budget.log
: > "$RESULT"; rm -f /tmp/budget.flag

BASE='2 91019 25528
2 29349 147043 623
2 1008 10484 318 15383 387
2 16 11 220 17 11 220 18 11
2 3575 374 279 6864 315 9822
2 22649 374 264 4221 369
2 791 7160 374
2 17 10 17 284'
PROMPTS=$(for _ in 1 2 3 4; do printf '%s\n' "$BASE"; done)
COUNT=$(printf '%s\n' "$PROMPTS" | wc -l)

run_at() {
    local megabytes=$1
    pkill -f "bin/pipeline-serve" 2>/dev/null; sleep 3
    rm -f /tmp/b.in /tmp/b.out /tmp/b.err; mkfifo /tmp/b.in
    (
        exec 9<>/tmp/b.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=4 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1 CLOVER_MEMORY_BUDGET_MB="$megabytes"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/b.in >/tmp/b.out 2>/tmp/b.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/b.out 2>/dev/null && break; sleep 1; done
    local derived started finished pid peak=0
    derived=$(sed -n 's/.*so \([0-9]*\) run at once.*/\1/p' /tmp/b.err | head -1)
    started=$(date +%s.%N)
    printf '%s\n' "$PROMPTS" > /tmp/b.in
    pid=$(pgrep -f "bin/pipeline-serve" | head -1)
    while [ "$(grep -c '^REQUEST' /tmp/b.out 2>/dev/null)" -lt "$COUNT" ]; do
        local now; now=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
        [ -n "$now" ] && [ "${now%.*}" -gt "${peak%.*}" ] && peak=$now
        sleep 5
    done
    finished=$(date +%s.%N)
    local wall; wall=$(echo "$finished - $started" | bc)
    printf 'budget=%-7s concurrent=%-4s wall=%9.2f s  throughput=%s prompts/s  peak_cpu=%s%%\n' \
        "${megabytes}MB" "${derived:-?}" "$wall" "$(echo "scale=4; $COUNT / $wall" | bc)" "$peak" >> "$RESULT"
}

for megabytes in 1024 3072 6144 12288 24576; do run_at "$megabytes"; done
pkill -f "bin/pipeline-serve" 2>/dev/null
echo DONE > /tmp/budget.flag
