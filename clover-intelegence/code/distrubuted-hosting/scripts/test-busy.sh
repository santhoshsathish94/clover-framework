#!/bin/bash
# Fill the queue until it refuses. Distinct callbacks so the per-client cap is not
# what rejects, and a small budget so few workers drain it while it fills.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/busy-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="
pkill -f '[b]in/pipeline-serve'
rm -f /tmp/b.in /tmp/b.out /tmp/b.err
mkfifo /tmp/b.in
(
    exec 9<>/tmp/b.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=4 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1 CLOVER_MEMORY_BUDGET_MB=2000
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/b.in >/tmp/b.out 2>/tmp/b.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/b.out 2>/dev/null && break; sleep 1; done
echo "fleet ready, workers: $(grep -o 'so [0-9]* run at once' /tmp/b.err)"

# 4200 submissions, every one a different client and a different prompt.
seq 1 4200 | awk '{printf "SUBMIT http://127.0.0.1:9%04d/x 2 91019 %d\n", $1, 20000 + $1}' > /tmp/b.in
sleep 20

echo "--- counts ---"
printf 'ACCEPTED  %s\n' "$(grep -c '^ACCEPTED' /tmp/b.out)"
printf 'BUSY      %s\n' "$(grep -c '^BUSY' /tmp/b.out)"
printf 'DUPLICATE %s\n' "$(grep -c '^DUPLICATE' /tmp/b.out)"
printf 'TOO_MANY  %s\n' "$(grep -c '^TOO_MANY' /tmp/b.out)"
echo "queue depth is 4096, so ACCEPTED should stop there and the rest should be BUSY"
echo "--- first BUSY appears after this many ACCEPTED ---"
grep -n '^BUSY' /tmp/b.out | head -1
pkill -f '[b]in/pipeline-serve'
echo "=== finished $(date -u +%FT%TZ) ==="
