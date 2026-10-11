#!/usr/bin/env bash
# Same shape as the earlier single-slot sample so the numbers are comparable:
# one request, OMP_NUM_THREADS from the caller, per-thread CPU over a real window.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
THREADS=${1:-4}
BIND=${2:-false}
rm -f /tmp/p.in /tmp/p.out /tmp/p.err /tmp/p.flag
mkfifo /tmp/p.in
(
    exec 9<>/tmp/p.in
    ulimit -l unlimited
    export OMP_NUM_THREADS="$THREADS" CLOVER_LOCK_TRUNK=1 CLOVER_MEMORY_BUDGET_MB=1
    if [ "$BIND" = close ]; then export OMP_PROC_BIND=close OMP_PLACES=cores; else export OMP_PROC_BIND=false; fi
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/p.in >/tmp/p.out 2>/tmp/p.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/p.out 2>/dev/null && break; sleep 1; done
printf '2 91019 25528\n' > /tmp/p.in
sleep 40
pid=$(pgrep -f "bin/pipeline-serve" | head -1)
snap() { for t in /proc/$pid/task/*; do awk -v id="$(basename "$t")" '{print id, $14+$15}' "$t/stat" 2>/dev/null; done; }
snap > /tmp/p1; sleep 10; snap > /tmp/p2
{
    echo "threads=$THREADS bind=$BIND"
    echo "per-thread busy over 10s (100% = one full core):"
    join /tmp/p1 /tmp/p2 | awk '{printf "  tid=%s %5.1f%%\n", $1, ($3-$2)/10}' | sort -k2 -rn
    join /tmp/p1 /tmp/p2 | awk '{s+=$3-$2} END {printf "  total %.0f%% of 3200%%\n", s/10}'
} > /tmp/p.sample
for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/p.out 2>/dev/null && break; sleep 2; done
cat /tmp/p.sample
grep '^REQUEST' /tmp/p.out
pkill -f "bin/pipeline-serve" 2>/dev/null
echo DONE > /tmp/p.flag
