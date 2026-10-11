#!/bin/bash
# The documented way to run this, which none of my measurements used.
#
# HOSTING.md: "Do not restart it between prompts - the whole point is that the
# weights stay mapped." Every sweep in this session started a fresh pipeline-serve
# per configuration, so every distributed figure recorded here is a cold start.
#
# One fleet, trunk locked, four threads as start-fleet.sh sets, then the same France
# prompt sent six times in sequence. The service prints its own per-request time.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/fleet-warm.log
rm -f /tmp/fleetwarm.done
: > "$LOG"
exec >>"$LOG" 2>&1

pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/clover.in /tmp/clover.out /tmp/clover.err
mkfifo /tmp/clover.in
(
    exec 9<>/tmp/clover.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=4 OMP_PROC_BIND=false
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/clover.in >/tmp/clover.out 2>/tmp/clover.err
) &

started=$(date +%s)
for _ in $(seq 1 600); do grep -q READY /tmp/clover.out 2>/dev/null && break; sleep 1; done
echo "fleet ready after $(( $(date +%s) - started ))s"
grep -m1 -i 'resident' /tmp/clover.out 2>/dev/null
echo

seen=0
for attempt in 1 2 3 4 5 6; do
    printf '1 1008 10484 318 15383 387\n' > /tmp/clover.in
    for _ in $(seq 1 900); do
        now=$(grep -c '^REQUEST' /tmp/clover.out 2>/dev/null || echo 0)
        [ "$now" -gt "$seen" ] && break
        sleep 1
    done
    seen=$(grep -c '^REQUEST' /tmp/clover.out 2>/dev/null || echo 0)
    printf 'request %d: %s\n' "$attempt" "$(grep '^REQUEST' /tmp/clover.out | tail -1 | cat -v)"
done

pkill -f '[b]in/pipeline-serve'; sleep 1
rm -f /tmp/clover.in
echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/fleetwarm.done
