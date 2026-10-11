#!/bin/bash
# "the capital of france is", run twice through one resident fleet.
#
# The first pass pulls whatever experts it needs off the NVMe. The second pass wants
# the same experts, which are now in page cache. The gap between the two is the part
# of the request that is waiting for storage rather than computing, which is exactly
# the part a pod holding its own experts in RAM would not pay.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/france-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="
pkill -f '[b]in/pipeline-serve'; pkill -f '[r]eceiver.py'
rm -f /tmp/f.in /tmp/f.out /tmp/f.err
mkfifo /tmp/f.in

(
    exec 9<>/tmp/f.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1 CLOVER_TIMING=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/f.in >/tmp/f.out 2>/tmp/f.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/f.out 2>/dev/null && break; sleep 1; done
echo "fleet ready"
echo "cache before: $(grep '^Cached:' /proc/meminfo)"

exec 8>/tmp/f.in
# One at a time: a second request running concurrently would share cores and cache.
printf '2 1008 10484 318 15383 387\n' >&8
for _ in $(seq 1 900); do [ "$(grep -c '^REQUEST' /tmp/f.out 2>/dev/null)" -ge 1 ] && break; sleep 2; done
echo "after pass 1: $(grep '^Cached:' /proc/meminfo)"

printf '2 1008 10484 318 15383 387\n' >&8
for _ in $(seq 1 900); do [ "$(grep -c '^REQUEST' /tmp/f.out 2>/dev/null)" -ge 2 ] && break; sleep 2; done
exec 8>&-
sleep 2

echo "--- stdout ---"
cat /tmp/f.out
echo "--- io ---"
grep '^IO' /tmp/f.err
echo "--- totals ---"
grep 'request-total\|,server,\|93,head\|93,normalize\|93,deliver\|-,queue' /tmp/f.err
echo "--- layer sums per pass ---"
awk -F, '/^TIMING/ {split($1,a," "); r=a[2]; if (a[3]+0>=1 && a[3]+0<=92) s[r]+=$4} END {for (r in s) printf "request %s layers total %.3f s\n", r, s[r]}' /tmp/f.err
pkill -f '[b]in/pipeline-serve'
echo "=== finished $(date -u +%FT%TZ) ==="
