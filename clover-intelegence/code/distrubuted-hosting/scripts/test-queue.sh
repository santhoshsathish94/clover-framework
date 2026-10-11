#!/bin/bash
# The admission answers, not the answers themselves. Submissions are instant, so this
# only needs the fleet up; it never waits for a request to finish.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/queue-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="
pkill -f '[b]in/pipeline-serve'
rm -f /tmp/q.in /tmp/q.out /tmp/q.err
mkfifo /tmp/q.in
(
    exec 9<>/tmp/q.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/q.in >/tmp/q.out 2>/tmp/q.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/q.out 2>/dev/null && break; sleep 1; done
echo "fleet ready"

exec 8>/tmp/q.in

echo "--- same submission twice, same callback ---"
printf 'SUBMIT http://127.0.0.1:8099/a 2 91019 25528\n' >&8
sleep 1
printf 'SUBMIT http://127.0.0.1:8099/a 2 91019 25528\n' >&8
sleep 1

echo "--- nine more distinct prompts, same callback (cap is 8) ---"
for n in 1 2 3 4 5 6 7 8 9; do
    printf 'SUBMIT http://127.0.0.1:8099/a 2 91019 %d\n' "$((25528 + n))" >&8
    sleep 0.4
done
sleep 2

echo "--- a different callback is a different client ---"
printf 'SUBMIT http://127.0.0.1:8099/b 2 91019 25528\n' >&8
sleep 2

exec 8>&-
echo "--- coordinator stdout ---"
cat /tmp/q.out
echo "--- counts ---"
printf 'ACCEPTED  %s\n' "$(grep -c '^ACCEPTED' /tmp/q.out)"
printf 'DUPLICATE %s\n' "$(grep -c '^DUPLICATE' /tmp/q.out)"
printf 'TOO_MANY  %s\n' "$(grep -c '^TOO_MANY' /tmp/q.out)"
echo "expect: 1 DUPLICATE, 8 ACCEPTED for client a then TOO_MANY, and client b accepted"
pkill -f '[b]in/pipeline-serve'
echo "=== finished $(date -u +%FT%TZ) ==="
