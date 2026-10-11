#!/usr/bin/env bash
# Two prompts with known-good outputs, run through the client-free fleet.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
rm -f /tmp/t.in /tmp/t.out /tmp/t.err /tmp/t.flag
mkfifo /tmp/t.in
(
    exec 9<>/tmp/t.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve \
        </tmp/t.in >/tmp/t.out 2>/tmp/t.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/t.out 2>/dev/null && break; sleep 1; done
printf '2 91019 25528\n2 1008 10484 318 15383 387\n' > /tmp/t.in
for _ in $(seq 1 900); do [ "$(grep -c '^REQUEST' /tmp/t.out 2>/dev/null)" -ge 2 ] && break; sleep 2; done
pkill -f "bin/pipeline-serve" 2>/dev/null
echo DONE > /tmp/t.flag
