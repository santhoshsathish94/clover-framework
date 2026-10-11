#!/bin/bash
# Full vectors from every owner, one position. 93 owners x 109 records x 28,672 bytes
# is about 290 MB, which is why this is one position and not six.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/a.in /tmp/a.out /tmp/a.err /tmp/vec-all.bin
mkfifo /tmp/a.in
(
    exec 9<>/tmp/a.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    export CLOVER_VECTORS=/tmp/vec-all.bin
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/a.in >/tmp/a.out 2>/tmp/a.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/a.out 2>/dev/null && break; sleep 1; done
echo "fleet ready"
printf '1 91019\n' > /tmp/a.in
for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/a.out 2>/dev/null && break; sleep 2; done
sleep 2
pkill -f '[b]in/pipeline-serve'
cat /tmp/a.out
ls -l /tmp/vec-all.bin
echo "CAPTURE COMPLETE"
