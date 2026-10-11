#!/bin/bash
# CLOVER_FOLD=residual must collapse every fold record to a single weight of 1.
# If it does, the switches reach the deployed layers and hardmax alone is at fault.
# If it does not, nothing is reaching them and the timestamps are lying.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/switch-probe.log
: > "$LOG"
exec >>"$LOG" 2>&1

pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/sp.in /tmp/sp.out /tmp/sp.err /tmp/probe-fold.*.bin
mkfifo /tmp/sp.in
(
    exec 9<>/tmp/sp.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    export CLOVER_VECTORS=/tmp/probe-fold CLOVER_VECTORS_KEEP=weights CLOVER_FOLD=residual
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/sp.in >/tmp/sp.out 2>/tmp/sp.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/sp.out 2>/dev/null && break; sleep 1; done
printf '1 1008 10484 318 15383 387\n' > /tmp/sp.in
for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/sp.out 2>/dev/null && break; sleep 2; done
sleep 2
pkill -f '[b]in/pipeline-serve'; sleep 1
grep '^REQUEST' /tmp/sp.out | cat -v | head -1
rm -f /tmp/sp.in

python3 - <<'PY'
import glob, struct
counts, ones = {}, 0
total = 0
for path in glob.glob('/tmp/probe-fold.*.bin'):
    blob = open(path, 'rb').read(); at = 0
    while at + 16 <= len(blob):
        o, l, p, c = struct.unpack_from('<4i', blob, at); at += 16
        if c <= 0 or at + c * 4 > len(blob): break
        if 2000 <= l < 3000 and p == 0:
            values = struct.unpack_from('<%df' % c, blob, at)
            counts[c] = counts.get(c, 0) + 1
            total += 1
            if c == 1 and values[0] == 1.0: ones += 1
        at += c * 4
print('fold records at position 0 :', total)
print('records with a single 1.0  :', ones)
print('source counts seen         :', dict(sorted(counts.items())))
PY
