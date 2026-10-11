#!/bin/bash
# Read the environment of the running coordinator rather than the shell that
# started it, because that is where transformer_hardmax does its getenv.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/env-probe.log
: > "$LOG"
exec >>"$LOG" 2>&1

pkill -f '[b]in/pipeline-serve'; sleep 2
rm -f /tmp/ep.in /tmp/ep.out /tmp/ep.err
mkfifo /tmp/ep.in
(
    exec 9<>/tmp/ep.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/ep.in >/tmp/ep.out 2>/tmp/ep.err
) &
sleep 20
pid=$(pgrep -f '[b]in/pipeline-serve' | head -1)
echo "coordinator pid: ${pid:-none}"
if [ -n "${pid:-}" ]; then
    echo "--- every CLOVER_ variable the process can see ---"
    tr '\0' '\n' < "/proc/$pid/environ" | grep CLOVER_ || echo "(no CLOVER_ variables at all)"
    echo "--- is CLOVER_SOFTMAX present? ---"
    tr '\0' '\n' < "/proc/$pid/environ" | grep -c '^CLOVER_SOFTMAX='
    echo "--- which objects are mapped ---"
    tr '\0' '\n' < "/proc/$pid/maps" 2>/dev/null | grep -o '/[^ ]*transformer-46[^ ]*' | sort -u
    grep -o '/[^ ]*transformer-4[0-9][^ ]*\.so' "/proc/$pid/maps" | sort -u | head -5
fi
pkill -f '[b]in/pipeline-serve'; sleep 1
rm -f /tmp/ep.in
echo done
