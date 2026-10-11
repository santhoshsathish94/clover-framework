#!/usr/bin/env bash
# Brings up one resident fleet and leaves it running, so observation steps can be sent
# to it one at a time without ever restarting it.
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
pkill -f "bin/pipeline-serve" 2>/dev/null
sleep 3
rm -f /tmp/obs.in /tmp/obs.out /tmp/obs.err
mkfifo /tmp/obs.in
exec 9<>/tmp/obs.in
ulimit -l unlimited
export OMP_NUM_THREADS=4 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
cd "$CODE/pipeline" || exit 1
exec ./bin/pipeline-serve "$CODE" serve </tmp/obs.in >/tmp/obs.out 2>/tmp/obs.err
