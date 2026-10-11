#!/bin/bash
# Starts the resident fleet with the trunk locked in RAM and keeps it fed by a FIFO.
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
pkill -f 'bin/pipeline-serve' 2>/dev/null
pkill -f 'exec 9>/tmp/clover.in' 2>/dev/null
sleep 1
rm -f /tmp/clover.in /tmp/clover.out /tmp/clover.err
mkfifo /tmp/clover.in
# read-write keeps the FIFO open without blocking, so prompts can arrive at any time
exec 9<>/tmp/clover.in
ulimit -l unlimited
# Four OpenMP threads per request: measured as the knee, 16 buys only 4% more.
# How many requests run at once is decided from free memory, not configured here.
export OMP_NUM_THREADS=4 OMP_PROC_BIND=false
cd "$CODE/pipeline" || exit 1
exec ./bin/pipeline-serve "$CODE" serve </tmp/clover.in >/tmp/clover.out 2>/tmp/clover.err
