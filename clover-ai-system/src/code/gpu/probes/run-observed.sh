#!/bin/bash
# Runs COMMAND detached, with an observer beside it, and returns at once. The job and the observer survive the ssh session.
#
#   run-observed.sh NAME COMMAND [ARGS...]
#
# Files in ~/runs/NAME:  job.log (the command's output), gpu.csv (every card's utilisation and memory each second), sys.log (vmstat each second: CPU, blocks read),
#                        done (appears when the job has ended, holding its exit code)
# Summary of a finished or running job: run-observed.sh --summary NAME
set -u
if [ "${1:-}" = "--summary" ]; then
    D=$HOME/runs/$2
    [ -f "$D/done" ] && echo "job ended, exit code $(cat "$D/done")" || echo "job still running (or never started)"
    echo "per card: mean and highest GPU utilisation over the samples where any card was above 0%, highest memory used"
    awk -F', ' '{ g=$2; u=$3+0; m=$4+0; t=$1; if (!(t in any)) any[t]=0; if (u>0) any[t]=1; U[t","g]=u; M[g]=(m>M[g]?m:M[g]); G[g]=1; T[t]=1 }
        END { for (t in T) if (any[t]) { n++; for (g in G) { s[g]+=U[t","g]; if (U[t","g]>h[g]) h[g]=U[t","g] } }
              for (g=0; g<8; g++) if (g in G) printf "card %d: mean %.0f%%, highest %.0f%%, memory %d MiB\n", g, n ? s[g]/n : 0, h[g], M[g]; printf "%d one-second samples with some card busy\n", n }' "$D/gpu.csv"
    tail -3 "$D/job.log"
    exit 0
fi
NAME=$1; shift
D=$HOME/runs/$NAME; mkdir -p "$D"; rm -f "$D/done" "$D/job.log" "$D/gpu.csv" "$D/sys.log"
setsid nohup bash -c '
    D=$1; shift
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$D/gpu.csv" 2>&1 & O1=$!
    vmstat -t 1 > "$D/sys.log" 2>&1 & O2=$!
    "$@" > "$D/job.log" 2>&1
    echo $? > "$D/done"
    kill $O1 $O2 2>/dev/null
' _ "$D" "$@" > /dev/null 2>&1 < /dev/null &
echo "started $NAME; files in $D"
