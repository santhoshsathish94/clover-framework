#!/usr/bin/env bash
# Why do 64 runnable threads produce only 22 cores of work?
#
# WHY THIS EXISTS
# Four lanes hold 65 threads for the whole run, spread across all 32 CPUs (checked:
# pinning and unpinning both use 32 distinct CPUs and finish within 0.04 s of each
# other), yet only ~22 cores' worth of work gets done and there are 4.6 M voluntary
# context switches. Voluntary means the thread chose to sleep. This samples what they
# are sleeping ON.
#
# /proc/PID/task/TID/wchan names the kernel function a sleeping thread is blocked in,
# and is empty or "0" for a thread that is currently running. So the histogram below
# splits into "running" plus a breakdown of every distinct reason a thread was parked.
# No profiler, no kernel config, no package install -- this works on a stock box and
# touches nothing.
#
# WHAT IT CANNOT DO: wchan only shows BLOCKED threads. It says nothing about where
# on-CPU kernel time goes, which is the separate and still-open question about ~435 s
# of system time. Do not read this as a system-time profile.
set -u

C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LANES=${1:-4}
SAMPLES=${2:-400}
THREADS=${3:-16}

cd "$C/pipeline" || exit 1
: > /tmp/wchan-in.txt
i=0; while [ $i -lt "$LANES" ]; do echo "1 1008 10484 318 15383 387" >> /tmp/wchan-in.txt; i=$((i+1)); done

env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
    ./bin/pipeline-serve "$C" serve < /tmp/wchan-in.txt > /tmp/wchan.out 2>&1 &
PID=$!

: > /tmp/wchan.raw
n=0
while kill -0 "$PID" 2>/dev/null && [ "$n" -lt "$SAMPLES" ]; do
    for f in /proc/$PID/task/*/wchan; do
        w=$(cat "$f" 2>/dev/null)
        case "${w:-}" in
            ""|0) echo "RUNNING-or-on-cpu" ;;
            *) echo "$w" ;;
        esac
    done >> /tmp/wchan.raw 2>/dev/null
    n=$((n + 1))
    sleep 0.1
done
wait "$PID"

if ! grep -q '^REQUEST' /tmp/wchan.out; then
    echo "*** FAILED, no REQUEST line: $(tail -2 /tmp/wchan.out)"
    exit 1
fi
grep -E '^REQUEST' /tmp/wchan.out

total=$(wc -l < /tmp/wchan.raw)
echo
echo "thread-samples: $total over $n sweeps at ~0.1 s, $LANES lanes, $THREADS threads per team"
echo
printf "%8s %7s  %s\n" "count" "share" "blocked in"
sort /tmp/wchan.raw | uniq -c | sort -rn | head -20 | \
    awk -v t="$total" '{printf "%8d %6.2f%%  %s\n", $1, 100*$1/t, $2}'
echo
echo "RUNNING-or-on-cpu is the share actually executing. Everything else is a thread"
echo "parked in the kernel, and the top entry among those is what is costing the"
echo "difference between 64 threads and 22 cores of work."
