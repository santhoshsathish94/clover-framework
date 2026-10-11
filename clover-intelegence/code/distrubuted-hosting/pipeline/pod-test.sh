#!/usr/bin/env bash
# Imposes the proposed pod spec on one layer: 4 CPUs, 16 GiB, no swap, cold cache.
# The question is whether a layer's 15.512 GiB working set stays resident in 16 GiB
# while it serves, because without local NVMe a miss refaults over the network.
set -uo pipefail
LAYER=${1:-46}
LIMIT=${2:-16G}
CG=/sys/fs/cgroup/podtest
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
RESULT=/tmp/podtest.log
: > "$RESULT"; rm -f /tmp/podtest.flag

grep -q memory /sys/fs/cgroup/cgroup.subtree_control || echo "+memory" > /sys/fs/cgroup/cgroup.subtree_control
grep -qw cpu /sys/fs/cgroup/cgroup.subtree_control || echo "+cpu" > /sys/fs/cgroup/cgroup.subtree_control
rmdir "$CG" 2>/dev/null
mkdir -p "$CG"
echo "$LIMIT" > "$CG/memory.max"
echo 0        > "$CG/memory.swap.max"
# cpuset is not in the root subtree_control here; a 400% quota is the same 4 vCPU.
echo "400000 100000" > "$CG/cpu.max"

sync; echo 3 > /proc/sys/vm/drop_caches; sleep 2
{
  echo "limit=$LIMIT cpu.max=$(cat $CG/cpu.max) swap=$(cat $CG/memory.swap.max)"
  echo "cold: $(free -g | awk '/^Mem:/{print "used="$3"G cache="$6"G"}')"
} >> "$RESULT"

( echo $BASHPID > "$CG/cgroup.procs"
  cd "$CODE/tansformers/transformer-$LAYER" || exit 1
  exec env OMP_NUM_THREADS=4 /tmp/test-layer-$LAYER bin/dataset "$LAYER" /tmp/layers.bin
) > /tmp/podtest.out 2>&1 &
child=$!

peak=0; started=$(date +%s.%N)
while kill -0 $child 2>/dev/null; do
    cur=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
    [ "$cur" -gt "$peak" ] && peak=$cur
    sleep 1
done
wait $child; status=$?
finished=$(date +%s.%N)

{
  echo "exit=$status  wall=$(echo "$finished - $started" | bc) s"
  echo "peak cgroup memory = $(echo "$peak" | awk '{printf "%.3f GiB", $1/1073741824}')"
  echo "oom kills = $(awk '/oom_kill /{print $2}' "$CG/memory.events" 2>/dev/null)"
  echo "--- output ---"; cat /tmp/podtest.out
} >> "$RESULT"
rmdir "$CG" 2>/dev/null
echo DONE > /tmp/podtest.flag
