#!/bin/bash
# Timing matrix. Aborts without timing if anything foreign is on the box. Log: out/timing.log
cd "$(dirname "$0")"
L=out/timing.log; : > $L
export PATH=/usr/local/cuda-12.6/bin:$PATH
foreign() {
  local a b
  a=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -v '^$')
  b=$(ps aux | grep -E "model-gpu|k3gpu|g3run|nvcc|cargo|diff_driver" | grep -v grep | grep -v "diff_driver ../" | grep -v timing.sh)
  echo "$a$b"
}
{
echo "=== idle check $(date +%T)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
echo "--- ps (model-gpu|k3gpu|g3run|nvcc|cargo)"; ps aux | grep -E "model-gpu|k3gpu|g3run|nvcc|cargo" | grep -v grep | cut -c1-160
echo "--- top CPU"; ps -eo pid,pcpu,comm --sort=-pcpu | head -5
} >> $L 2>&1
F=$(foreign)
if [ -n "$F" ]; then echo "ABORT-FOREIGN: $F" >> $L; echo DONE >> $L; exit 0; fi
cat data/rec8.bin > /dev/null   # page cache warm for T2/T3
run() { # tag lib T [prefix]
  local pre="$4"; local f=$(foreign); [ -n "$f" ] && echo "FOREIGN-DURING before $1 $3: $f" >> $L
  $pre ./diff_driver $2 $3 --tag "$1$5" 2>>out/timing.err | grep TIMING >> $L
}
for T in T1 T2 T3; do
  for p in 1 2 3 4 5; do
    run cpp ../ref/libcopyq_ref.so $T
    run rust ../target/release/libcopyq.so $T
  done
done
for p in 1 2 3; do
  run cpp ../ref/libcopyq_ref.so T1 "numactl --cpunodebind=0 --membind=0" "-numa0"
  run rust ../target/release/libcopyq.so T1 "numactl --cpunodebind=0 --membind=0" "-numa0"
done
{ echo "=== end check $(date +%T)"; uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; } >> $L 2>&1
echo DONE >> $L
