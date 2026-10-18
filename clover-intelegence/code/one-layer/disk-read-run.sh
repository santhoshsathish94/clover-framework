#!/bin/bash
# usage: disk-read-run.sh   (read-only sweep of expert read speed from disk on the host; evicts the expert file from the page cache)
set -u
H=/tmp/one-layer
F=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers/transformer-1/bin/dataset/root-1/experts.direct
cd $H
gcc -std=gnu11 -O2 -fopenmp -Wall disk-read-bench.c -o disk-read-bench 2> build-dr.log || { grep error build-dr.log; exit 1; }
ls -la $F | awk '{print "file bytes:", $5}'
for E in 101 893; do
  echo "--- $E experts"
  ./disk-read-bench $F $E 16 mmap   17547264 2
  ./disk-read-bench $F $E 16 pread  17547264 2
  ./disk-read-bench $F $E 16 direct 17547264 2
  ./disk-read-bench $F $E 16 direct 4194304  2
  ./disk-read-bench $F $E 16 direct 1048576  2
  ./disk-read-bench $F $E 32 direct 1048576  2
  ./disk-read-bench $F $E 8  direct 4194304  2
  ./disk-read-bench $F $E 4  direct 4194304  2
  ./disk-read-bench $F $E 1  direct 4194304  2
done
