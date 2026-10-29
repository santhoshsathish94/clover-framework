#!/bin/bash
# Builds and runs the M52 demonstration (real record size, sparse file) against the control (M00) and M52. No GPU, no CUDA; the sparse file is deleted afterwards.
cd "$(dirname "$0")"
gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I ../../../include -I ../../../../slotcache/include -o m52_real_size m52_real_size.c -ldl || exit 1
W=$HOME/rustmig/supply/verify/work
for so in M00 M52; do
  for e in 5 100 200 895; do
    echo "== $so (md5 $(md5sum $W/mut/$so/target/release/libsupply.so | cut -c1-8)) e=$e e*raw=$((e*17547264))"
    nice -n 10 ./m52_real_size $W/mut/$so/target/release/libsupply.so $W/m52.sparse $e 2>&1 | cut -c1-200
    echo "exit=${PIPESTATUS[0]}"
  done
done
rm -f $W/m52.sparse m52_real_size
df -h / | tail -1
