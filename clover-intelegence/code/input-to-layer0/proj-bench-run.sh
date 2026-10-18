#!/bin/bash
# usage: proj-bench-run.sh   (builds and runs the projection kernel study on the real matrix shapes)
set -eu
H=/tmp/one-layer
cd "$H"
gcc -std=gnu11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    proj-bench.c -o proj-bench 2> build-pb.log || { grep -E "error" build-pb.log | head -20; exit 1; }
ulimit -c 0
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
for shape in "12288 7168" "4096 7168" "4096 1024" "7168 33792"; do ./proj-bench $shape 5; echo; done
