#!/bin/bash
# usage: expert-kernel-run.sh [REPS]
set -u
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
ulimit -c 0
gcc -std=gnu11 -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I$D/common -I$D/tansformers/transformer-1 expert-kernel-bench.c -lm -o expert-kernel-bench 2> build-ek.log || { grep -E "error" build-ek.log | head -30; exit 1; }
grep -c warning build-ek.log | sed 's/^/build warnings: /'
env -i PATH="$PATH" EK_THREADS="${EK_THREADS:-16}" OMP_NUM_THREADS="${EK_THREADS:-16}" OMP_PROC_BIND=close OMP_PLACES=cores \
    ./expert-kernel-bench $D/tansformers/transformer-1/bin/dataset ${1:-6} 2>&1
