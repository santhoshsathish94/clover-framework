#!/bin/bash
# usage: expert-lut2-run.sh [REPS]   (needs expert-kernel-bench.c, expert-lut-bench.c, expert-lut2-bench.c in /tmp/one-layer)
set -u
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
ulimit -c 0
gcc -std=gnu11 -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I$D/common -I$D/tansformers/transformer-1 expert-lut2-bench.c -lm -o expert-lut2-bench 2> build-lut2.log || { grep -E "error" build-lut2.log | head -30; exit 1; }
grep -c warning build-lut2.log | sed 's/^/build warnings: /'
env -i PATH="$PATH" EK_THREADS="${EK_THREADS:-16}" OMP_NUM_THREADS="${EK_THREADS:-16}" OMP_PROC_BIND=close OMP_PLACES=cores \
    ./expert-lut2-bench $D/tansformers/transformer-1/bin/dataset ${1:-5} 2>&1
