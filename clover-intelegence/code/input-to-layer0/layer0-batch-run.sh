#!/bin/bash
# usage: layer0-batch-run.sh N W1,W2,.. DEDUP [THREADS]   (needs tok-1000.txt and l1in-1000.resid.f32 in /tmp/one-layer)
set -eu
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
H=/tmp/one-layer
cd "$H"
gcc -std=gnu11 ${L0_OPT:--O2 -funroll-loops} -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I"$D/server" layer0-batch.c -lm -o layer0-batch 2> build-lb.log || { grep -E "error" build-lb.log | head -20; exit 1; }
ulimit -c 0
exec env -i PATH="$PATH" L0_WIDE="${L0_WIDE:-1}" L0_VEC="${L0_VEC:-1}" L0_PF="${L0_PF:-0}" OMP_NUM_THREADS="${4:-16}" OMP_PROC_BIND=close OMP_PLACES=cores \
    ./layer0-batch "$D" tok-1000.txt l1in-1000.resid.f32 "${1:-1000}" "${2:-8,32,128,1000}" "${3:-1}"
