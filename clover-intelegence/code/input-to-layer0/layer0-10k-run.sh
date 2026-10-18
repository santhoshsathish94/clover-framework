#!/bin/bash
# usage: layer0-10k-run.sh   (needs real-ids.txt in /tmp/one-layer; runs 10,000 real tokens, reference = server_process itself)
set -eu
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
H=/tmp/one-layer
cd "$H"
gcc -std=gnu11 ${L0_OPT:--O2 -funroll-loops} -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I"$D/server" layer0-batch.c -lm -o layer0-batch 2> build-lb.log || { grep -E "error" build-lb.log | head -20; exit 1; }
ulimit -c 0
exec env -i PATH="$PATH" L0_WIDE=1 L0_VEC="${L0_VEC:-1}" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./layer0-batch "$D" real-ids.txt - 10000 32,1000,10000 1
