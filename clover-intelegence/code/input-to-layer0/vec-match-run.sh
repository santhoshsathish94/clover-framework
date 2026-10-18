#!/bin/bash
# usage: vec-match-run.sh   (needs tok-1000.txt, l1in-1000.s0.f32, l1in-1000.resid.f32 in /tmp/one-layer)
set -eu
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
H=/tmp/one-layer
cd "$H"
gcc -std=gnu11 -O3 -march=native -fopenmp -Wall -Wno-unused-function -I"$D/server" vec-match.c -lm -o vec-match 2> build-vm.log || { grep -E "error" build-vm.log | head; exit 1; }
ulimit -c 0
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./vec-match "$D/server/bin/dataset/inputs/seed.bin" tok-1000.txt 1000 \
    "embedding(control)=l1in-1000.s0.f32" "attention-out=vm.attn.f32" "aggregate=vm.agg.f32" \
    "dense-out=vm.dense.f32" "layer0-output=l1in-1000.resid.f32"
