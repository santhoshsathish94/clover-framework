#!/bin/bash
# usage: layer0-run.sh [N] [SHOW]   (needs tok-1000.txt and l1in-1000.resid.f32 in /tmp/one-layer)
set -eu
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
H=/tmp/one-layer
cd "$H"
gcc -std=gnu11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I"$D/server" layer0-observe.c -lm -o layer0-observe 2> build-l0.log || { grep -E "error" build-l0.log | head -20; exit 1; }
ulimit -c 0
exec env -i PATH="$PATH" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./layer0-observe "$D" tok-1000.txt l1in-1000.resid.f32 "${1:-1000}" "${2:-12}" ${3:+"$3"}
