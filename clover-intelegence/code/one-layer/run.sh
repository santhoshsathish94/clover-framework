#!/bin/bash
# usage: run.sh IDS_FILE pin|cold|warm
set -eu
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
HERE=$(cd "$(dirname "$0")" && pwd)
[ "$HERE/one-layer" -nt "$HERE/one-layer.c" ] || gcc -std=gnu11 -O2 -march=native -ffp-contract=off -fno-fast-math \
    -fopenmp -Wall -Wno-unused-function -I"$CODE/server" "$HERE/one-layer.c" -lm -ldl -lpthread -o "$HERE/one-layer"
ulimit -l unlimited
ulimit -c 0
exec env -i PATH="$PATH" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$HERE/one-layer" "$CODE" "$1" "$2"
