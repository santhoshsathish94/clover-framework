#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
ROOT=/opt/clover-k3/clover-intelegence
LEAVES=${1:?leaves path required}
printf '%s  %s\n' a6a886747d400a545f4e16c2831ef0db7e8eab90575184a54d3bffe64f5b5f2a "$LEAVES" | sha256sum --check --status
gcc -std=c11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-reference.c -lm -o bin/test-reference
for prompt in france japan; do
    ids=1008,10484,318,15383,387
    if [[ "$prompt" == japan ]]; then ids=1008,10484,318,10417,387; fi
    env -i PATH="$PATH" K3_INDEX="$ROOT/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS="$ids" K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        ./bin/test-reference "$ROOT/dataset" "$prompt" "$LEAVES"
done