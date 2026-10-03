#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
selection=${1:?all or comma-separated layers}
root=/opt/clover-k3/clover-intelegence
packages="$root/code/tansformers"
gcc -std=c11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-reference.c -lm -ldl -o test-reference
for prompt in france japan; do
    ids=1008,10484,318,15383,387
    if [[ "$prompt" == japan ]]; then ids=1008,10484,318,10417,387; fi
    env -i PATH="$PATH" K3_INDEX="$root/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS="$ids" K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        ./test-reference "$root/dataset" "$packages" "$prompt" "$selection"
done