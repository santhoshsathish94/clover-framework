#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
DATASET=${1:?dataset directory required}
BASE=/opt/clover-k3/clover-intelegence
for binding in \
    'e1b6e44fa29154c0f8d3da8525657b6c997c35e66c8123aff4a229fb598fc548:france/inputs.f32' \
    '6829a694eed5682e3a049ea1446857e8ead9ce4b2e581a1d76c9f686d2e46b40:france/results.bin' \
    '221eb09509df37d41d5c6f47b6e88232bc34251188c150a2892ac4965f815b30:japan/inputs.f32' \
    'ff107fa98be44137d11745fdb2f18ee981f00dc38760bfc36967393d1ba9c740:japan/results.bin'; do
    printf '%s  %s\n' "${binding%%:*}" "$DATASET/root-1/observations/${binding#*:}" | sha256sum --check --status
done
gcc -std=c11 -O2 -ffp-contract=off -fno-fast-math -fopenmp test-reference.c -lm -o bin/test-reference
for prompt in france japan; do
    ids=1008,10484,318,15383,387
    if [[ "$prompt" == japan ]]; then ids=1008,10484,318,10417,387; fi
    env -i PATH="$PATH" K3_INDEX="$BASE/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS="$ids" K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        ./bin/test-reference "$DATASET" "$prompt"
done