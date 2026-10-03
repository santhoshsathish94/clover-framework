#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
QKV=${1:?QKV path required}
TRUNK=${2:?prepared trunk-0 path required}
INDEX=${3:?original binary index required}
RAW_TRUNK=${4:?original raw trunk required}
for binding in \
    "4925ea619a4fd268b441e8008e4c1cd714f681b328b03cccd52c69b6a9b6f654:$QKV" \
    "dd58a012a1e795a959f14f2c8bf114921018e7cecd668bf505e966cdd4868722:$TRUNK/index.bin" \
    "475cd08da99c4f422f2c284906eb8a7ff0b16a5852bd77204cd210275cbcff66:$TRUNK/common.bin" \
    "79887e36c2e9d736e5446901d8059142b7b0f9f2d4aa8e1035446bda20390634:$TRUNK/kda.bin" \
    "70d210e1400dff4a59a583e0bfac3937a55beea93175e2931eacf2a7559ba2dd:$TRUNK/dense.bin"; do
    printf '%s  %s\n' "${binding%%:*}" "${binding#*:}" | sha256sum --check --status
done
gcc -std=c11 -O2 -ffp-contract=off -fno-fast-math -fopenmp \
    test-server-reference.c -lm -o bin/test-server-reference
for ids in 1008,10484,318,15383,387 1008,10484,318,10417,387; do
    env -i PATH="$PATH" K3_INDEX="$INDEX" K3_TRUNKPATH="$RAW_TRUNK" K3_IDS="$ids" \
        K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
        ./bin/test-server-reference "$QKV" "$TRUNK"
done