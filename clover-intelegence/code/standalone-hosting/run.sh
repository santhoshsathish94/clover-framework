#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
DATASET=${CLOVER_DATASET:-$HERE/bin/dataset}
CONFIG=${CLOVER_CONFIG:-$HERE/bin/configs/generation.json}
exec env -i PATH="$PATH" CLOVER_DATASET="$DATASET" CLOVER_CONFIG="$CONFIG" \
    CLOVER_SNAPFOLD="${CLOVER_SNAPFOLD:-}" \
    CLOVER_SNAPSHOT="${CLOVER_SNAPSHOT:-}" \
    CLOVER_FOLD_DUMP="${CLOVER_FOLD_DUMP:-}" \
    K3_RESIDENT_WARMUP="${K3_RESIDENT_WARMUP:-0}" \
    K3_PAYLOAD_RAM="${K3_PAYLOAD_RAM:-1}" \
    K3_EXPERT_DIRECT="${K3_EXPERT_DIRECT:-1}" \
    K3_RESULT_SET="${K3_RESULT_SET:-}" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$HERE/bin/clover-one" "$@"