#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
DATASET=${CLOVER_DATASET:-$HERE/bin/dataset}
CONFIG=${CLOVER_CONFIG:-$HERE/bin/configs/generation.json}
exec env -i PATH="$PATH" CLOVER_DATASET="$DATASET" CLOVER_CONFIG="$CONFIG" \
    K3_RESIDENT_WARMUP="${K3_RESIDENT_WARMUP:-0}" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$HERE/bin/clover-one" "$@"