#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${CLOVER_DATASET:-$HERE/dataset}"
BINARY="${CLOVER_RESIDENT_BINARY:-/opt/clover-k3/clover-k3-resident-20261002-a/clover-k3}"
test -x "$BINARY"
test -f "$DATASET/eqidx.bin"
test -f "$DATASET/operators/trunk-0-qkv/qkv.bin"
exec env -i PATH="$PATH" \
  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
  K3_INDEX="$DATASET/eqidx.bin" \
  K3_PREPARED_DATA="$DATASET" \
  K3_OPERATOR_DIRECTORY="$DATASET/operators/qkv-all" \
  K3_TRUNK0_QKV="$DATASET/operators/trunk-0-qkv/qkv.bin" \
  "$BINARY"