#!/usr/bin/env bash
# Converts each routed layer to directly mapped expert values, verifies it against the
# compressed source, then retires that source's payload. Two layers run at a time.
set -euo pipefail
ulimit -c 0
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
DATASET=${CLOVER_DATASET:-$HERE/bin/dataset}
FIRST=${1:?first layer}
LAST=${2:?last layer}
JOBS=${3:-2}
INDEX_BYTES=1094944          # 32-byte header plus the block index the reader still needs
NEED_GB=17
LOG="$HERE/bin/direct-conversion.log"

convert_layer() {
    local layer=$1 root="$DATASET/root-$1"
    if [ -e "$root/experts.direct" ]; then
        printf 'SKIP layer=%s already converted\n' "$layer"; return 0
    fi
    local avail
    avail=$(df -BG --output=avail "$root" | tail -n 1 | tr -dc '0-9')
    if [ "$avail" -lt "$NEED_GB" ]; then
        printf 'STOP layer=%s free=%sG below %sG\n' "$layer" "$avail" "$NEED_GB" >&2; return 2
    fi
    OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/build-direct-root "$DATASET" "$layer"
    truncate -s "$INDEX_BYTES" "$root/experts.bin"
    printf 'RETIRED layer=%s compressed payload reduced to %s bytes\n' "$layer" "$INDEX_BYTES"
}

export -f convert_layer
export DATASET HERE NEED_GB INDEX_BYTES

printf 'START %s layers %s..%s jobs=%s\n' "$(date -Is)" "$FIRST" "$LAST" "$JOBS" | tee -a "$LOG"
seq "$FIRST" "$LAST" | xargs -P "$JOBS" -I{} bash -c 'convert_layer {}' 2>&1 | tee -a "$LOG"
printf 'DONE %s\n' "$(date -Is)" | tee -a "$LOG"
df -BG --output=size,used,avail / | tail -n 1 | tee -a "$LOG"
