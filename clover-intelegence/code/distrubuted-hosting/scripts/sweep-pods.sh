#!/bin/bash
# Every layer pod in turn, each one measured with its own experts locked in RAM.
# One layer at a time, because 92 x 15.72 GB is 1.45 TB and this box has 124 GB.
# The process exits between layers, which releases the lock and the mapping.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
FIRST=${1:-1}
LAST=${2:-92}
OUT=${3:-/tmp/pod-sweep.csv}
LOG=/tmp/pod-sweep.log
: > "$LOG"
exec 2>>"$LOG"

echo "layer,cold_s,pin_s,resident_s,resident_ms_per_position,cold_ms_per_position,experts_gb" > "$OUT"

for n in $(seq "$FIRST" "$LAST"); do
    so="$CODE/tansformers/transformer-$n/bin/pipeline-stage.so"
    ds="$CODE/tansformers/transformer-$n/bin/dataset"
    ex="$DATA/root-$n/experts.direct"
    [ -f "$so" ] && [ -d "$ds" ] && [ -f "$ex" ] || { echo "skip $n" >>"$LOG"; continue; }
    # Evict this layer's experts so the cold pass is genuinely cold, without
    # disturbing the rest of the machine's page cache.
    python3 -c "
import os
fd = os.open('$ex', os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
"
    ( cd "$CODE/tansformers/transformer-$n" && ulimit -l unlimited && \
      OMP_NUM_THREADS=8 OMP_PROC_BIND=false /tmp/pod-cost "$n" "$so" "$ds" "$ex" ) >> "$OUT"
    echo "done $n at $(date -u +%T)" >> "$LOG"
done

echo "SWEEP COMPLETE $(date -u +%FT%TZ)" >> "$LOG"
