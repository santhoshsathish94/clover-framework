#!/bin/bash
# How a pod responds to CPU. Same layer, same six positions, experts pinned each
# time, thread count varied. This is what decides how many cores a pod is worth
# giving before the next core stops paying for itself.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
OUT=${1:-/tmp/thread-scaling.csv}

echo "layer,type,threads,resident_s,resident_ms_per_position" > "$OUT"

# One KDA layer and one MLA layer. They have different shapes, so they may not
# respond to cores the same way.
for pair in "46 KDA" "7 MLA"; do
    set -- $pair
    n=$1; kind=$2
    so="$CODE/tansformers/transformer-$n/bin/pipeline-stage.so"
    ds="$CODE/tansformers/transformer-$n/bin/dataset"
    ex="$DATA/root-$n/experts.direct"
    for threads in 1 2 4 8 16 32; do
        row=$( cd "$CODE/tansformers/transformer-$n" && ulimit -l unlimited && \
            OMP_NUM_THREADS=$threads OMP_PROC_BIND=false /tmp/pod-cost "$n" "$so" "$ds" "$ex" 2>/dev/null )
        resident=$(echo "$row" | cut -d, -f4)
        per=$(echo "$row" | cut -d, -f5)
        echo "$n,$kind,$threads,$resident,$per" >> "$OUT"
    done
done
echo "THREAD SWEEP COMPLETE" >&2
