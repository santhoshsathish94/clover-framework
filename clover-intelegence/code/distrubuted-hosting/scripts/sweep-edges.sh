#!/bin/bash
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
OUT=${1:-/tmp/edge-scaling.csv}
echo "pod,threads,warm_s,work_s,per_unit_ms,units" > "$OUT"
for t in 1 2 4 8 16 32; do
    OMP_NUM_THREADS=$t /tmp/edge-cost "$CODE" "$DATA" 2>/dev/null | tail -n +2 | while IFS= read -r line; do
        pod=$(echo "$line" | cut -d, -f1)
        rest=$(echo "$line" | cut -d, -f2-)
        echo "$pod,$t,$rest" >> "$OUT"
    done
done
echo EDGE SWEEP COMPLETE >&2
