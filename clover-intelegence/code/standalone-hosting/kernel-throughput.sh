#!/bin/sh
# Compare the two weight-reading kernels the engine already has.
#
# Q reads int8 and achieves 39.5 GB/s against a measured 47.5 GB/s ceiling. X
# reads mxfp4 for the experts. If X sustains comparable GB/s then halving the
# trunk's bytes would halve its time, and fetching BF16 source to requantise is
# worth the download. If the 4-bit unpack costs throughput, the gain is smaller
# than the byte ratio suggests and should be known before moving 100+ GB onto a
# disk that is already 97% full.
#
# OPSTAT_JSON carries s, gflop, wgb and calls per operator, so one decode step
# answers it. The last line is a decode step, not the prompt pass.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TSV=/opt/clover-k3/prompts.tsv

ids=$(awk -F'\t' '$1==1 {print $3}' "$TSV")
[ -z "$ids" ] && { echo "no row 1 in $TSV"; exit 1; }
echo "prompt ids: $ids"

req="{\"input_ids\":[$ids],\"max_new_tokens\":3}"
echo "$req" > /tmp/req.json

env -i PATH="$PATH" \
    CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$S/bin/configs/generation.json" \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=24 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$S/bin/clover-one" "$S/bin/dataset" < /tmp/req.json > /tmp/opstat.out 2> /tmp/opstat.err

echo "exit $?"
echo "--- tokens emitted:"
grep -c TOKEN_JSON /tmp/opstat.out 2>/dev/null || echo 0
echo "--- last OPSTAT_JSON (a decode step):"
grep OPSTAT_JSON /tmp/opstat.err | tail -1 > /tmp/opstat.last
wc -c /tmp/opstat.last
echo "--- stderr tail if it failed:"
tail -3 /tmp/opstat.err
