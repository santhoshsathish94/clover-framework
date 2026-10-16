#!/bin/sh
# 1000 input tokens, 1000 output tokens, one prompt.
#
# Needs a capacity build: generation.h caps max_input+max_output at
# GENERATION_CAPACITY, shipped as 256, so this uses the -DGENERATION_CAPACITY=2048
# binary. Memory is not the obstacle at this length - MLA KV is 2.37 MB a
# position, so 2000 positions is 4.7 GB - provided the expert cache stays off,
# which is why it is the RAM the creator wants back for context.
#
# Expert cache is 0 deliberately: with it on, peak RSS was 116.7 GB of 124 GB
# and there would be no room for the KV this run exists to exercise.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=${OUT:-/tmp/run1000}
BIN=${BIN:-/tmp/clover-one-2k}
REQ=${REQ:-/tmp/req1000.json}
CFG=${CFG:-/tmp/gen1000.json}

mkdir -p "$OUT"
echo "binary $BIN"
echo "request $(python3 -c "import json;d=json.load(open('$REQ'));print(len(d['input_ids']),'in,',d['max_new_tokens'],'out')")"
date

/usr/bin/time -v -o "$OUT/time.txt" \
  env -i PATH="$PATH" \
    CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$CFG" \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" "$S/bin/dataset" \
    < "$REQ" > "$OUT/out.txt" 2> "$OUT/err.txt"
echo "exit $?"
date

grep STEP_JSON "$OUT/err.txt" > "$OUT/steps.jsonl" 2>/dev/null
echo "steps: $(wc -l < "$OUT/steps.jsonl")"
echo "tokens emitted: $(grep -c TOKEN_JSON "$OUT/out.txt" 2>/dev/null)"
grep -E 'wall clock|User time|System time|Maximum resident|Major' "$OUT/time.txt"
touch /tmp/run1000.done
