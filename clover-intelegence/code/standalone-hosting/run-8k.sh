#!/bin/sh
# One request: 8192 input tokens, 1000 output, with the expert cache on.
#
# Capacity 24 does not fit here. Base is 81.3 GB measured from the 1000/1000
# run, MLA KV at 9192 positions is 21.8 GB, and the cache is capacity x 93 x
# 17,547,264 B, so 24 would need 39.2 GB for a 142 GB total against 124 GB of
# RAM. Capacity 8 costs 13.1 GB and lands near 116 GB, the same peak already
# run successfully.
#
# The prompt is 8192 tokens but only 952 distinct: the box has 2,066 real
# tokens and no tokenizer, so it repeats 4x. That flatters the cache, and the
# number should be read with that attached.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=${OUT:-/tmp/run8k}
BIN=${BIN:-/tmp/clover-one-9k}
REQ=${REQ:-/tmp/req8k.json}
CFG=${CFG:-/tmp/gen8k.json}
CACHE=${CACHE:-8}

mkdir -p "$OUT"
echo "binary $BIN  cache $CACHE"
date

/usr/bin/time -v -o "$OUT/time.txt" \
  env -i PATH="$PATH" \
    CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$CFG" \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE="$CACHE" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" "$S/bin/dataset" \
    < "$REQ" > "$OUT/out.txt" 2> "$OUT/err.txt"
echo "exit $?"
date

grep STEP_JSON "$OUT/err.txt" > "$OUT/steps.jsonl" 2>/dev/null
echo "steps: $(wc -l < "$OUT/steps.jsonl")"
echo "tokens emitted: $(grep -c TOKEN_JSON "$OUT/out.txt" 2>/dev/null)"
grep -E 'wall clock|User time|System time|Maximum resident|Major' "$OUT/time.txt"
touch /tmp/run8k.done
