#!/bin/bash
# usage: profile-run.sh base|blocked W NTOKENS PIN_LAYERS IDS_FILE
# Builds the engine with the stage profiler at width W and measures the pinned layers in place.
# "blocked" first patches a COPY of clover-one.c (Qm token-blocked); the deployed file is untouched.
set -u
VARIANT=$1; W=$2; NTOK=$3; PINS=$4; IDS=$5
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
H=/tmp/one-layer
INC="-I$S"
if [ "$VARIANT" = blocked ]; then
    mkdir -p "$H/patched"
    python3 "$H/patch-qm.py" "$S/clover-one.c" "$H/patched/clover-one.c" || exit 1
    INC="-I$H/patched -I$S"
fi
BIN="$H/lp-$VARIANT-$W"
gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS="$W" -DROOT_MAX_ROWS=64 \
    -DGENERATION_CAPACITY=2048 $INC "$H/layer-profile.c" -lm -lpthread -o "$BIN" 2> "$H/build-$VARIANT-$W.log" \
    || { tail -20 "$H/build-$VARIANT-$W.log"; exit 1; }
cat > "$H/prof-cfg.json" <<EOF
{
 "max_input_tokens": 512,
 "max_output_tokens": 8,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 8192
}
EOF
ulimit -l unlimited; ulimit -c 0
OUT="$H/prof-$VARIANT-$W-$NTOK.out"
/usr/bin/time -v -o "$OUT.time" env -i PATH="$PATH" CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$H/prof-cfg.json" \
    K3_INDEX=/opt/clover-k3/clover-intelegence/dataset/eqidx.bin K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" "$S/bin/dataset" "$IDS" "$NTOK" "$PINS" > "$OUT" 2>&1
echo "exit $?" >> "$OUT"
grep -E "Maximum resident|Elapsed|Major" "$OUT.time" >> "$OUT"
