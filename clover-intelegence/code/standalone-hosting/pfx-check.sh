#!/bin/sh
# Prefix cache reworked onto the sequence arrays. Run 1 generates and saves,
# run 2 loads and must produce the same tokens from the same prompt.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/pfx-check
mkdir -p $OUT
N=${N:-4}

cat > $OUT/cfg.json <<'EOF'
{
 "max_input_tokens": 128,
 "max_output_tokens": 128,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 8192
}
EOF
ids=$(awk -F'\t' '$1==1 {print $3}' /opt/clover-k3/prompts.tsv)
echo "{\"input_ids\":[$ids],\"max_new_tokens\":6}" > $OUT/req.json
echo "prompt positions: $(echo $ids | tr ',' '\n' | wc -l), caching $N"

run() {
  env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg.json \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores "$@" \
    $S/bin/clover-one $S/bin/dataset < $OUT/req.json
}

rm -f $OUT/cache.bin
run K3_PFXN=$N K3_PFXSAVE=$OUT/cache.bin > $OUT/save.out 2> $OUT/save.err || echo "SAVE EXIT $?"
echo "save tokens: $(grep -o '"token":[0-9]*' $OUT/save.out | grep -o '[0-9]*' | tr '\n' ' ')"
echo "cache bytes: $(stat -c %s $OUT/cache.bin 2>/dev/null || echo MISSING)"

run K3_PFXN=$N K3_PFXLOAD=$OUT/cache.bin > $OUT/load.out 2> $OUT/load.err || echo "LOAD EXIT $?"
echo "load tokens: $(grep -o '"token":[0-9]*' $OUT/load.out | grep -o '[0-9]*' | tr '\n' ' ')"

echo "--- per-step seconds, save run ---"
grep -o '"seconds":[0-9.]*' $OUT/save.err | grep -o '[0-9.]*' | tr '\n' ' '; echo
echo "--- per-step seconds, load run ---"
grep -o '"seconds":[0-9.]*' $OUT/load.err | grep -o '[0-9.]*' | tr '\n' ' '; echo
tail -2 $OUT/load.err | head -1
touch $OUT/DONE
