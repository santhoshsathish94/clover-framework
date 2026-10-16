#!/bin/sh
# The question the 64-repeat prompt could not answer: does an exactly repeated REQUEST hit?
# Same prompt three times in one process, so the cache survives between them.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/repeat-req
mkdir -p $OUT
cat > $OUT/cfg.json <<'EOF'
{
 "max_input_tokens": 128,
 "max_output_tokens": 128,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 4096,
 "memory_reserve_mib": 8192
}
EOF
ids=$(awk -F'\t' '$1==1 {print $3}' /opt/clover-k3/prompts.tsv)
other=$(awk -F'\t' '$1==2 {print $3}' /opt/clover-k3/prompts.tsv)
{
  echo "{\"input_ids\":[$ids],\"max_new_tokens\":6}"
  echo "{\"input_ids\":[$ids],\"max_new_tokens\":6}"
  echo "{\"input_ids\":[$other],\"max_new_tokens\":6}"
  echo "{\"input_ids\":[$ids],\"max_new_tokens\":6}"
} > $OUT/req.json

env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg.json \
  K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
  $S/bin/clover-one $S/bin/dataset < $OUT/req.json > $OUT/out.txt 2> $OUT/err.txt

echo "allocation:"
grep -o '"expert_pipeline_bytes":[0-9]*,"result_cache_bytes":[0-9]*,"result_cache_entries":[0-9]*' $OUT/out.txt
echo
echo "request 1 = A, 2 = A repeat, 3 = B different, 4 = A again"
echo "tokens per request:"
grep -o '"tokens":\[[0-9,]*\]' $OUT/out.txt || grep -o '"token":[0-9]*' $OUT/out.txt | grep -o '[0-9]*' | tr '\n' ' '
echo
echo "cumulative hits/misses after each request:"
grep -o '"result_hits":[0-9]*,"result_misses":[0-9]*,"result_stores":[0-9]*,"result_evictions":[0-9]*' $OUT/out.txt
echo
echo "wall seconds per step (prefill first of each request):"
grep -o '"seconds":[0-9.]*' $OUT/err.txt | grep -o '[0-9.]*' | tr '\n' ' '
echo
touch $OUT/DONE
