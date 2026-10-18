#!/bin/sh
# Five in, five out, fully instrumented. Mapped rather than copied into RAM so it
# can share the box with a running job: counts do not depend on how fast it pages.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/trace55
mkdir -p $OUT

cat > $OUT/cfg.json <<'EOF'
{
 "max_input_tokens": 16,
 "max_output_tokens": 16,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 1024
}
EOF

ids=$(awk -F'\t' '$1==1 {print $3}' /opt/clover-k3/prompts.tsv)
echo "{\"input_ids\":[$ids],\"max_new_tokens\":5}" > $OUT/req.json
echo "prompt positions: $(echo $ids | tr ',' '\n' | wc -l)"

rm -f $OUT/stage.tsv $OUT/lstat.tsv
env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg.json \
  K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=0 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
  K3_STAGE=$OUT/stage.tsv K3_LSTAT=$OUT/lstat.tsv \
  OMP_NUM_THREADS=6 OMP_PROC_BIND=close OMP_PLACES=cores \
  $S/bin/clover-one-trace $S/bin/dataset < $OUT/req.json > $OUT/out.txt 2> $OUT/err.txt
echo "exit=$?"
echo "tokens: $(grep -o '"token":[0-9]*' $OUT/out.txt | grep -o '[0-9]*' | tr '\n' ' ')"
echo "stage rows: $(wc -l < $OUT/stage.tsv)   lstat rows: $(wc -l < $OUT/lstat.tsv)"
touch $OUT/DONE
