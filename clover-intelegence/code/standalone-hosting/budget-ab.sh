#!/bin/sh
# Isolates what the cache-budget fix actually bought: the expert pipeline waking up,
# against the result cache that the budget fix also enabled but which never hits.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/budget-ab
mkdir -p $OUT
ids=$(awk -F'\t' '$1==1 {print $3}' /opt/clover-k3/prompts.tsv)
echo "{\"input_ids\":[$ids],\"max_new_tokens\":12}" > $OUT/req.json

arm() {
  name=$1; pipe=$2; res=$3
  cat > $OUT/cfg-$name.json <<EOF
{
 "max_input_tokens": 128,
 "max_output_tokens": 128,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": $pipe,
 "expert_result_cache_mib": $res,
 "memory_reserve_mib": 8192
}
EOF
  env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg-$name.json \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    $S/bin/clover-one $S/bin/dataset < $OUT/req.json > $OUT/$name.out 2> $OUT/$name.err
}

# A reproduces pre-fix behaviour: every cache starved to zero.
arm prefix_off 0 0
# B is the fix with only the pipeline awake.
arm pipe_only 72 0
# C adds the result cache, to price a cache that cannot hit.
arm pipe_res 72 4096

for a in prefix_off pipe_only pipe_res; do
  echo "=== $a ==="
  grep -o '"expert_pipeline_bytes":[0-9]*,"result_cache_bytes":[0-9]*' $OUT/$a.out
  echo "tokens: $(grep -o '"token":[0-9]*' $OUT/$a.out | grep -o '[0-9]*' | tr '\n' ' ')"
  # Drop prefill and the first two decode steps, then average the rest.
  grep -o '"seconds":[0-9.]*' $OUT/$a.err | grep -o '[0-9.]*' | tail -n +4 \
    | awk '{s+=$1; n++} END {printf "steady s/token: %.4f over %d steps\n", s/n, n}'
  grep -o '"result_hits":[0-9]*,"result_misses":[0-9]*' $OUT/$a.out | tail -1
  grep -o '"pipeline_reads":[0-9]*,"pipeline_bytes":[0-9]*' $OUT/$a.out | tail -1
  grep -o '"pipeline_wait":[0-9.]*' $OUT/$a.out | tail -1
done
touch $OUT/DONE
