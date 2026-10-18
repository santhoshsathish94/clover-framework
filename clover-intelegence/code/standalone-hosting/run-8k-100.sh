#!/bin/sh
# Baseline: 8192-token prompt, 100 output tokens, no prefix cache (save is still capped at 8).
# Expert cache held to 12/layer because 8292 positions of MLA K/V take ~19.6 GB.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/run8k100
rm -rf $OUT; mkdir -p $OUT

cat > $OUT/config.json <<'EOF'
{
 "max_input_tokens": 8192,
 "max_output_tokens": 100,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 8192
}
EOF

python3 - <<'PY'
import json
d = json.load(open("/tmp/req8k.json"))
d["max_new_tokens"] = 100
json.dump(d, open("/tmp/run8k100/req.json", "w"))
print("prompt tokens", len(d["input_ids"]), "distinct", len(set(d["input_ids"])))
PY

date +%s > $OUT/start.epoch
env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/config.json \
  K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=12 \
  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
  /tmp/clover-one-9k2 $S/bin/dataset < $OUT/req.json > $OUT/out.txt 2> $OUT/err.txt &
ENGINE=$!
echo $ENGINE > $OUT/pid

# Peak RSS matters here: an OOM two hours in would cost the whole run.
( while kill -0 $ENGINE 2>/dev/null; do
    awk '/^VmRSS/ {print systime(), $2}' /proc/$ENGINE/status >> $OUT/rss.log 2>/dev/null || true
    grep -m1 MemAvailable /proc/meminfo | awk -v t=$(date +%s) '{print t, $2}' >> $OUT/avail.log
    sleep 30
  done ) &

wait $ENGINE
echo $? > $OUT/exit
date +%s > $OUT/end.epoch
touch $OUT/DONE
