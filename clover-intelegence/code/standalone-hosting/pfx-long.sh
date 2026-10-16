#!/bin/sh
# Prefix lengths past one pass, which is what the old mla_klat[8] indexing could not do.
# PROMPT positions taken from the head of the 8k request, CACHE of them saved.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/pfx-long
mkdir -p $OUT
PROMPT=${PROMPT:-256}
CACHE=${CACHE:-192}
NEW=${NEW:-6}

python3 - "$PROMPT" "$NEW" <<'PY'
import json, sys
n, new = int(sys.argv[1]), int(sys.argv[2])
ids = json.load(open("/tmp/req8k.json"))["input_ids"][:n]
body = json.dumps({"input_ids": ids, "max_new_tokens": new}, separators=(",", ":"))
# request_next splits on newline, so the terminator is required.
open("/tmp/pfx-long/req.json", "w").write(body + "\n")
print(f"prompt {len(ids)} positions, {len(set(ids))} distinct")
PY

cat > $OUT/cfg.json <<EOF
{
 "max_input_tokens": $PROMPT,
 "max_output_tokens": 128,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 8192
}
EOF

run() {
  env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg.json \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=${EXPERTS:-12} \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores "$@" \
    /tmp/clover-one-9k2 $S/bin/dataset < $OUT/req.json
}

echo "caching $CACHE of $PROMPT"
rm -f $OUT/cache.bin
t0=$(date +%s)
run K3_PFXN=$CACHE K3_PFXSAVE=$OUT/cache.bin > $OUT/save.out 2> $OUT/save.err || echo "SAVE EXIT $?"
t1=$(date +%s)
echo "save wall $((t1-t0))s  tokens: $(grep -o '"token":[0-9]*' $OUT/save.out | grep -o '[0-9]*' | tr '\n' ' ')"
echo "cache bytes: $(stat -c %s $OUT/cache.bin 2>/dev/null || echo MISSING)"

t2=$(date +%s)
run K3_PFXN=$CACHE K3_PFXLOAD=$OUT/cache.bin > $OUT/load.out 2> $OUT/load.err || echo "LOAD EXIT $?"
t3=$(date +%s)
echo "load wall $((t3-t2))s  tokens: $(grep -o '"token":[0-9]*' $OUT/load.out | grep -o '[0-9]*' | tr '\n' ' ')"

echo "startup seconds: save $(grep -o '"startup_seconds":[0-9.]*' $OUT/save.out | grep -o '[0-9.]*') load $(grep -o '"startup_seconds":[0-9.]*' $OUT/load.out | grep -o '[0-9.]*')"
touch $OUT/DONE
