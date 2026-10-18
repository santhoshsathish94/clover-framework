#!/bin/sh
# Does per-position prefill cost really bottom out at width 8? Measured, not reasoned.
# Everything except NPOS_SLOTS is held constant: same prompt, no expert cache,
# same thread count, payload in RAM.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/wsweep
mkdir -p $OUT

PROMPT=256
python3 - "$PROMPT" <<'PY'
import json, sys
ids = json.load(open("/tmp/req8k.json"))["input_ids"][:int(sys.argv[1])]
open("/tmp/wsweep/req.json", "w").write(
    json.dumps({"input_ids": ids, "max_new_tokens": 2}, separators=(",", ":")) + "\n")
PY

cat > $OUT/cfg.json <<EOF
{
 "max_input_tokens": $PROMPT,
 "max_output_tokens": 64,
 "eos_token_id": 163585,
 "head_cache_mib": 2240,
 "expert_pipeline_mib": 72,
 "expert_result_cache_mib": 0,
 "memory_reserve_mib": 8192
}
EOF

echo "width sweep, prompt $PROMPT positions, expert cache off"
echo
printf "%4s %10s %10s %12s %14s %16s\n" W wall startup "prefill s" "s/position" "expert GB read"
for W in 8 10 12 16 24 32; do
  t0=$(date +%s)
  env -i PATH="$PATH" CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$OUT/cfg.json \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    /tmp/cw$W $S/bin/dataset < $OUT/req.json > $OUT/w$W.out 2> $OUT/w$W.err || echo "W=$W EXIT $?"
  t1=$(date +%s)
  awk -v W=$W -v wall=$((t1-t0)) '
    /"startup_seconds"/ { match($0,/"startup_seconds":[0-9.]+/); s=substr($0,RSTART+19,RLENGTH-19) }
    END { }' $OUT/w$W.out > /dev/null
  start=$(grep -o '"startup_seconds":[0-9.]*' $OUT/w$W.out | grep -o '[0-9.]*')
  steps=$(grep -o '"seconds":[0-9.]*' $OUT/w$W.err | grep -o '[0-9.]*' | awk '{s+=$1} END {print s+0}')
  bytes=$(grep -o '"expert_read_bytes":[0-9]*' $OUT/w$W.err | grep -o '[0-9]*' | tail -1)
  tok=$(grep -o '"token":[0-9]*' $OUT/w$W.out | grep -o '[0-9]*' | tr '\n' ' ')
  echo "$W $((t1-t0)) $start $steps $bytes $tok" >> $OUT/raw.txt
  awk -v W=$W -v wall=$((t1-t0)) -v st="$start" -v steps="$steps" -v b="$bytes" -v P=$PROMPT \
    'BEGIN { pf = wall - st - steps; printf "%4d %10d %10.2f %12.1f %14.4f %16.1f\n", W, wall, st, pf, pf/P, b/1e9 }'
  echo "     tokens: $tok"
done
touch $OUT/DONE
