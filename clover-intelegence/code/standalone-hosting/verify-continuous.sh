#!/bin/sh
# Static batching has to drain a batch before admitting the next, so one long
# request pins seven idle lanes for the rest of the batch. Sixteen requests,
# two of 32 output tokens and fourteen of 4, eight lanes.
#
# Decode steps are counted as STEP_JSON minus one prefill per request; every
# prompt here is at most eight tokens, so each takes exactly one prefill pass.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
cd "$S" || exit 1

set -- 1008,10484,318,15383,387 1008,10484,318,10417,387 \
       1008,10484,318,19509,387 1008,10604,19645,387 \
       17285,9620,2069,28542 1008,12955,318,276,18985,387 \
       68430,387 1008,17296,318,6903,1900,67244

: > /tmp/skew.jsonl
n=0
for pass in 1 2; do
  for ids in "$@"; do
    n=$((n+1))
    if [ $((n % 8)) -eq 1 ]; then m=32; else m=4; fi
    printf '{"input_ids":[%s],"max_new_tokens":%d}\n' "$ids" "$m" >> /tmp/skew.jsonl
  done
done
echo "requests $(wc -l < /tmp/skew.jsonl), output tokens $(awk -F'max_new_tokens":' '{split($2,a,"}");s+=a[1]} END{print s}' /tmp/skew.jsonl)"

run() {
  start=$(date +%s.%N)
  env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
      CLOVER_CONFIG="$S/bin/configs/generation.json" \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
      "$S/bin/$1" < /tmp/skew.jsonl > "/tmp/$1.s.out" 2>"/tmp/$1.s.err"
  end=$(date +%s.%N)
  w=$(echo "$end - $start" | bc)
  t=$(grep -c TOKEN_JSON "/tmp/$1.s.out")
  e=$(grep -c STEP_JSON "/tmp/$1.s.err")
  d=$((e - 16))
  printf "%-16s wall %7.1f s   tokens %3d   evaluations %3d   decode steps %3d\n" "$1" "$w" "$t" "$e" "$d"
  if grep -q STEP_WIDTH "/tmp/$1.s.err"; then
    printf "   step widths: %s\n" "$(grep STEP_WIDTH "/tmp/$1.s.err" | awk '{printf "%s ",$2}')"
  fi
  grep -oE '"request":[0-9]+,"index":[0-9]+,"token":[0-9]+' "/tmp/$1.s.out" \
    | tr -d '"' | tr ',' ' ' \
    | awk '{split($1,a,":");split($2,b,":");split($3,c,":");print a[2],b[2],c[2]}' \
    | sort -k1,1n -k2,2n > "/tmp/$1.s.ids"
}

run clover-one-B8
run clover-one-C8
echo "--- tokens"
if cmp -s /tmp/clover-one-B8.s.ids /tmp/clover-one-C8.s.ids; then echo "  IDENTICAL"
else echo "  DIFFERENT"; diff /tmp/clover-one-B8.s.ids /tmp/clover-one-C8.s.ids | head -20; fi
