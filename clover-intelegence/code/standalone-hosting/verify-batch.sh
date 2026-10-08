#!/bin/sh
# Batching must not change a single token. The same six prompts go in at once;
# each lane's ids are compared against the same binary run one lane at a time.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
cd "$S" || exit 1

for ids in 1008,10484,318,15383,387 1008,10484,318,10417,387 \
           1008,10484,318,19509,387 1008,10604,19645,387 \
           17285,9620,2069,28542 1008,12955,318,276,18985,387; do
  printf '{"input_ids":[%s],"max_new_tokens":6}\n' "$ids"
done > /tmp/six.jsonl

run() {
  start=$(date +%s.%N)
  env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
      CLOVER_CONFIG="$S/bin/configs/generation.json" \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
      "$S/bin/$1" < /tmp/six.jsonl > "/tmp/$1.out" 2>"/tmp/$1.err"
  end=$(date +%s.%N)
  echo "$1 wall $(echo "$end - $start" | bc) s"
  # group ids by request, in request order
  grep -oE '"request":[0-9]+,"index":[0-9]+,"token":[0-9]+' "/tmp/$1.out" \
    | tr -d '"' | tr ',' ' ' \
    | awk '{split($1,a,":");split($2,b,":");split($3,c,":");print a[2],b[2],c[2]}' \
    | sort -k1,1n -k2,2n | awk '{printf "%s%s", (r!=$1?(NR>1?"\n":"")$1": ":" "), $3; r=$1} END{print ""}' \
    > "/tmp/$1.ids"
}

run clover-one-L1
run clover-one-L8
echo "--- one lane at a time"
cat /tmp/clover-one-L1.ids
echo "--- eight lanes, batched"
cat /tmp/clover-one-L8.ids
if cmp -s /tmp/clover-one-L1.ids /tmp/clover-one-L8.ids; then echo IDENTICAL; else echo DIFFERENT; fi
