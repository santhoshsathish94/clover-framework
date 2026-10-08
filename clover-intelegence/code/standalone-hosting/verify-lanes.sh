#!/bin/sh
# Lane-ising the sequence state must not change a single token. Runs the
# reference binary and the lane build over the six prompts and diffs the ids.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
cd "$S" || exit 1
run() {
  { for ids in 1008,10484,318,15383,387 1008,10484,318,10417,387 \
               1008,10484,318,19509,387 1008,10604,19645,387 \
               17285,9620,2069,28542 1008,12955,318,276,18985,387; do
      printf '{"input_ids":[%s],"max_new_tokens":6}\n' "$ids"
    done; } \
  | env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
      CLOVER_CONFIG="$S/bin/configs/generation.json" \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
      "$S/bin/$1" >"/tmp/$1.out" 2>/dev/null
  grep -oE '"index":[0-9]+,"token":[0-9]+' "/tmp/$1.out" \
    | grep -oE 'token":[0-9]+' | cut -d: -f2 | tr '\n' ' '
}
A=$(run clover-one)
B=$(run clover-one-lane)
echo "shared sequence state : $A"
echo "lane-owned state      : $B"
[ "$A" = "$B" ] && echo "IDENTICAL" || echo "DIFFERENT"
