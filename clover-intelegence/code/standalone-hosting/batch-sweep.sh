#!/bin/sh
# Throughput against batch width. Eight distinct prompts, sixteen output tokens
# each, submitted together; the lane count decides how many share a weight sweep.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
cd "$S" || exit 1

for ids in 1008,10484,318,15383,387 1008,10484,318,10417,387 \
           1008,10484,318,19509,387 1008,10604,19645,387 \
           17285,9620,2069,28542 1008,12955,318,276,18985,387 \
           68430,387 1008,17296,318,6903,1900,67244; do
  printf '{"input_ids":[%s],"max_new_tokens":16}\n' "$ids"
done > /tmp/eight.jsonl

echo "lanes    wall      tok/s   per-token"
for L in 1 2 4 8; do
  gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp \
      -DNPOS_SLOTS=8 -DK3_LANES=$L clover-one.c -lm -lpthread -o bin/clover-one-B$L || exit 1
  start=$(date +%s.%N)
  env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
      CLOVER_CONFIG="$S/bin/configs/generation.json" \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
      "$S/bin/clover-one-B$L" < /tmp/eight.jsonl > "/tmp/batch-$L.out" 2>"/tmp/batch-$L.err"
  end=$(date +%s.%N)
  w=$(echo "$end - $start" | bc)
  n=$(grep -c TOKEN_JSON "/tmp/batch-$L.out")
  printf "%5d  %7.1f  %9.4f  %8.3f\n" "$L" "$w" \
    "$(echo "$n / $w" | bc -l)" "$(echo "$w / $n" | bc -l)"
  grep -oE '"request":[0-9]+,"index":[0-9]+,"token":[0-9]+' "/tmp/batch-$L.out" \
    | tr -d '"' | tr ',' ' ' \
    | awk '{split($1,a,":");split($2,b,":");split($3,c,":");print a[2],b[2],c[2]}' \
    | sort -k1,1n -k2,2n | awk '{print $1,$3}' > "/tmp/batch-$L.ids"
done

echo "--- tokens identical across every lane count?"
for L in 2 4 8; do
  if cmp -s /tmp/batch-1.ids /tmp/batch-$L.ids; then echo "  lanes=$L IDENTICAL to lanes=1"
  else echo "  lanes=$L DIFFERENT from lanes=1"; fi
done
