#!/bin/sh
# How many concurrent requests does this box actually serve?
# Shared mmap payload (K3_PAYLOAD_RAM=0) so processes share physical pages, and
# O_DIRECT experts so expert traffic cannot evict the shared trunk.
# Total OMP threads held at 32 across all workers, so the comparison is fair.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
P1='{"input_ids":[1008,10484,318,15383,387],"max_new_tokens":8}'
P2='{"input_ids":[1008,10484,318,10417,387],"max_new_tokens":8}'
P3='{"input_ids":[1008,10484,318,19509,387],"max_new_tokens":8}'
P4='{"input_ids":[1008,10604,19645,387],"max_new_tokens":8}'
P5='{"input_ids":[17285,9620,2069,28542],"max_new_tokens":8}'
P6='{"input_ids":[1008,12955,318,276,18985,387],"max_new_tokens":8}'
P7='{"input_ids":[1008,17296,318,6903,1900,67244,24231,43458],"max_new_tokens":8}'
P8='{"input_ids":[68430,387],"max_new_tokens":8}'

echo "warming the page cache"
printf '%s\n' "$P1" | K3_PAYLOAD_RAM=0 K3_EXPERT_DIRECT=1 "$S/run.sh" >/dev/null 2>&1

for N in 1 2 4 8; do
  THREADS=$((32 / N))
  T0=$(date +%s.%N)
  I=1
  while [ "$I" -le "$N" ]; do
    eval "REQ=\$P$I"
    printf '%s\n' "$REQ" \
      | OMP_NUM_THREADS=$THREADS K3_PAYLOAD_RAM=0 K3_EXPERT_DIRECT=1 \
        "$S/run.sh" >"/tmp/c$N-$I.out" 2>/dev/null &
    I=$((I + 1))
  done
  wait
  T1=$(date +%s.%N)
  W=$(echo "$T1-$T0" | bc)
  OK=$(cat /tmp/c$N-*.out 2>/dev/null | grep -c DONE_JSON)
  printf 'workers %-2d  threads/worker %-2d  wall %7.2f s  done %d  throughput %.3f req/s\n' \
    "$N" "$THREADS" "$W" "$OK" "$(echo "$N / $W" | bc -l)"
  free -g | sed -n 2p
done
