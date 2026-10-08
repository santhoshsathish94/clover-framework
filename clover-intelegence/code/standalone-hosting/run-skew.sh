#!/bin/sh
# Re-run one binary against the skewed workload and record wall clock.
# Detached by the caller so an ssh drop cannot take it with it.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
B=$1
start=$(date +%s.%N)
env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
    CLOVER_CONFIG="$S/bin/configs/generation.json" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
    "$S/bin/$B" < /tmp/skew.jsonl > "/tmp/$B.s.out" 2>"/tmp/$B.s.err"
end=$(date +%s.%N)
{
  printf "%-16s wall %7.1f s   tokens %3d   evaluations %3d   decode steps %3d\n" "$B" \
    "$(echo "$end - $start" | bc)" \
    "$(grep -c TOKEN_JSON /tmp/$B.s.out)" \
    "$(grep -c STEP_JSON /tmp/$B.s.err)" \
    "$(( $(grep -c STEP_JSON /tmp/$B.s.err) - 16 ))"
  printf "   widths: %s\n" "$(grep STEP_WIDTH /tmp/$B.s.err | awk '{printf "%s ",$2}')"
} > "/tmp/$B.summary"
echo DONE >> "/tmp/$B.summary"
