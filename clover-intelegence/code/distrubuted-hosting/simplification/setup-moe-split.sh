#!/usr/bin/env bash
# Patch all 92 pods to record the routed and shared MoE paths separately, rebuild,
# verify a sample with tracing off, and capture two prompts.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers
S=$CODE/simplification

echo "=== backup ==="
cd "$T" || exit 1
tar -cf /tmp/moesplit-before.tar $(for l in $(seq 1 92); do echo transformer-$l/transformer-$l.c; done)
echo "backup: $(stat -c%s /tmp/moesplit-before.tar) bytes"

echo "=== patch ==="
python3 "$S/port-moe-split.py" || exit 1

echo "=== rebuild all 92 ==="
fail=0
for layer in $(seq 1 92); do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; fail=1; }
done
[ $fail -eq 0 ] && echo "all built" || exit 1

echo "=== gate, tracing off, stratified sample including MLA pods ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in 2 7 12 23 45 59 72 91; do
    env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
        ./test-reference "$R/dataset" "$T" france "$layer" 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"
done

echo "=== capture ==="
rm -rf /tmp/moe-genesis /tmp/moe-french
for pair in "genesis:767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252" \
            "french:3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68"; do
    name=${pair%%:*}; ids=${pair##*:}
    mkdir -p "/tmp/moe-$name"
    cd "$CODE/pipeline" || exit 1
    CLOVER_VECTORS="/tmp/moe-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $CODE serve" > "/tmp/moe-$name.log" 2>&1
    echo "$name: $(grep -o '\[.*\]' "/tmp/moe-$name.log" | head -1)"
done
df -h / | tail -1
echo "=== done ==="
