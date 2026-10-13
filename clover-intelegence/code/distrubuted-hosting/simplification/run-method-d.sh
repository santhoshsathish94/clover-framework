#!/usr/bin/env bash
# Method D: patch all 92 pods for a configurable top-K, rebuild, verify with the hook
# off, then measure fidelity at K = 12, 8 and 4 across the full twelve prompts.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers
S=$CODE/simplification

echo "=== backup ==="
cd "$T" || exit 1
tar -cf /tmp/topk-before.tar $(for l in $(seq 1 92); do echo transformer-$l/transformer-$l.c; done)
echo "backup: $(stat -c%s /tmp/topk-before.tar) bytes"

echo "=== patch ==="
python3 "$S/port-topk.py" || exit 1

echo "=== rebuild ==="
fail=0
for layer in $(seq 1 92); do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; fail=1; }
done
[ $fail -eq 0 ] && echo "all built" || exit 1

echo "=== gate with CLOVER_TOPK unset, must be unchanged ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in 2 12 45 91; do
    env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
        ./test-reference "$R/dataset" "$T" france "$layer" 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"
done

for k in 12 8 4; do
    echo "=== K=$k ==="
    /tmp/fidelity.sh "topk$k" CLOVER_TOPK=$k
    /tmp/extract.sh "topk$k" > /dev/null 2>&1
    /tmp/compare.sh "topk$k"
done
echo "=== done ==="
