#!/usr/bin/env bash
# The layer 12 / layer 24 pair says the output gate decides whether the attention's
# constancy survives. That is one pair. This patches seven more KDA layers, three of
# them block-write and four ordinary, to see whether gate transparency predicts
# output constancy across all of them or only happened to fit two.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers
LAYERS="10 13 36 48 60 72 84"

echo "=== backup ==="
cd "$T" || exit 1
tar -cf /tmp/kda-before-more.tar $(for l in $LAYERS; do echo transformer-$l/transformer-$l.c; done)
echo "backup: $(stat -c%s /tmp/kda-before-more.tar) bytes"

echo "=== patch ==="
python3 "$T/port-kda-internals.py" $LAYERS || exit 1

echo "=== rebuild ==="
for layer in $LAYERS; do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; exit 1; }
done
echo "built: $LAYERS"

echo "=== gate, tracing off ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in $LAYERS; do
    env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
        ./test-reference "$R/dataset" "$T" france "$layer" 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"
done

echo "=== capture ==="
rm -rf /tmp/kda-genesis /tmp/kda-french
for pair in "genesis:767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252" \
            "french:3638 14059 154745 144797 2613 655 21594 289 36283 916 4334 632 2917 3829 68"; do
    name=${pair%%:*}; ids=${pair##*:}
    mkdir -p "/tmp/kda-$name"
    cd "$CODE/pipeline" || exit 1
    CLOVER_VECTORS="/tmp/kda-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $CODE serve" > "/tmp/kda-$name.log" 2>&1
    echo "$name: $(grep -o '\[.*\]' "/tmp/kda-$name.log" | head -1)"
done
echo "=== done ==="
