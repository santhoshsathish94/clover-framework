#!/usr/bin/env bash
# Patch the KDA internals tracing into pods 12 and 24, rebuild them, and prove the
# arithmetic is unchanged before trusting anything the new records say.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers

echo "=== backup first ==="
tar -cf /tmp/kda-before.tar -C "$T" transformer-12/transformer-12.c transformer-24/transformer-24.c
echo "backup: $(stat -c%s /tmp/kda-before.tar) bytes"
sha256sum "$T/transformer-12/transformer-12.c" "$T/transformer-24/transformer-24.c"

echo "=== patch ==="
python3 "$T/port-kda-internals.py" 12 24 || exit 1

echo "=== rebuild just those two pods, same flags as build.sh ==="
for layer in 12 24; do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; exit 1; }
    echo "built layer $layer"
done

echo "=== gate: the reference harness for both layers, tracing OFF ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in 12 24; do
    env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
        ./test-reference "$R/dataset" "$T" france "$layer" 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"
done

echo "=== done ==="
