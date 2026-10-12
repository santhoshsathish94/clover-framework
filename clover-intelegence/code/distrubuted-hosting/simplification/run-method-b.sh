#!/usr/bin/env bash
# Method B: patch pod 12 to accept a cached constant attention output, rebuild, prove
# the gate still passes with the substitution off, then cache a vector and run fidelity.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers
S=$CODE/simplification

echo "=== backup ==="
tar -cf /tmp/constattn-before.tar -C "$T" transformer-12/transformer-12.c
echo "backup: $(stat -c%s /tmp/constattn-before.tar) bytes"

echo "=== patch ==="
python3 "$S/port-const-attn.py" 12 || exit 1

echo "=== rebuild ==="
path="$T/transformer-12"
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
    -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
    -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
    "$path/transformer-12.c" -lm -o "$path/bin/pipeline-stage.so" || exit 1
echo built

echo "=== gate with the substitution OFF, must be unchanged ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
    K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
    ./test-reference "$R/dataset" "$T" france 12 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"

echo "=== cache pod 12's attention from genesis position 4 ==="
python3 "$S/extract-record.py" /tmp/kda-genesis/t.12.bin 5007 4 /tmp/const-attn.f32 || exit 1

echo "=== fidelity ==="
/tmp/fidelity.sh methodB CLOVER_CONST_ATTN=/tmp/const-attn.f32
/tmp/extract.sh methodB > /dev/null 2>&1
/tmp/compare.sh methodB
