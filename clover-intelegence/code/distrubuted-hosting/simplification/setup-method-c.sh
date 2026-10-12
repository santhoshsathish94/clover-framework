#!/usr/bin/env bash
# Method C setup: restore pod 12 from the Method B backup, patch every KDA pod with the
# per-layer substitution hook, rebuild, verify a sample with the hook off, and cache one
# attention vector per layer from genesis position 4.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers
S=$CODE/simplification

MLA=" 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 92 "
KDA=""
for L in $(seq 1 92); do
    case "$MLA" in *" $L "*) continue ;; esac
    KDA="$KDA $L"
done

echo "=== restore pod 12 to its pre-Method-B state ==="
tar -xf /tmp/constattn-before.tar -C "$T"
grep -c CLOVER_CONST_ATTN "$T/transformer-12/transformer-12.c" | sed 's/^/const-attn hunks in pod 12 after restore: /'

echo "=== backup all KDA pods ==="
cd "$T" || exit 1
tar -cf /tmp/constattn-all-before.tar $(for l in $KDA; do echo transformer-$l/transformer-$l.c; done)
echo "backup: $(stat -c%s /tmp/constattn-all-before.tar) bytes"

echo "=== patch ==="
python3 "$S/port-const-attn-layered.py" $KDA || exit 1

echo "=== rebuild ==="
for layer in $KDA; do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; exit 1; }
done
echo "all built"

echo "=== gate with the hook off, stratified sample ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in 2 12 26 45 65 90; do
    env -i PATH="$PATH" K3_INDEX="$R/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
        K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
        ./test-reference "$R/dataset" "$T" france "$layer" 2>&1 | grep -E "VERIFIED|FAIL|differs|rror"
done

echo "=== cache one attention vector per KDA layer, genesis position 4 ==="
rm -rf /tmp/ca; mkdir -p /tmp/ca
n=0
for layer in $KDA; do
    python3 "$S/extract-record.py" "/tmp/kda-genesis/t.$layer.bin" 5007 4 "/tmp/ca/t.$layer.f32" > /dev/null 2>&1 && n=$((n+1))
done
echo "cached $n of $(echo $KDA | wc -w) layers"
echo "=== done ==="
