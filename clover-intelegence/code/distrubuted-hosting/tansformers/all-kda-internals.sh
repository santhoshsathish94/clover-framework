#!/usr/bin/env bash
# Gate openness was measured on nine pods, seven of them deep, and the two shallowest
# were the two most open. That is a hint, not a finding. This patches every KDA pod so
# the gate can be profiled across the whole depth rather than sampled where I happened
# to look.
#
# MLA pods are skipped: they have no KDA output gate to record.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$CODE/tansformers

# full_attn_layers from config.json, converted to pod numbers (pod N = model layer N+1).
MLA="3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 92"
KDA=""
for L in $(seq 1 92); do
    case " $MLA " in *" $L "*) continue ;; esac
    KDA="$KDA $L"
done
echo "KDA pods: $(echo $KDA | wc -w)"

echo "=== backup ==="
cd "$T" || exit 1
tar -cf /tmp/kda-before-all.tar $(for l in $KDA; do echo transformer-$l/transformer-$l.c; done)
echo "backup: $(stat -c%s /tmp/kda-before-all.tar) bytes"

echo "=== patch (already-patched pods are skipped by the script) ==="
python3 "$T/port-kda-internals.py" $KDA || exit 1

echo "=== rebuild ==="
fail=0
for layer in $KDA; do
    path="$T/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
        -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN \
        "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so" || { echo "BUILD FAILED $layer"; fail=1; }
done
[ $fail -eq 0 ] && echo "all built" || exit 1

# Verifying all 68 would take half an hour of reference runs. A stratified sample is
# honest about what it covers: the patch is identical and mechanical in every pod.
echo "=== gate, tracing off, stratified sample ==="
cd "$T/remaining" || exit 1
R=/opt/clover-k3/clover-intelegence
for layer in 1 2 5 9 26 45 65 86 90; do
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
df -h / | tail -1
echo "=== done ==="
