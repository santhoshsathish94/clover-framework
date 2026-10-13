#!/usr/bin/env bash
# Capture expert selection under three conditions on one prompt, so selection drift can
# be set against the fidelity already measured:
#   none    baseline, already captured as /tmp/rt-genesis
#   one     pod 12 only, the substitution that held 12 of 12
#   all     all 68 KDA layers, the substitution that collapsed every prompt
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
IDS="767 276 9868 4975 5232 276 29052 316 276 13618 316 276 13618 743 2932 2252"

# Confirm the substitution hook survived the later patches rather than assuming it did.
grep -c CLOVER_CONST_ATTN "$CODE/tansformers/transformer-12/transformer-12.c" \
    | sed 's/^/const-attn hunks in pod 12: /'
ls /tmp/ca/*.f32 2>/dev/null | wc -l | sed 's/^/cached vectors available: /'

rm -rf /tmp/rt-one /tmp/rt-all /tmp/one
mkdir -p /tmp/rt-one /tmp/rt-all /tmp/one
cp /tmp/ca/t.12.f32 /tmp/one/t.12.f32

cd "$CODE/pipeline" || exit 1
for arm in one all; do
    [ "$arm" = one ] && prefix=/tmp/one/t || prefix=/tmp/ca/t
    CLOVER_VECTORS="/tmp/rt-$arm/t" CLOVER_CONST_ATTN="$prefix" \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $IDS' | ./bin/pipeline-serve $CODE serve" > "/tmp/rt-$arm.log" 2>&1
    echo "$arm: $(grep -o '\[.*\]' "/tmp/rt-$arm.log" | head -1)"
done
df -h / | tail -1
echo "=== done ==="
