#!/usr/bin/env bash
# Derive the tail walker from the real gate and run it on the france prompt.
set -uo pipefail
ulimit -c 0
ROOT=/opt/clover-k3/clover-intelegence
N=$ROOT/code/distrubuted-hosting/normalization
LEAVES=$ROOT/dataset/leaves.json
cd "$N"

echo "=== backup before touching anything ==="
tar -cf /tmp/normalization-before.tar test-reference.c tail-walk.c derive-tail-walk.py 2>/dev/null
echo "backup: $(ls -l /tmp/normalization-before.tar 2>/dev/null | awk '{print $5}') bytes"

echo "=== derive ==="
python3 derive-tail-walk.py || exit 1

echo "=== confirm only additions, no assertion touched ==="
diff <(grep -c assert test-reference.c) <(grep -c assert tail-walk.c) && echo "assert count UNCHANGED" || echo "assert count differs (expected: walker adds 2)"
diff test-reference.c tail-walk.c | grep -E "^<" && echo "REMOVED LINES ABOVE - investigate" || echo "no lines removed from the gate"

echo "=== build ==="
gcc -std=c11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp tail-walk.c -lm -o bin/tail-walk || exit 1

echo "=== run (france) ==="
env -i PATH="$PATH" K3_INDEX="$ROOT/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
    K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
    OMP_PROC_BIND=close OMP_PLACES=cores \
    ./bin/tail-walk "$ROOT/dataset" france "$LEAVES" 2>&1 | grep -E "TAILWALK|TAILSRC|NORMALIZATION_PASS|differs|Error|error"

echo "=== done rc=$? ==="
