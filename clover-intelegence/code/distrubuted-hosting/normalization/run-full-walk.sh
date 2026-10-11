#!/usr/bin/env bash
# Re-derive the tail walker (now dumping the final vectors), then observe the
# last transformation: vector to token, through the real vocabulary head.
set -uo pipefail
ulimit -c 0
ROOT=/opt/clover-k3/clover-intelegence
N=$ROOT/code/distrubuted-hosting/normalization
S=$ROOT/code/distrubuted-hosting/server
LEAVES=$ROOT/dataset/leaves.json
HEAD=$ROOT/dataset/outputs/fruit.bin

echo "=== backup ==="
tar -cf /tmp/walk-before.tar -C "$N" tail-walk.c derive-tail-walk.py 2>/dev/null
tar -rf /tmp/walk-before.tar -C "$S" head-walk.c 2>/dev/null
echo "backup: $(stat -c%s /tmp/walk-before.tar 2>/dev/null) bytes"

cd "$N" || exit 1
echo "=== re-derive with vector dump ==="
python3 derive-tail-walk.py || exit 1
diff --strip-trailing-cr test-reference.c tail-walk.c | grep -c "^<" | sed 's/^/gate lines missing from walker: /'

echo "=== build tail-walk ==="
gcc -std=c11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp tail-walk.c -lm -o bin/tail-walk || exit 1

echo "=== run tail-walk (france) ==="
rm -f /tmp/tail-final.f32
env -i PATH="$PATH" K3_INDEX="$ROOT/dataset/eqidx.bin" K3_TRUNKPATH=/root/k3trunk_i8/trunk.bin \
    K3_IDS=1008,10484,318,15383,387 K3_PREFETCH=0 K3_TRUNKRAM=0 OMP_NUM_THREADS=16 \
    OMP_PROC_BIND=close OMP_PLACES=cores \
    ./bin/tail-walk "$ROOT/dataset" france "$LEAVES" 2>&1 | grep -E "NORMALIZATION_PASS|differs|rror"
echo "dumped: $(stat -c%s /tmp/tail-final.f32 2>/dev/null) bytes (expect 143360 = 5 x 7168 x 4)"

cd "$S" || exit 1
echo "=== build head-walk ==="
gcc -std=c11 -O2 -march=native -fopenmp head-walk.c -lm -o bin/head-walk || exit 1

echo "=== run head-walk ==="
env -i PATH="$PATH" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./bin/head-walk "$HEAD" /tmp/tail-final.f32 5 2>&1

echo "=== done ==="
