#!/usr/bin/env bash
# Patch the server for tracing, rebuild it, and run its own reference gate with tracing
# off. The gate checks four sha256 bindings before it will run, so it also confirms the
# weights underneath are the ones it was written against.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
S=$CODE/simplification
R=/opt/clover-k3/clover-intelegence

echo "=== backup ==="
tar -cf /tmp/servertrace-before.tar -C "$CODE" server/server.c
echo "backup: $(stat -c%s /tmp/servertrace-before.tar) bytes"
sha256sum "$CODE/server/server.c"

echo "=== patch ==="
python3 "$S/port-server-trace.py" || exit 1

echo "=== rebuild the pipeline stage, same flags as build.sh ==="
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math \
    -fopenmp -fPIC -shared -DSERVER_NO_MAIN "$CODE/server/server.c" -lm \
    -o "$CODE/server/bin/pipeline-stage.so" || { echo "BUILD FAILED"; exit 1; }
echo built

echo "=== the server's own gate, tracing off ==="
"$CODE/server/test-reference.sh" \
    "$CODE/server/bin/dataset/trunk-0-qkv/qkv.bin" \
    "$CODE/server/bin/dataset/trunk-0" \
    "$R/dataset/eqidx.bin" \
    /root/k3trunk_i8/trunk.bin 2>&1 | tail -6

echo "=== done ==="
