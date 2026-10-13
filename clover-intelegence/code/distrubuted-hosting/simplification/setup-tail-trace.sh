#!/usr/bin/env bash
# Patch the tail (owner 93) with vector tracing, rebuild it, and prove nothing moved
# with tracing off. Same discipline every other patch in this sequence went through:
# the gate runs the real fleet and compares the answer, not a hash of the source.
set -uo pipefail
ulimit -c 0
C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
T=$C/tansformers/transformer-93

echo "=== backup before touching anything ==="
tar -cf /tmp/tailtrace-before.tar -C "$T" transformer-93.c 2>/dev/null
echo "backup: $(ls -l /tmp/tailtrace-before.tar 2>/dev/null | awk '{print $5}') bytes"

echo "=== patch ==="
python3 /tmp/port-tail-trace.py || { echo "PATCH FAILED"; exit 1; }

echo "=== confirm only additions ==="
tar -xf /tmp/tailtrace-before.tar -O transformer-93.c > /tmp/tail-before.c
diff /tmp/tail-before.c "$T/transformer-93.c" | grep -E "^<" && echo "REMOVED LINES ABOVE - investigate" || echo "no lines removed"

echo "=== build the tail only ==="
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function \
    -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
    -I"$C/common" "$T/transformer-93.c" -lm -o "$T/bin/pipeline-stage.so" \
    || { echo "BUILD FAILED"; exit 1; }
echo "BUILT OK"
