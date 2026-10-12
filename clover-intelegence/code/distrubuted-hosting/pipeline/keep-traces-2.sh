#!/usr/bin/env bash
# Two more prompts to test the position-0 convergence seen between france and water.
#   once   - a different first token, so it tests the claim
#   thethe - the same first token as france, so position 0 must be bit identical
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

run_one() {
    local name="$1" ids="$2"
    echo "=== $name ===" >&2
    rm -rf "/tmp/keep-$name"; mkdir -p "/tmp/keep-$name"
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS="/tmp/keep-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $R serve" > "/tmp/keep-$name.log" 2>&1
    grep -o '\[.*\]' "/tmp/keep-$name.log" | head -1 >&2
}

run_one once   "13426 7625 261 1169 1459"
run_one thethe "1008 276 276 276 276"
echo "=== done ===" >&2
