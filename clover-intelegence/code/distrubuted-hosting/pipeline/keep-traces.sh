#!/usr/bin/env bash
# Keep the traces for a few prompts at once so they can be compared directionally.
# Norms turned out to be nearly prompt independent; the angle between prompts is
# where the content shows, so this keeps the dumps instead of summarising and deleting.
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

run_one() {
    local name="$1" ids="$2"
    echo "=== $name ===" >&2
    rm -rf "/tmp/keep-$name"; mkdir -p "/tmp/keep-$name"
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS="/tmp/keep-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 ${ids//,/ }' | ./bin/pipeline-serve $R serve" > "/tmp/keep-$name.log" 2>&1
    grep -o '\[.*\]' "/tmp/keep-$name.log" | head -1 >&2
}

run_one france "1008,10484,318,15383,387"
run_one japan  "1008,10484,318,10417,387"
run_one water  "29349,154199,623,10113,16525"

echo "=== disk ===" >&2
du -sh /tmp/keep-* >&2
echo "=== done ===" >&2
