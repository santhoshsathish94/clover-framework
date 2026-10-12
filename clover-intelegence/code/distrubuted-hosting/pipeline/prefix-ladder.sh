#!/usr/bin/env bash
# A ladder of nested prompts: "The", then "The capital", and so on up to five tokens.
# Each is a strict prefix of the next, so if the model is causal every position must be
# identical across every run that contains it. That is a prediction, not an assumption,
# and the point of the ladder is to test it before reading anything else from the traces.
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

run_one() {
    local name="$1" ids="$2"
    echo "=== $name : $ids ===" >&2
    rm -rf "/tmp/keep-$name"; mkdir -p "/tmp/keep-$name"
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS="/tmp/keep-$name/t" OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $ids' | ./bin/pipeline-serve $R serve" > "/tmp/keep-$name.log" 2>&1
    grep -o '\[.*\]' "/tmp/keep-$name.log" | head -1 >&2
}

run_one n1 "1008"
run_one n2 "1008 10484"
run_one n3 "1008 10484 318"
run_one n4 "1008 10484 318 15383"
echo "=== done, n5 is the existing france trace ===" >&2
df -h / | tail -1 >&2
