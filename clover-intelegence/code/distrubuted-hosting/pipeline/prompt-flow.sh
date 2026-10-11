#!/usr/bin/env bash
# Run several prompts through the resident fleet with vector tracing on, and record
# the residual norm entering every layer. One prompt at a time: a full trace is about
# 1.5 GB, so each is summarised and deleted before the next starts.
#
# Observation only. The request time printed by the fleet includes 1.5 GB of trace
# writes and is not a measurement of anything.
set -uo pipefail
ulimit -c 0
R=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
OUT=/tmp/prompt-flow.tsv
: > "$OUT"

run_one() {
    local name="$1" ids="$2"
    echo "=== $name : $ids ===" >&2
    rm -rf /tmp/vtrace; mkdir -p /tmp/vtrace
    cd "$R/pipeline" || return 1
    CLOVER_VECTORS=/tmp/vtrace/t OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 ${ids//,/ }' | ./bin/pipeline-serve $R serve" > "/tmp/flow-$name.log" 2>&1

    local answer
    answer=$(grep -o '\[.*\]' "/tmp/flow-$name.log" | head -1)
    echo "    answer: $answer" >&2

    # The residual entering each layer chains exactly: layer N's input is layer N-1's
    # output, so this is the trajectory with no stage-numbering assumptions in it.
    local line="$name"
    for L in $(seq 1 92); do
        local v
        v=$(/tmp/vector-walk "/tmp/vtrace/t.$L.bin" 4 2>/dev/null | awk '$1==0{printf "%.3f",$3; exit}')
        line="$line	${v:-NA}"
    done
    printf '%s\t%s\n' "$line" "$answer" >> "$OUT"
    rm -rf /tmp/vtrace
}

run_one france  "1008,10484,318,15383,387"
run_one japan   "1008,10484,318,10417,387"
run_one germany "1008,10484,318,16458,387"
run_one once    "13426,7625,261,1169,1459"
run_one thethe  "1008,276,276,276,276"
run_one water   "29349,154199,623,10113,16525"

echo "=== done, results in $OUT ===" >&2
