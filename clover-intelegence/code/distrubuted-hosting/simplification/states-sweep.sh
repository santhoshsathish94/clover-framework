#!/usr/bin/env bash
# Capture each prompt, keep its 13 MB of states, delete its 1.6 GB dump, move on.
#
# The disk holds one dump at a time, so the loop is capture, extract, delete, and the
# peak cost is a single dump whatever the number of prompts.
#
# japan and germany differ from france in one token out of five. They are in the list on
# purpose: if the measurement works, a near-duplicate prompt must add almost no new
# directions while an unrelated one adds many. They are the positive control.
#
# usage: states-sweep.sh OUT-DIR
set -u
C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
OUT=${1:?out dir}
mkdir -p "$OUT"

PROMPTS="
france:1008 10484 318 15383 387
japan:1008 10484 318 10417 387
germany:1008 10484 318 16458 387
water:29349 154199 623 10113 16525
once:13426 7625 261 1169 1459
thethe:1008 276 276 276 276
question:5376 387 276 12156 6664 318 94916
nonsense:36324 291 79896 275 47369 304 57045 350 60142
"

echo "$PROMPTS" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    name=${line%%:*}
    ids=${line#*:}
    [ -f "$OUT/$name.npy" ] && { echo "$name already have"; continue; }
    rm -rf /tmp/sweep
    mkdir -p /tmp/sweep
    cd "$C/pipeline" || exit 1
    answer=$(env CLOVER_VECTORS=/tmp/sweep/v OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo \"1 $ids\" | ./bin/pipeline-serve $C serve" 2>&1 | grep '^REQUEST')
    echo "$name  $answer"
    python3 /tmp/extract-states.py /tmp/sweep/v "$OUT/$name.npy" || exit 1
    rm -rf /tmp/sweep
    df -h /tmp | tail -1
done

touch "$OUT/.done"
echo "sweep complete"
