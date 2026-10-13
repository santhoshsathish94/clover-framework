#!/usr/bin/env bash
# The cumulative sweep used one prompt. Re-run the two interesting cuts, sixteen and
# thirty-two layers in tolerance order, across all twelve prompts, because a cut chosen
# on france says nothing about french or nonsense until it is tried on them.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

TOLERANT=$(awk '/^ *[0-9]+ \[/ {print $(NF-1), $1}' /tmp/mc-sweep.log | sort -rg | awk '{printf "%s ", $2}')

for count in 16 32; do
    sel=/tmp/sel$count
    rm -rf "$sel"; mkdir -p "$sel"
    n=0
    for L in $TOLERANT; do
        [ $n -ge $count ] && break
        cp "/tmp/ca/t.$L.f32" "$sel/t.$L.f32" 2>/dev/null && n=$((n+1))
    done
    echo "=== n=$count, layers: $(ls "$sel" | sed 's/t\.//;s/\.f32//' | sort -n | tr '\n' ' ') ===" >&2
    /tmp/fidelity.sh "cut$count" CLOVER_CONST_ATTN="$sel/t"
    /tmp/extract.sh "cut$count" > /dev/null 2>&1
    /tmp/compare.sh "cut$count"
done
echo "=== done ==="
