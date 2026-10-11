#!/bin/bash
# What each pod has to hold. Measured from the files it opens, not estimated.
# stat per known file rather than du over a cold 1.6 TB tree.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset

size() { stat -Lc %s "$1" 2>/dev/null || echo 0; }

echo "=== one layer directory, for reference ==="
ls -lL "$DATA/root-46/" 2>/dev/null | awk '{print $5, $9}'

echo
echo "=== per layer, excluding experts ==="
for n in 1 2 46 92; do
    total=0
    for f in "$DATA/root-$n"/*; do
        case "$f" in *experts.direct) continue;; esac
        total=$((total + $(size "$f")))
    done
    printf "layer %-3s other files %12d bytes\n" "$n" "$total"
done

echo
echo "=== server pod ==="
printf "trunk-0 qkv        %12d\n" "$(size "$CODE/server/bin/dataset/trunk-0-qkv/qkv.bin")"
t=0; for f in "$CODE/server/bin/dataset/trunk-0"/*; do t=$((t + $(size "$f"))); done
printf "trunk-0 records    %12d\n" "$t"
printf "seed.bin.direct    %12d\n" "$(size "$DATA/inputs/seed.bin.direct")"

echo
echo "=== layer 93 pod ==="
printf "leaves.json        %12d\n" "$(size "$DATA/leaves.json")"
printf "fruit.bin.direct   %12d\n" "$(size "$DATA/outputs/fruit.bin.direct")"
printf "tiktoken.model     %12d\n" "$(size "$CODE/server/bin/dataset/tiktoken.model")"
printf "vocabulary.bin     %12d\n" "$(size "$CODE/server/bin/dataset/vocabulary.bin")"

echo
echo "=== experts, every layer ==="
find "$DATA" -name experts.direct -printf '%s\n' | awk '{s+=$1; n++} END {printf "%d layers, %d bytes each, %.1f GB total\n", n, s/n, s/1e9}'

echo
echo "=== per request, per layer, conversation state ==="
grep -n 'TransformerSequence\|sizeof' "$CODE/pipeline/CONTEXT.md" 2>/dev/null | grep -i '7,771,448\|7.41' | head -2
echo "(TransformerSequence measured earlier at 7,771,448 bytes)"
