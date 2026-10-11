#!/usr/bin/env bash
# Gives every stage its own dataset directory containing only what that stage opens.
# Without this each stage sees the whole 1.4 TB tree, which defeats the point of splitting.
set -euo pipefail
DATASET=${1:?usage: link-stage-datasets.sh DATASET_DIR [CODE_DIR]}
CODE=${2:-$(cd -- "$(dirname -- "$0")" && pwd)}
DATASET=$(cd -- "$DATASET" && pwd)

for required in tiktoken.model vocabulary.bin inputs/seed.bin outputs/fruit.bin \
                leaves.json trunk-0 operators/trunk-0-qkv; do
    [[ -e "$DATASET/$required" ]] || { echo "missing $DATASET/$required" >&2; exit 1; }
done

rm -rf "$CODE/server/bin/dataset"
mkdir -p "$CODE/server/bin/dataset/inputs" "$CODE/server/bin/dataset/outputs"
ln -sfn "$DATASET/tiktoken.model"    "$CODE/server/bin/dataset/tiktoken.model"
ln -sfn "$DATASET/vocabulary.bin"    "$CODE/server/bin/dataset/vocabulary.bin"
ln -sfn "$DATASET/inputs/seed.bin"   "$CODE/server/bin/dataset/inputs/seed.bin"
ln -sfn "$DATASET/outputs/fruit.bin" "$CODE/server/bin/dataset/outputs/fruit.bin"

rm -rf "$CODE/server/bin/dataset/trunk-0" "$CODE/server/bin/dataset/trunk-0-qkv" "$CODE/server/bin/dataset/leaves.json"
mkdir -p "$CODE/server/bin/dataset"
ln -sfn "$DATASET/trunk-0"                 "$CODE/server/bin/dataset/trunk-0"
ln -sfn "$DATASET/operators/trunk-0-qkv"   "$CODE/server/bin/dataset/trunk-0-qkv"
ln -sfn "$DATASET/leaves.json"             "$CODE/server/bin/dataset/leaves.json"

for layer in $(seq 1 92); do
    pod="$CODE/tansformers/transformer-$layer/bin"
    [[ -d "$DATASET/root-$layer" ]] || { echo "missing $DATASET/root-$layer" >&2; exit 1; }
    rm -rf "$pod/dataset"
    mkdir -p "$pod/dataset/operators/qkv-all"
    ln -sfn "$DATASET/root-$layer"                    "$pod/dataset/root-$layer"
    ln -sfn "$DATASET/trunk-$layer"                   "$pod/dataset/trunk-$layer"
    ln -sfn "$DATASET/operators/qkv-all/layer-$layer" "$pod/dataset/operators/qkv-all/layer-$layer"
done

printf 'PASS: per-stage datasets linked\n'
printf '  server      %8.3f GB\n' "$(du -sLb "$CODE/server/bin/dataset" | awk '{print $1/1073741824}')"
printf '  each pod    %8.3f GB\n' "$(du -sLb "$CODE/tansformers/transformer-1/bin/dataset" | awk '{print $1/1073741824}')"
