#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
WORK="$ROOT/code/tansformers/remaining"
test ! -e "$WORK/metadata.tar"
files=()
for layer in $(seq 2 92); do
    test ! -e "$ROOT/code/tansformers/transformer-$layer"
    for relative in "trunk-$layer/manifest.json" "root-$layer/maps.json" \
        "root-$layer/build-verification.json" "operators/qkv-all/layer-$layer/manifest.json"; do
        test -f "$ROOT/dataset/$relative"
        files+=("$relative")
    done
done
tar -cf "$WORK/metadata.tar" -C "$ROOT/dataset" "${files[@]}"
printf 'PASS: 364 metadata files collected, no numeric payload copied\n'