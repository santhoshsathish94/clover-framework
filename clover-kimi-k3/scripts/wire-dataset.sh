#!/bin/bash
# Wires the dataset into the engine folder: the links under pod/ that the program opens at run time (layer 0 and layer 1 are built from pod sources that
# look for their files under pod/). Run it after the dataset is in place and again if the dataset moves. It changes nothing in the dataset.
# The original open weights (OPEN_WEIGHTS_DIR) need no wiring: the program is given that folder and reads the files by their own names.
# usage: [DATASET_DIR=...] wire-dataset.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
DATASET=$(cd "$DATASET_DIR" && pwd)
need() { [ -e "$DATASET/$1" ] || { echo "wire-dataset: missing $DATASET/$1"; exit 1; }; }
for f in trunk-0 trunk-1 root-1 operators/trunk-0-qkv/qkv.bin operators/qkv-all/layer-1; do need "$f"; done

SERVER="$POD_DIR/server/bin"
rm -rf "$SERVER/dataset" "$SERVER/configs"
mkdir -p "$SERVER/dataset"
ln -sfn "$DATASET/trunk-0" "$SERVER/dataset/trunk-0"
ln -sfn "$DATASET/operators/trunk-0-qkv" "$SERVER/dataset/trunk-0-qkv"

LAYER1="$POD_DIR/tansformers/transformer-1/bin/dataset"
rm -rf "$LAYER1"
mkdir -p "$LAYER1/operators/qkv-all"
ln -sfn "$DATASET/root-1" "$LAYER1/root-1"
ln -sfn "$DATASET/trunk-1" "$LAYER1/trunk-1"
ln -sfn "$DATASET/operators/qkv-all/layer-1" "$LAYER1/operators/qkv-all/layer-1"

broken=$(find "$POD_DIR" -xtype l | wc -l)
[ "$broken" = 0 ] || { echo "wire-dataset: $broken broken links"; find "$POD_DIR" -xtype l; exit 1; }
echo "wired: dataset $DATASET -> $POD_DIR (layer 0, layer 1), no broken links"
