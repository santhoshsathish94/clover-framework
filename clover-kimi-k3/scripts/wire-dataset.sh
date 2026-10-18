#!/bin/bash
# Wires the dataset into the engine folder: the links under pod/ that the program opens at run time, the tokenizer files under work/inputs,
# and the tokenizer config. Run it after the dataset is in place and again if the dataset moves. It changes nothing in the dataset.
# usage: [DATASET_DIR=...] wire-dataset.sh
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
DATASET=$(cd "$DATASET_DIR" && pwd)
need() { [ -e "$DATASET/$1" ] || { echo "wire-dataset: missing $DATASET/$1"; exit 1; }; }
for f in tiktoken.model vocabulary.bin leaves.json inputs/seed.bin outputs/fruit.bin trunk-0 trunk-1 root-1 operators/trunk-0-qkv/qkv.bin operators/qkv-all/layer-1; do need "$f"; done

SERVER="$POD_DIR/server/bin"
rm -rf "$SERVER/dataset"
mkdir -p "$SERVER/dataset/inputs" "$SERVER/dataset/outputs" "$SERVER/configs"
for f in tiktoken.model vocabulary.bin leaves.json trunk-0; do ln -sfn "$DATASET/$f" "$SERVER/dataset/$f"; done
ln -sfn "$DATASET/operators/trunk-0-qkv" "$SERVER/dataset/trunk-0-qkv"
for f in inputs/seed.bin inputs/seed.bin.direct outputs/fruit.bin outputs/fruit.bin.direct; do
  [ -e "$DATASET/$f" ] && ln -sfn "$DATASET/$f" "$SERVER/dataset/$f"
done
cp "$ENGINE_DIR/assets/tokenizer_config.json" "$SERVER/configs/tokenizer_config.json"

LAYER1="$POD_DIR/tansformers/transformer-1/bin/dataset"
rm -rf "$LAYER1"
mkdir -p "$LAYER1/operators/qkv-all"
ln -sfn "$DATASET/root-1" "$LAYER1/root-1"
ln -sfn "$DATASET/trunk-1" "$LAYER1/trunk-1"
ln -sfn "$DATASET/operators/qkv-all/layer-1" "$LAYER1/operators/qkv-all/layer-1"

rm -rf "$INPUTS_DIR/tokfiles"
mkdir -p "$INPUTS_DIR/tokfiles"
ln -sfn "$DATASET/tiktoken.model" "$INPUTS_DIR/tokfiles/tiktoken.model"
ln -sfn "$ENGINE_DIR/assets/tokenizer_config.json" "$INPUTS_DIR/tokfiles/tokenizer_config.json"

broken=$(find "$POD_DIR" "$INPUTS_DIR/tokfiles" -xtype l | wc -l)
[ "$broken" = 0 ] || { echo "wire-dataset: $broken broken links"; find "$POD_DIR" "$INPUTS_DIR/tokfiles" -xtype l; exit 1; }
echo "wired: dataset $DATASET -> $POD_DIR (server, layer 1) and $INPUTS_DIR/tokfiles, no broken links"
