#!/bin/bash
# Paths shared by every script in this folder. Source it: `. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"`.
# Every value can be overridden from the environment.

# The engine folder (this folder's parent): src, scripts, tools, prompts, docs.
ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Everything a run creates or reads: build/ (compiled program), inputs/ (recorded prompt, pod chain reference values), results/ (logs and outputs).
WORK_DIR="${WORK_DIR:-$ENGINE_DIR/work}"
# The model: root-N (experts, 15.7 GB per layer), trunk-N (the other weights), operators. What was precomputed for this engine; 1.4 TB.
DATASET_DIR="${DATASET_DIR:-$ENGINE_DIR/dataset}"
# The original open weights (moonshotai/Kimi-K3), only what the engine reads from them, under their original names: model-00094-of-000096.safetensors (the embedding table,
# the output head and the final normalisation tensors, 4.4 GiB), tiktoken.model, tokenizer_config.json. Listed with sizes and sha256 in scripts/open-weights-files.tsv.
OPEN_WEIGHTS_DIR="${OPEN_WEIGHTS_DIR:-$ENGINE_DIR/open-weights}"
# The 21 source files of the earlier pod tree the engine builds from (pod/, in this repo) plus, created by wire-dataset.sh and run-model.sh, the links into the dataset and the tail library the engine opens at run time.
POD_DIR="${POD_DIR:-$ENGINE_DIR/pod}"
# The five tokenizer headers (Apache-2.0, from kimi-k3-in-c at commit a2ad8e5, with licence) the words-to-embeddings stage is built from (third_party/, in this repo).
TOKENIZER_DIR="${TOKENIZER_DIR:-$ENGINE_DIR/third_party/kimi-k3-in-c}"
# The Kimi K3 DSpark draft checkpoint (7.1 GB).
DSPARK_DIR="${DSPARK_DIR:-$ENGINE_DIR/models/dspark}"

BUILD_DIR="$WORK_DIR/build"
INPUTS_DIR="$WORK_DIR/inputs"
RESULTS_DIR="$WORK_DIR/results"
SCRIPTS_DIR="$ENGINE_DIR/scripts"
mkdir -p "$BUILD_DIR" "$INPUTS_DIR" "$RESULTS_DIR"
