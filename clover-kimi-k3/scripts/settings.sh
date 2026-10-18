#!/bin/bash
# Paths shared by every script in this folder. Source it: `. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"`.
# Every value can be overridden from the environment.

# The engine folder (this folder's parent): src, scripts, tools, prompts, docs.
ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Everything a run creates or reads: build/ (compiled program), inputs/ (recorded prompt, tokenizer files, pod reference values), results/ (logs and outputs).
WORK_DIR="${WORK_DIR:-$ENGINE_DIR/work}"
# The model: root-N (experts, 15.7 GB per layer), trunk-N (the other weights), operators, tokenizer files. 1.4 TB.
DATASET_DIR="${DATASET_DIR:-$ENGINE_DIR/dataset}"
# The deployed per-layer pod sources and configs the engine reuses at build time and for the final head (server, common, tansformers/transformer-N). Read only.
POD_DIR="${POD_DIR:-/opt/clover-k3/clover-intelegence/code/distrubuted-hosting}"
# The C tokenizer the words-to-embeddings stage is built from. Read only.
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/kimi-k3-in-c}"
# The Kimi K3 DSpark draft checkpoint (7.1 GB).
DSPARK_DIR="${DSPARK_DIR:-$ENGINE_DIR/models/dspark}"

BUILD_DIR="$WORK_DIR/build"
INPUTS_DIR="$WORK_DIR/inputs"
RESULTS_DIR="$WORK_DIR/results"
SCRIPTS_DIR="$ENGINE_DIR/scripts"
mkdir -p "$BUILD_DIR" "$INPUTS_DIR" "$RESULTS_DIR"
