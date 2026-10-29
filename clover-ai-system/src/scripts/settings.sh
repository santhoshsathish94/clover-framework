#!/bin/bash
# Paths shared by every script in this folder. Source it: `. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"`.
# Every value can be overridden from the environment.

# The engine folder: README.md, docs/, src/.
ENGINE_DIR="${ENGINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# All code (code/, with code/third-party) and everything it works with.
SRC_DIR="$ENGINE_DIR/src"
CODE_DIR="$SRC_DIR/code"
SCRIPTS_DIR="$SRC_DIR/scripts"
TOOLS_DIR="$SRC_DIR/tools"
PROMPTS_DIR="$SRC_DIR/prompts"
# The data: datasets/third-party (made by others) and datasets/computed (computed for this engine).
DATASETS_DIR="$SRC_DIR/datasets"
# Everything a run creates or reads: build/ (compiled program), inputs/ (recorded prompt, pod chain reference values), results/ (logs and outputs).
WORK_DIR="${WORK_DIR:-$SRC_DIR/work}"
# Computed for this engine: expert-N (15.7 GB per layer), trunk-N (the other weights, only the records the engine reads, and operator.bin, the Q/K/V weights). 1.4 TB.
DATASET_DIR="${DATASET_DIR:-$DATASETS_DIR/computed}"
# Third party: the original open weights (moonshotai/Kimi-K3), only what the engine reads from them, under their original names: model-00094-of-000096.safetensors (the embedding
# table, the output head and the final normalisation tensors, 4.4 GiB), tiktoken.model, tokenizer_config.json. Listed with sizes and sha256 in scripts/open-weights-files.tsv.
OPEN_WEIGHTS_DIR="${OPEN_WEIGHTS_DIR:-$DATASETS_DIR/third-party/kimi-k3-open-weights}"
# Third party: the Kimi K3 DSpark draft checkpoint (Inferact/Kimi-K3-DSpark, 7.1 GB).
DSPARK_DIR="${DSPARK_DIR:-$DATASETS_DIR/third-party/kimi-k3-dspark}"
# Clover Kimi K3 + HybridAgent + NCM unified checkpoints and evaluation datasets (downloaded from Santhoshsathish94/clover-kimi-k3/hybrid-ncm-k3).
HYBRID_NCM_DIR="${HYBRID_NCM_DIR:-$DATASETS_DIR/hybrid-ncm-k3}"
# Third party code: the five tokenizer headers (Apache-2.0, from kimi-k3-in-c at commit a2ad8e5, with licence) the words-to-embeddings stage is built from.
TOKENIZER_DIR="${TOKENIZER_DIR:-$CODE_DIR/third-party/kimi-k3-in-c}"

BUILD_DIR="$WORK_DIR/build"
INPUTS_DIR="$WORK_DIR/inputs"
RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$BUILD_DIR" "$INPUTS_DIR" "$RESULTS_DIR"