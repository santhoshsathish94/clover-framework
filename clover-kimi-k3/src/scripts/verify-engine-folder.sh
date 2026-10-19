#!/bin/bash
# Checks a fresh engine folder end to end, one step after the other: build, then one recorded prompt exact and with USE_4BIT_TRUNK=1 against the earlier results.
# usage: [DATASET_DIR=...] verify-engine-folder.sh [PROMPT_NUMBER]      (run detached; progress in work/results/verify-engine-folder.log)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
N=${1:-4}
{
  echo "start $(date); dataset: $DATASET_DIR"
  BUILD_ONLY=1 "$SCRIPTS_DIR/run-model.sh" text "$PROMPTS_DIR/code-is-prime.txt"
  echo "build finished rc=$? $(date)"
  "$SCRIPTS_DIR/check-one-prompt.sh" $N
  echo "check finished $(date)"
  cat "$RESULTS_DIR/check-one-prompt.out"
} > "$RESULTS_DIR/verify-engine-folder.log" 2>&1
