#!/bin/bash
# The recorded 1000-token prompt pass of the final version (exact): embeddings and layer 92 output must equal the pod's own chain, bit for bit,
# and the whole request takes about 300 s. (With USE_4BIT_TRUNK=1 the prompt pass is the same exact one; only decode steps differ.)
# usage: check-1000-token-prompt.sh      (run detached; result in work/results/check-1000-token-prompt.result)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
BUILD=0 "$SCRIPTS_DIR/run-model.sh" recorded > check-1000-exact.log 2>&1
{
  grep -hE "ARITHMETIC|one-time setup|REQUEST|layer 0 |layers 2-92|next token id|against the pod" check-1000-exact.log | cut -c1-260
  echo "done $(date)"
} > check-1000-token-prompt.result 2>&1
