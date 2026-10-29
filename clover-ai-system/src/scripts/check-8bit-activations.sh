#!/bin/bash
# The recorded 1000-token prompt pass with 8-bit activations, one run after the other (never two runs at once):
#   1. experts only      (ACTIVATION_8BIT_EXPERT=1)
#   2. experts and trunk (ACTIVATION_8BIT_EXPERT=1 ACTIVATION_8BIT_TRUNK=1)
# The exact run to compare with is check-1000-token-prompt.sh (about 300 s, bit-identical to the pod).
# usage: check-8bit-activations.sh      (run detached; result in work/results/check-8bit-activations.result; the first run builds)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
ACTIVATION_8BIT_EXPERT=1 "$SCRIPTS_DIR/run-model.sh" recorded > check-8bit-experts.log 2>&1
ACTIVATION_8BIT_EXPERT=1 ACTIVATION_8BIT_TRUNK=1 BUILD=0 "$SCRIPTS_DIR/run-model.sh" recorded > check-8bit-experts-and-trunk.log 2>&1
{
  for m in experts experts-and-trunk; do
    echo "=== 8-bit activations: $m"
    grep -hE "build failed|ARITHMETIC|REQUEST|layer 0 |layers 2-92|next token id|against the pod" check-8bit-$m.log | cut -c1-300
  done
  echo "done $(date)"
} > check-8bit-activations.result 2>&1
