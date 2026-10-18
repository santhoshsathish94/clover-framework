#!/bin/bash
# Plain decoding against speculative decoding with the DSpark drafter (8 draft tokens) on three prompts, one run after the other (never two at once):
# the generated ids must be identical to plain decoding; the speedup and the share of draft tokens kept are printed.
# usage: check-dspark-drafter.sh [NEW_TOKENS]      (run detached; summary in work/results/check-dspark-drafter.result)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
DEC=${1:-48}
for prompt in code-is-prime prose repeating-numbers; do
  SPECULATIVE_DRAFTER=dspark REUSE_PLAIN=1 "$SCRIPTS_DIR/check-speculative-decoding.sh" "$ENGINE_DIR/prompts/$prompt.txt" 8 $DEC 0
done
{
  for prompt in code-is-prime prose repeating-numbers; do
    echo "=== $prompt"
    grep -hE "IDENTICAL|DIFFERENT|^DECODE|SPECULATIVE" speculative-$prompt-4bit0-dspark-draft8.result | cut -c1-330
  done
  echo "done $(date)"
} > check-dspark-drafter.result 2>&1
