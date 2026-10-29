#!/bin/bash
# Cost of a pass of K tokens through all layers, measured on a recorded continuation fed K tokens at a time (the cost a speculative draft of K-1 tokens pays).
# usage: measure-multi-token-pass.sh NAME PROMPT_FILE DECODE CACHE K [K...]
# One normal decode of DECODE tokens first (records the ids), then one run per K. Logs: work/results/pass-NAME-base.log, pass-NAME-K.log. One run at a time (timings).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
NAME=$1; PROMPT=$(readlink -f "$2"); DEC=$3; C=$4; shift 4
if [ ! -s ids-$NAME.txt ]; then
  PF_DECODE=$DEC PF_CACHE=$C PF_SAVE_IDS=ids-$NAME.txt "$SCRIPTS_DIR/run-model.sh" text $PROMPT > pass-$NAME-base.log 2>&1
fi
for K in "$@"; do
  BUILD=0 PF_VERBOSE=1 PF_SWEEP=$K PF_SWEEP_IDS=ids-$NAME.txt PF_SWEEP_N=$DEC PF_CACHE=$C "$SCRIPTS_DIR/run-model.sh" text $PROMPT > pass-$NAME-$K.log 2>&1
  grep -E "^SWEEP|distinct experts per layer per pass|the next token at each" pass-$NAME-$K.log
done
