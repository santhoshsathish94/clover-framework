#!/bin/bash
# usage: sweep-run.sh NAME PROMPT_FILE DECODE CACHE K [K...]
# One normal decode of DECODE tokens (records the ids), then for each K a run that feeds the same continuation K tokens per pass.
# Logs: sw-NAME-base.log, sw-NAME-K.log in /tmp/one-layer. Runs one after the other (never in parallel: timings).
set -u
cd /tmp/one-layer
NAME=$1; PROMPT=$2; DEC=$3; C=$4; shift 4
if [ ! -s ids-$NAME.txt ]; then
  PF_DECODE=$DEC PF_CACHE=$C PF_SAVE_IDS=ids-$NAME.txt ./full-run.sh text $PROMPT > sw-$NAME-base.log 2>&1
fi
for K in "$@"; do
  PF_NOBUILD=1 PF_VERBOSE=1 PF_SWEEP=$K PF_SWEEP_IDS=ids-$NAME.txt PF_SWEEP_N=$DEC PF_CACHE=$C ./full-run.sh text $PROMPT > sw-$NAME-$K.log 2>&1
  grep -E "^SWEEP|distinct experts per layer per pass|the next token at each" sw-$NAME-$K.log
done
