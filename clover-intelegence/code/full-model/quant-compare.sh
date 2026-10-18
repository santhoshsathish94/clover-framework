#!/bin/bash
# Runs ONE prompt under the arithmetic modes, one after the other, and prints the outputs side by side.
# usage: [MODES="exact use_4bit_trunk"] [PREFIX=qc] [NOBUILD=1] [PF_CHAT=nothink] quant-compare.sh PROMPT_FILE [MAX_NEW_TOKENS]
# (needs full-run.sh and quant-compare.py in /tmp/one-layer; the first run builds unless NOBUILD=1). PF_CHAT / PF_SYSTEM pass through.
set -u
cd /tmp/one-layer
FILE=$1; DEC=${2:-24}; MODES=${MODES:-exact use_4bit_trunk}; PREFIX=${PREFIX:-qc}
first=1; [ -n "${NOBUILD:-}" ] && first=0
for m in $MODES; do
  nb=1; [ $first = 1 ] && nb=""
  USE_4BIT_TRUNK=$([ $m = use_4bit_trunk ] && echo 1 || echo 0) PF_NOBUILD=$nb PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$PREFIX-$m.ids ./full-run.sh text "$FILE" > $PREFIX-$m.log 2>&1
  first=0
done
python3 quant-compare.py single $PREFIX
