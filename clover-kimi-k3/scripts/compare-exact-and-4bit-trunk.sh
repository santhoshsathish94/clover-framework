#!/bin/bash
# Runs ONE prompt under the arithmetic modes, one after the other, and prints the outputs side by side.
# usage: [MODES="exact use_4bit_trunk"] [PREFIX=compare] [BUILD=0] [PF_CHAT=nothink] compare-exact-and-4bit-trunk.sh PROMPT_FILE [MAX_NEW_TOKENS]
# The first run builds unless BUILD=0. PF_CHAT / PF_SYSTEM pass through. Logs and ids land in work/results as <PREFIX>-<mode>.log / .ids.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
FILE=$(readlink -f "$1"); DEC=${2:-24}; MODES=${MODES:-exact use_4bit_trunk}; PREFIX=${PREFIX:-compare}
first=1; [ "${BUILD:-1}" = "0" ] && first=0
for m in $MODES; do
  build=0; [ $first = 1 ] && build=1
  USE_4BIT_TRUNK=$([ $m = use_4bit_trunk ] && echo 1 || echo 0) BUILD=$build PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$PREFIX-$m.ids "$SCRIPTS_DIR/run-model.sh" text "$FILE" > $PREFIX-$m.log 2>&1
  first=0
done
python3 "$ENGINE_DIR/tools/compare-outputs.py" single $PREFIX
