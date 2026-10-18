#!/bin/bash
# Plain decoding and speculative decoding of one prompt, one after the other (never two runs at once), then the generated ids compared.
# usage: [REUSE_PLAIN=1] check-speculative-decoding.sh PROMPT_FILE MAX_DRAFT NEW_TOKENS [USE_4BIT_TRUNK 0|1]      (run detached; result in work/results/<tag>.result)
# REUSE_PLAIN=1 skips the plain run when its ids from an earlier run of the same prompt exist. The first run builds; later runs of this script do not.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
FILE=$(readlink -f "$1"); DRAFT=$2; DEC=$3; Q=${4:-0}
BASE=speculative-$(basename "$FILE" .txt)-4bit$Q
if [ -z "${REUSE_PLAIN:-}" ] || [ ! -s $BASE-plain.ids ]; then
  USE_4BIT_TRUNK=$Q PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$BASE-plain.ids "$SCRIPTS_DIR/run-model.sh" text "$FILE" > $BASE-plain.log 2>&1
fi
USE_4BIT_TRUNK=$Q SPECULATIVE_DRAFT_TOKENS=$DRAFT BUILD=0 PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$BASE-draft$DRAFT.ids "$SCRIPTS_DIR/run-model.sh" text "$FILE" > $BASE-draft$DRAFT.log 2>&1
{
  echo "prompt file $FILE, up to $DRAFT draft tokens, $DEC new tokens, USE_4BIT_TRUNK=$Q"
  for m in plain draft$DRAFT; do "$BUILD_DIR/ids-to-text" "$INPUTS_DIR/tokfiles" $BASE-$m.ids $BASE-$m.out > /dev/null 2>&1; done
  if cmp -s $BASE-plain.ids $BASE-draft$DRAFT.ids; then echo "generated ids plain vs speculative: IDENTICAL"; else echo "generated ids plain vs speculative: DIFFERENT"; fi
  grep -h "build failed" $BASE-plain.log
  echo "--- plain";       grep -hE "ARITHMETIC|DECODE," $BASE-plain.log | cut -c1-220
  echo "--- speculative"; grep -hE "ARITHMETIC|DECODE,|SPECULATIVE" $BASE-draft$DRAFT.log | cut -c1-330
  echo "--- plain output:";       tr '\n' '~' < $BASE-plain.out; echo
  echo "--- speculative output:"; tr '\n' '~' < $BASE-draft$DRAFT.out; echo
  echo "done $(date)"
} > $BASE-draft$DRAFT.result 2>&1
