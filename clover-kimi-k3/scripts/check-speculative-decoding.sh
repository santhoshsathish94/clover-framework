#!/bin/bash
# Plain decoding and speculative decoding of one prompt, one after the other (never two runs at once), then the generated ids compared.
# usage: [SPECULATIVE_DRAFTER=dspark] [SPECULATIVE_FIXED_WIDTH=1] [REUSE_PLAIN=1] check-speculative-decoding.sh PROMPT_FILE MAX_DRAFT NEW_TOKENS [USE_4BIT_TRUNK 0|1]      (run detached; result in work/results/<tag>.result)
# REUSE_PLAIN=1 skips the plain run when its ids from an earlier run of the same prompt exist. The first run builds; later runs of this script do not.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
FILE=$(readlink -f "$1"); DRAFT=$2; DEC=$3; Q=${4:-0}; DRAFTER=${SPECULATIVE_DRAFTER:-lookup}; TAG=$DRAFTER${SPECULATIVE_FIXED_WIDTH:+-fixed}
BASE=speculative-$(basename "$FILE" .txt)-4bit$Q
if [ -z "${REUSE_PLAIN:-}" ] || [ ! -s $BASE-plain.ids ]; then
  SPECULATIVE_DRAFTER= USE_4BIT_TRUNK=$Q PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$BASE-plain.ids "$SCRIPTS_DIR/run-model.sh" text "$FILE" > $BASE-plain.log 2>&1
fi
USE_4BIT_TRUNK=$Q SPECULATIVE_DRAFT_TOKENS=$DRAFT BUILD=0 PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$BASE-$TAG-draft$DRAFT.ids "$SCRIPTS_DIR/run-model.sh" text "$FILE" > $BASE-$TAG-draft$DRAFT.log 2>&1
{
  echo "prompt file $FILE, up to $DRAFT draft tokens, $DEC new tokens, USE_4BIT_TRUNK=$Q"
  for m in plain $TAG-draft$DRAFT; do "$BUILD_DIR/ids-to-text" "$OPEN_WEIGHTS_DIR" $BASE-$m.ids $BASE-$m.out > /dev/null 2>&1; done
  if cmp -s $BASE-plain.ids $BASE-$TAG-draft$DRAFT.ids; then echo "generated ids plain vs speculative: IDENTICAL"; else echo "generated ids plain vs speculative: DIFFERENT"; fi
  grep -h "build failed" $BASE-plain.log
  echo "--- plain";       grep -hE "ARITHMETIC|DECODE," $BASE-plain.log | cut -c1-220
  echo "--- speculative"; grep -hE "ARITHMETIC|DECODE,|SPECULATIVE|DSpark draft" $BASE-$TAG-draft$DRAFT.log | cut -c1-330
  echo "--- plain output:";       tr '\n' '~' < $BASE-plain.out; echo
  echo "--- speculative output:"; tr '\n' '~' < $BASE-$TAG-draft$DRAFT.out; echo
  echo "done $(date)"
} > $BASE-$TAG-draft$DRAFT.result 2>&1
