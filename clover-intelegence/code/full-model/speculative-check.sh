#!/bin/bash
# Plain decoding and speculative decoding of one prompt, one after the other (never two runs at once), then the generated ids compared.
# usage: speculative-check.sh PROMPT_FILE MAX_DRAFT NEW_TOKENS [USE_4BIT_TRUNK 0|1]       (run detached; result in <tag>.result)
cd /tmp/one-layer
FILE=$1; DRAFT=$2; DEC=$3; Q=${4:-0}
TAG=spec-$(basename "$FILE" .txt)-draft$DRAFT-4bit$Q
USE_4BIT_TRUNK=$Q PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$TAG-plain.ids ./full-run.sh text "$FILE" > $TAG-plain.log 2>&1
USE_4BIT_TRUNK=$Q SPECULATIVE_DRAFT_TOKENS=$DRAFT PF_NOBUILD=1 PF_DECODE=$DEC PF_CACHE=16 PF_SAVE_IDS=$TAG-spec.ids ./full-run.sh text "$FILE" > $TAG-spec.log 2>&1
{
  echo "prompt file $FILE, up to $DRAFT draft tokens, $DEC new tokens, USE_4BIT_TRUNK=$Q"
  for m in plain spec; do ./ids-to-words tokfiles $TAG-$m.ids $TAG-$m.out > /dev/null 2>&1; done
  if cmp -s $TAG-plain.ids $TAG-spec.ids; then echo "generated ids plain vs speculative: IDENTICAL"; else echo "generated ids plain vs speculative: DIFFERENT"; fi
  grep -h "build failed" $TAG-plain.log
  echo "--- plain";       grep -hE "ARITHMETIC|DECODE," $TAG-plain.log | cut -c1-220
  echo "--- speculative"; grep -hE "ARITHMETIC|DECODE,|SPECULATIVE" $TAG-spec.log | cut -c1-330
  echo "--- plain output:";       tr '\n' '~' < $TAG-plain.out; echo
  echo "--- speculative output:"; tr '\n' '~' < $TAG-spec.out; echo
  echo "done $(date)"
} > $TAG.result 2>&1
