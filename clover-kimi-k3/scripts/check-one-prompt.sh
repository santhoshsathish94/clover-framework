#!/bin/bash
# One recorded prompt under exact and USE_4BIT_TRUNK=1 with the built program, compared with the earlier results of the same prompt:
# the exact ids must equal recorded-34-exact/N.ids (the bit-identical path); the 4-bit ids are shown against recorded-34-use_4bit_trunk/N.ids.
# usage: check-one-prompt.sh N      (N = 1-34; run detached; result in work/results/check-one-prompt.out)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
N=${1:?prompt number 1-34}
cd "$RESULTS_DIR"
awk -F'\t' -v n=$N '$1==n{printf "%s", $4}' "$ENGINE_DIR/prompts/recorded-34-prompts.tsv" > check-prompt.txt
{
  echo "prompt $N: $(cat check-prompt.txt)"
  BUILD=0 MODES="exact use_4bit_trunk" PREFIX=check "$SCRIPTS_DIR/compare-exact-and-4bit-trunk.sh" check-prompt.txt 11 > /dev/null 2>&1
  for m in exact use_4bit_trunk; do "$BUILD_DIR/ids-to-text" "$INPUTS_DIR/tokfiles" check-$m.ids check-$m.out > /dev/null 2>&1; done
  echo "exact output (this build)    : $(tr '\n' '~' < check-exact.out)"
  echo "exact output (earlier)       : $(tr '\n' '~' < recorded-34-exact/$N.out)"
  if cmp -s check-exact.ids recorded-34-exact/$N.ids; then echo "exact ids identical to the earlier exact run: YES"; else echo "exact ids identical to the earlier exact run: NO"; fi
  echo "4-bit output (this build)    : $(tr '\n' '~' < check-use_4bit_trunk.out)"
  echo "4-bit output (earlier build) : $(tr '\n' '~' < recorded-34-use_4bit_trunk/$N.out)"
  if cmp -s check-use_4bit_trunk.ids recorded-34-use_4bit_trunk/$N.ids; then echo "4-bit ids identical to the earlier 4-bit build: YES"; else echo "4-bit ids identical to the earlier 4-bit build: NO"; fi
  grep -hE "ARITHMETIC|DECODE," check-exact.log check-use_4bit_trunk.log | cut -c1-200
  echo "done $(date)"
} > check-one-prompt.out 2>&1
