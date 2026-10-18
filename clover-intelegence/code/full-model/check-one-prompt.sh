#!/bin/bash
# One recorded prompt under exact and USE_4BIT_TRUNK=1 with the built binary, compared with the earlier results of the same prompt:
# the exact ids must equal p34/N.ids (bit-identical path); the 4-bit ids are shown against p34_4bit_trunk_previous/N.ids (earlier 4-bit build).
# usage: check-one-prompt.sh N      (run detached; result in check-one-prompt.out)
N=${1:?prompt number 1-34}
cd /tmp/one-layer
awk -F'\t' -v n=$N '$1==n{printf "%s", $4}' prompts.tsv > check-prompt.txt
{
  echo "prompt $N: $(cat check-prompt.txt)"
  NOBUILD=1 MODES="exact use_4bit_trunk" PREFIX=check ./quant-compare.sh check-prompt.txt 11 > /dev/null 2>&1
  for m in exact use_4bit_trunk; do ./ids-to-words tokfiles check-$m.ids check-$m.out > /dev/null 2>&1; done
  echo "exact output (this build)    : $(tr '\n' '~' < check-exact.out)"
  echo "exact output (earlier, p34)  : $(tr '\n' '~' < p34/$N.out)"
  if cmp -s check-exact.ids p34/$N.ids; then echo "exact ids identical to the earlier exact run: YES"; else echo "exact ids identical to the earlier exact run: NO"; fi
  echo "4-bit output (this build)    : $(tr '\n' '~' < check-use_4bit_trunk.out)"
  echo "4-bit output (earlier build) : $(tr '\n' '~' < p34_4bit_trunk_previous/$N.out)"
  if cmp -s check-use_4bit_trunk.ids p34_4bit_trunk_previous/$N.ids; then echo "4-bit ids identical to the earlier 4-bit build: YES"; else echo "4-bit ids identical to the earlier 4-bit build: NO"; fi
  grep -hE "ARITHMETIC|DECODE," check-exact.log check-use_4bit_trunk.log | cut -c1-200
  echo "done $(date)"
} > check-one-prompt.out 2>&1
