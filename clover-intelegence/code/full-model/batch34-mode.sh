#!/bin/bash
# The 34 recorded prompts under one arithmetic mode, one process per prompt, one prompt at a time; output in the directory quant-compare.py expects
# (exact -> p34, use_4bit_trunk -> p34_4bit_trunk: the whole trunk 4-bit, USE_4BIT_TRUNK=1). Older runs p34a8 (8-bit activations) and p34q4 (decode-only 4-bit, layers 2-92) stay for comparison.
# usage: batch34-mode.sh MODE [FIRST LAST]      (uses the built ./pipeline-full; do not rebuild while it runs)
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
cd $H
MODE=${1:?mode}
case $MODE in
  exact)   DIR=p34;     E="LB_EXPK=1" ;;
  use_4bit_trunk) DIR=p34_4bit_trunk; E="LB_EXPK=1 USE_4BIT_TRUNK=1 CLOVER_LOCK_TRUNK=0 PF_CACHE=16" ;;
  *) echo "mode must be exact or use_4bit_trunk"; exit 2 ;;
esac
mkdir -p $DIR
ulimit -l unlimited
for i in $(seq ${2:-1} ${3:-34}); do
  awk -F'\t' -v n=$i '$1==n{printf "%s", $4}' prompts.tsv > $DIR/$i.txt
  env -i PATH="$PATH" LB_IO=4 LB_SPEC=0 LB_THP=1 $E PF_DECODE=11 PF_SAVE_IDS=$DIR/$i.ids \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./pipeline-full $D $DATA tokfiles $DIR/$i.txt - > $DIR/$i.log 2>&1
  ./ids-to-words tokfiles $DIR/$i.ids $DIR/$i.out > /dev/null 2>&1
  echo "prompt $i done $(date +%H:%M:%S)"
done
echo "ALL DONE"
