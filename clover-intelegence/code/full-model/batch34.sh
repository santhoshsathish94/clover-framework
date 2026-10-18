#!/bin/bash
# Runs each of the 34 prompts in prompts.tsv through the whole-model process (prompt pass + 11 decode steps = 12 new tokens), one process per prompt.
# Then compare34.py compares (a) our token ids for the prompt text with the recorded ids and (b) our 12 generated tokens, as text, with
# the 12 tokens recorded in results.md.
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
cd $H
mkdir -p p34
ulimit -l unlimited
for i in $(seq ${1:-1} ${2:-34}); do
  awk -F'\t' -v n=$i '$1==n{printf "%s", $4}' prompts.tsv > p34/$i.txt
  env -i PATH="$PATH" LB_IO=4 LB_SPEC=0 LB_THP=1 LB_EXPK=1 PF_DECODE=11 PF_SAVE_IDS=p34/$i.ids \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./pipeline-full $D $DATA tokfiles p34/$i.txt - > p34/$i.log 2>&1
  ./ids-to-words tokfiles p34/$i.ids p34/$i.out > /dev/null 2>&1
  echo "prompt $i done $(date +%H:%M:%S)"
done
echo "ALL DONE"
