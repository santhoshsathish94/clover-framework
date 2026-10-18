#!/bin/bash
# The 34 recorded prompts through the whole model with the 8-bit-activation expert kernel in the PROMPT pass (LB_EXPK=3; decode steps use the
# exact kernel). Same as batch34.sh otherwise; output in p34a8/. Compare with: P34DIR=p34a8 python3 compare34.py
# usage: batch34-a8.sh [FIRST LAST]   (needs pipeline-full built by full-run.sh from the current layer-batch.c)
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
cd $H
mkdir -p p34a8
ulimit -l unlimited
for i in $(seq ${1:-1} ${2:-34}); do
  awk -F'\t' -v n=$i '$1==n{printf "%s", $4}' prompts.tsv > p34a8/$i.txt
  env -i PATH="$PATH" LB_IO=4 LB_SPEC=0 LB_THP=1 LB_EXPK=${A8_EXPK:-3} LB_A8T=${A8T:-} LB_A8T_GS=${A8T_GS:-} PF_DECODE=11 PF_SAVE_IDS=p34a8/$i.ids \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./pipeline-full $D $DATA tokfiles p34a8/$i.txt - > p34a8/$i.log 2>&1
  ./ids-to-words tokfiles p34a8/$i.ids p34a8/$i.out > /dev/null 2>&1
  echo "prompt $i done $(date +%H:%M:%S)"
done
echo "ALL DONE"
