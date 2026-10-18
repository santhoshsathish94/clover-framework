#!/bin/bash
# Second worker for the 34-prompt run: takes the prompts from the high end (34 downwards) while batch34.sh works from the low end,
# and stops the low-end worker as soon as it has finished prompt $1 - 1, so no prompt is run twice.
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
cd $H
ulimit -l unlimited
FIRST=$1
(
  while ! grep -q "prompt $((FIRST - 1)) done" batch34.log; do sleep 3; done
  pkill -P "$(pgrep -f 'bash batch34.sh 1 34' | head -1)" -f pipeline-full 2>/dev/null
  kill "$(pgrep -f 'bash batch34.sh 1 34' | head -1)" 2>/dev/null
  echo "low-end worker stopped after prompt $((FIRST - 1))" >> batch34b.log
) &
for i in $(seq 34 -1 $FIRST); do
  awk -F'\t' -v n=$i '$1==n{printf "%s", $4}' prompts.tsv > p34/$i.txt
  env -i PATH="$PATH" LB_IO=4 LB_SPEC=0 LB_THP=1 LB_EXPK=1 PF_DECODE=11 PF_SAVE_IDS=p34/$i.ids \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./pipeline-full $D $DATA tokfiles p34/$i.txt - > p34/$i.log 2>&1
  ./ids-to-words tokfiles p34/$i.ids p34/$i.out > /dev/null 2>&1
  echo "prompt $i done $(date +%H:%M:%S)" >> batch34b.log
done
wait
echo "ALL DONE" >> batch34b.log
