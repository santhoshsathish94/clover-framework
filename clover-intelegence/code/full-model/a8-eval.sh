#!/bin/bash
# 8-bit-activation evaluation: (1) rebuild, (2) the recorded 1000-token prompt with LB_EXPK=3 and with the exact kernel (timing + next tokens),
# (3) the 34 recorded prompts with LB_EXPK=3 in two parallel halves (fidelity only, timings from those runs are not meaningful).
H=/tmp/one-layer
cd $H
PF_NOBUILD= ./full-run.sh text sw-prose.txt > a8-build.log 2>&1
for K in 3 1; do
  LB_EXPK=$K PF_NOBUILD=1 PF_DECODE=11 PF_CACHE=24 ./full-run.sh recorded > a8-rec-k$K.log 2>&1
done
./batch34-a8.sh 1 17 > a8-b1.log 2>&1 &
./batch34-a8.sh 18 34 > a8-b2.log 2>&1 &
wait
echo "A8 EVAL DONE" > a8-done.txt
