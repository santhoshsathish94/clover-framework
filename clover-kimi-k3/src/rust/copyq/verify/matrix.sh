#!/bin/bash
# Correctness matrix (cpp reference vs rust) followed by the negative controls. Log: out/matrix.log
cd "$(dirname "$0")"
L=out/matrix.log; : > $L
run() { bash one.sh "$@" >> $L 2>&1; }
for s in 1 2 3 4 5; do for t in cpp rust; do run $t S1 $s; done; done
for s in 1 2 3; do for t in rust cpp; do run $t S2 $s; done; done
for s in 1 2; do for t in cpp rust; do run $t S3 $s; done; done
for t in cpp rust; do run $t S5 1; done
for s in 1 2 3; do for t in cpp rust; do run $t S4 $s; done; done
echo CORRECTNESS-DONE >> $L
for s in 1 2 3; do run M2 S1 $s; run M2v S1 $s; done
run M2 S2 1; run M4 S1 1; run M4 S2 1; run M3 S1 1
for s in 1 2 3; do run M3 S3 $s; done
for s in $(seq 1 20); do run M1 S1 $s; done
run M1 S4 1
echo DONE >> $L
