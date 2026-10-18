#!/bin/bash
# usage: batch-run.sh NTOKENS WINDOW[,WINDOW...]   (needs l1in-1000.{resid,s0}.f32 from expert-run.sh)
set -u
N=${1:-128}; W=${2:-64}
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
ulimit -l unlimited; ulimit -c 0
gcc -std=gnu11 ${LB_OPT:--O3} -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function ${EXTRA:-} \
    -I$D/common -I$D/tansformers/transformer-1 layer-batch.c -lm -o layer-batch${SUFFIX:-} 2> build-lb.log || { grep -E "error" build-lb.log | head -30; exit 1; }
grep -c warning build-lb.log | sed 's/^/build warnings: /'
env -i PATH="$PATH" LB_WIDE="${LB_WIDE:-1}" LB_VEC="${LB_VEC:-1}" LB_ROUTE="${LB_ROUTE:-1}" LB_EXPK="${LB_EXPK:-1}" LB_PRED="${LB_PRED:-0}" LB_LIN="${LB_LIN:-0}" LB_DUMP="${LB_DUMP:-}" LB_THP="${LB_THP:-1}" LB_SAVE_REF="${LB_SAVE_REF:-}" LB_IOLO="${LB_IOLO:-}" LB_IOHI="${LB_IOHI:-}" LB_DISK="${LB_DISK:-0}" LB_IO="${LB_IO:-8}" LB_SPEC="${LB_SPEC:-896}" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./layer-batch${SUFFIX:-} $D/tansformers/transformer-1/bin/dataset l1in-1000.resid.f32 l1in-1000.s0.f32 $N $W 2>&1 | tee batch-$N-$W${SUFFIX:-}.out
