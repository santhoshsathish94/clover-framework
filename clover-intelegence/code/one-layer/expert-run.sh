#!/bin/bash
# usage: expert-run.sh NTOKENS   (needs /tmp/one-layer/tok-1000.txt; writes /tmp/one-layer/expert-bench-N.out)
set -u
N=${1:-1000}
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
ulimit -l unlimited; ulimit -c 0
python3 - "$N" <<'PY'
import sys
n = int(sys.argv[1])
ids = open("/tmp/one-layer/tok-1000.txt").read().strip().split(",")[:n]
open("/tmp/one-layer/tok-%d.txt" % n, "w").write(",".join(ids) + "\n")
PY
gcc -std=gnu11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I$D/server one-layer.c -lm -ldl -lpthread -o one-layer 2> build-ol.log || { tail -20 build-ol.log; exit 1; }
if [ ! -s l1in-$N.resid.f32 ]; then
  env -i PATH="$PATH" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./one-layer $D tok-$N.txt warm l1in-$N
fi
gcc -std=gnu11 -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I$D/common -I$D/tansformers/transformer-1 expert-bench.c -lm -o expert-bench 2> build-eb.log || { tail -30 build-eb.log; exit 1; }
env -i PATH="$PATH" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./expert-bench $D/tansformers/transformer-1/bin/dataset l1in-$N.resid.f32 l1in-$N.s0.f32 $N 2>&1 | tee expert-bench-$N.out
