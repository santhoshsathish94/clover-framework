#!/bin/sh
set -eu
work=$1
stage=$2
cd "$work"
test ! -e results.json
test ! -e france.bin
test ! -e japan.bin
deps=/opt/clover-k3/reverse-integration-20260929-a/deps
gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -DREVERSE_STAGE="$stage" \
    -I"$deps/gmp/usr/include/x86_64-linux-gnu" -o candidate candidate.c \
    "$deps/gmp/usr/lib/x86_64-linux-gnu/libgmp.a" -lm
. /opt/clover-k3/config.env
export OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
export K3_PREFETCH=4 K3_TRUNKRAM=0 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1
export K3_INDEX=/opt/clover-k3/build/eqidx.bin
joint=0
if test "$stage" -le 11; then joint=1; fi
for name in france japan; do
    ids=1008,10484,318,15383,387
    if test "$name" = japan; then ids=1008,10484,318,10417,387; fi
    K3_IDS="$ids" K3_GROUP_COVERAGE=2 K3_JOINT_RANK="$joint" K3_SCALE_ROWS=1 K3_SCALE_EXCEPTIONS=1 \
        K3_GROUP_COVERAGE_REPORT="$work/$name.jsonl" K3_LOGITS="$work/$name.bin" \
        K3_DUMPSEL="$work/$name.routes" ./candidate > "$name.log" 2> "$name.stderr"
    cmp "$name-reference.bin" "$name.bin"
    cmp "$name-reference.routes" "$name.routes"
done
python3 reverse-check.py "$work" --stage "$stage" --runs france japan --output "$work/results.json"