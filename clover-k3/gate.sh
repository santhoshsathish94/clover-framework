#!/bin/bash
# The correctness gate. "The capital of France is" must produce logits whose
# md5 is the preserved baseline, and the argmax must be 17374 (' Paris').
#
# This is the check every change in the arc had to pass. A timing result from a
# build that has not passed it means nothing.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$HERE/build"
: "${K3_TRUNK:=/root/k3trunk_i8}"
: "${K3_TRUNKPATH:=$K3_TRUNK/trunk.bin}"

BASE5=23d162dcefb18211a7540ef12948f1eb
P5="1008,10484,318,15383,387"

[ -x "$B/clover-k3" ] || { echo "build it first: ./build.sh" >&2; exit 1; }

OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
K3_PREFETCH=4 K3_TRUNKRAM=0 K3_TRUNKPATH="$K3_TRUNKPATH" K3_NREADER=14 \
K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
K3_INDEX="$B/eqidx.bin" K3_IDS="$P5" K3_LOGITS="$B/gate.bin" \
    "$B/clover-k3" | grep -E "emitted token|total wall"

m=$(md5sum "$B/gate.bin" | cut -d' ' -f1)
if [ "$m" = "$BASE5" ]; then
    echo "PASS  logits md5 $m == preserved baseline"
else
    echo "FAIL  logits md5 $m != $BASE5"
    exit 1
fi
