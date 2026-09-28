#!/bin/bash
# Rebuild the equation from a checkpoint and a packed trunk.
#
# Paths come from config.env. NPOS is the prompt length to compile for; it is
# a compile-time constant, so a build is for one length.
#
# Everything this writes goes in build/. Nothing in the repo is modified.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/config.env"
: "${NPOS:=5}"
B="$HERE/build"
mkdir -p "$B"

[ -d "$K3_MODEL" ] || { echo "no model directory: $K3_MODEL" >&2; exit 1; }
[ -f "$K3_TRUNK/trunk.json" ] || { echo "no trunk.json in: $K3_TRUNK" >&2; exit 1; }

echo "== 1/3  locating the five non-layer tensors"
python3 "$HERE/dump_st_model.py" "$K3_MODEL" "$B/st_model.json"

echo "== 2/3  flattening the index"
K3_MODEL="$K3_MODEL" \
K3_TRUNKJSON="$K3_TRUNK/trunk.json" \
K3_STMODEL="$B/st_model.json" \
K3_INDEX="$B/eqidx.bin" \
    python3 "$HERE/dump_eqidx.py"

echo "== 3/3  compiling for NPOS=$NPOS"
# -ffp-contract=off is part of the verified build, not a preference: without it
# GCC fuses mul+add and the logits move by 2-3 ULP.
gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS="$NPOS" \
    -o "$B/clover-k3" "$HERE/clover-k3.c" -lm

echo
echo "built $B/clover-k3   index $B/eqidx.bin"
echo "run ./gate.sh to check it against the preserved baseline"
