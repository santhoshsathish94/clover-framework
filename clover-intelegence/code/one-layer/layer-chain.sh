#!/bin/bash
# Runs the batched layers FROM..TO one after another, each against the pod on the previous layer's pod output
# (so every layer is checked on the same inputs it would really receive), and stops at the first layer that is not
# bit-identical. K (input snapshots) is read from each deployed layer's own source.
# usage: layer-chain.sh FROM TO RESID_F32 SNAPS_F32     (inputs of layer FROM; files are not modified)
# env: N (tokens, default 1000), W (window, default = N)
set -u
FROM=$1; TO=$2; RES=$3; SNP=$4
N=${N:-1000}; W=${W:-$N}
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/tansformers
cd $H
cp $RES chain.cur.resid; cp $SNP chain.cur.snaps
: > chain-summary.txt
for L in $(seq $FROM $TO); do
  K=$(grep -E '^#define TRANSFORMER_INPUT_SNAPSHOTS' $D/transformer-$L/transformer-$L.c | awk '{print $3}')
  start=$(date +%s)
  LB_SAVE_REF=chain.new.resid LB_SAVE_SNAPS=chain.new.snaps bash layer-gen-run.sh $L $K $N $W chain.cur.resid chain.cur.snaps > chain-$L.log 2>&1
  line=$(grep -E "^  layer $L, " chain-$L.log | head -1)
  echo "L$L K=$K $(( $(date +%s) - start ))s | $line" | tee -a chain-summary.txt
  case "$line" in *bit-identical*) ;; *) echo "STOP at layer $L: not bit-identical or failed" | tee -a chain-summary.txt; exit 1;; esac
  grep -q "as expected" chain-$L.log || { echo "STOP at layer $L: pod snapshots not as expected" | tee -a chain-summary.txt; exit 1; }
  mv chain.new.resid chain.cur.resid; mv chain.new.snaps chain.cur.snaps
done
echo "DONE layers $FROM..$TO" | tee -a chain-summary.txt
