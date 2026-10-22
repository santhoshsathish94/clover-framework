#!/bin/bash
# Cycle 32 baseline + inference-only levers on run1 (last.pt = step 5000), one config per card, same 598 validation plans.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; CK=${1:-$P/out/run1/last.pt}; TAG=${2:-run1}
mkdir -p $P/out/q-$TAG
run() { # gpu name args...
  g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $CK --out $P/out/q-$TAG/$n.json --name $n "$@" > $P/out/q-$TAG/$n.log 2>&1 &
}
run 0 base
run 1 t08 --temp 0.8
run 2 t06 --temp 0.6
run 3 topp09 --top-p 0.9
run 4 adj5 --adj 5,4,0.5
run 5 adj10 --adj 10,4,0.5
run 6 mbr8 --cands 8
run 7 mbr8_adj5_t08 --cands 8 --adj 5,4,0.5 --temp 0.8
wait
echo finished
