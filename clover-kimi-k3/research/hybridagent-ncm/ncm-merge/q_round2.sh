#!/bin/bash
# Cycle 32 round 2: combine the levers that helped (low temperature, adjacency penalty, MBR) on run1, one config per card.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; CK=${1:-$P/out/run1/last.pt}; TAG=${2:-run1b}
mkdir -p $P/out/q-$TAG
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $CK --out $P/out/q-$TAG/$n.json --name $n "$@" > $P/out/q-$TAG/$n.log 2>&1 &
}
run 0 t06_adj10 --temp 0.6 --adj 10,4,0.5
run 1 t05_adj10 --temp 0.5 --adj 10,4,0.5
run 2 t07_adj10 --temp 0.7 --adj 10,4,0.5
run 3 t06_adj10_mbr8 --temp 0.6 --adj 10,4,0.5 --cands 8
run 4 t07_adj10_mbr8 --temp 0.7 --adj 10,4,0.5 --cands 8
run 5 t06_adj10_mbr16 --temp 0.6 --adj 10,4,0.5 --cands 16
run 6 t08_adj10_mbr8 --temp 0.8 --adj 10,4,0.5 --cands 8
run 7 t04_adj10 --temp 0.4 --adj 10,4,0.5
wait
echo finished
