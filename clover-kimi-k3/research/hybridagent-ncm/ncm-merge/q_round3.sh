#!/bin/bash
# Cycle 32 round 3: phrase repetition. Wider/stronger adjacency window, temperature, all with MBR8; adds repeated-trigram rate; true-story reference included in each JSON.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; CK=${1:-$P/out/run1/last.pt}; TAG=${2:-run1c}
mkdir -p $P/out/q-$TAG
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $CK --out $P/out/q-$TAG/$n.json --name $n "$@" > $P/out/q-$TAG/$n.log 2>&1 &
}
run 0 base
run 1 t06_adj10_mbr8 --temp 0.6 --adj 10,4,0.5 --cands 8
run 2 t06_adj10w8_mbr8 --temp 0.6 --adj 10,8,0.8 --cands 8
run 3 t07_adj10w8_mbr8 --temp 0.7 --adj 10,8,0.8 --cands 8
run 4 t08_adj10w8_mbr8 --temp 0.8 --adj 10,8,0.8 --cands 8
run 5 t06_adj20w8_mbr8 --temp 0.6 --adj 20,8,0.8 --cands 8
run 6 t07_adj10_mbr8 --temp 0.7 --adj 10,4,0.5 --cands 8
run 7 t10_adj10w8_mbr8 --temp 1.0 --adj 10,8,0.8 --cands 8
wait
echo finished
