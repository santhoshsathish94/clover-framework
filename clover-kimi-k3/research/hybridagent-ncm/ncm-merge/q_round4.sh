#!/bin/bash
# Cycle 32 round 4: soft n-gram repetition penalty (token n-grams already produced in the same text), with MBR8 and the wide adjacency window.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; CK=${1:-$P/out/run1/last.pt}; TAG=${2:-run1d}
mkdir -p $P/out/q-$TAG
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $CK --out $P/out/q-$TAG/$n.json --name $n --cands 8 --adj 10,8,0.8 "$@" > $P/out/q-$TAG/$n.log 2>&1 &
}
run 0 t06_ng3p4 --temp 0.6 --ngram 3,4
run 1 t07_ng3p4 --temp 0.7 --ngram 3,4
run 2 t08_ng3p4 --temp 0.8 --ngram 3,4
run 3 t06_ng4p4 --temp 0.6 --ngram 4,4
run 4 t07_ng4p4 --temp 0.7 --ngram 4,4
run 5 t06_ng3p8 --temp 0.6 --ngram 3,8
run 6 t07_ng3p8 --temp 0.7 --ngram 3,8
run 7 t08_ng3p8 --temp 0.8 --ngram 3,8
wait
echo finished
