#!/bin/bash
# Cycle 35: level-0-fixed checkpoints, baseline and balanced decoding, same 598 validation plans as Cycle 32.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python
mkdir -p $P/out/q-c35
run() { g=$1; ck=$2; n=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $P/out/$ck/last.pt --out $P/out/q-c35/$n.json --name $n "$@" > $P/out/q-c35/$n.log 2>&1 &
}
run 0 e2_tiednorm e2_base
run 1 e2_tiednorm e2_bal --temp 0.8 --cands 8 --adj 10,8,0.8 --ngram 3,8
run 2 e1_untied e1_base
run 3 e1_untied e1_bal --temp 0.8 --cands 8 --adj 10,8,0.8 --ngram 3,8
wait
echo finished > $P/out/q-c35/done