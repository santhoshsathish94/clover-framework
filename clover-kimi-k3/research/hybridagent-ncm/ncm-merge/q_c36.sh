#!/bin/bash
# Cycle 36: E3 (plan dropout) with and without classifier-free guidance; balanced decoding = temp 0.8 + adj 10,8,0.8 + ngram 3,8 + MBR8.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python
mkdir -p $P/out/q-c36
run() { g=$1; n=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $P/data/pairs_valid.jsonl --ckpt $P/out/e3_drop/last.pt --out $P/out/q-c36/$n.json --name $n "$@" > $P/out/q-c36/$n.log 2>&1 &
}
B="--temp 0.8 --cands 8 --adj 10,8,0.8 --ngram 3,8"
run 0 e3_base
run 1 e3_bal $B
run 2 e3_base_cfg1 --cfg 1.0
run 3 e3_base_cfg2 --cfg 2.0
run 4 e3_bal_cfg05 $B --cfg 0.5
run 5 e3_bal_cfg1 $B --cfg 1.0
run 6 e3_bal_cfg2 $B --cfg 2.0
run 7 e3_bal_cfg4 $B --cfg 4.0
wait
echo finished > $P/out/q-c36/done