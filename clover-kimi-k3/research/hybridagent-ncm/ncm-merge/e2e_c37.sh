#!/bin/bash
# Cycle 37: end-to-end 48 requests (plans v3) with E3; run1 + balanced decoding as comparator; and the CFG null-prefix paired checks.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; O=$P/out/e2e37; mkdir -p $O
PL=$P/out/e2e2/topic_plans_v3.jsonl
B="--temp 0.8 --cands 8 --adj 10,8,0.8 --ngram 3,8"
ev() { g=$1; n=$2; ck=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $PL --ckpt $P/out/$ck/last.pt --out $O/$n.json --name $n --n 48 --bs 48 "$@" > $O/$n.log 2>&1 &
}
ev 0 run1_bal run1 $B
ev 1 e3_bal e3_drop $B
ev 2 e3_bal_cfg1 e3_drop $B --cfg 1.0
ev 3 e3_base e3_drop
CUDA_VISIBLE_DEVICES=4 $PY cfg_check.py $P/out/e3_drop/last.pt $P/data/pairs_valid.jsonl $O/cfgcheck_e3.json > $O/cfgcheck_e3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY cfg_check.py $P/out/e2_tiednorm/last.pt $P/data/pairs_valid.jsonl $O/cfgcheck_e2.json > $O/cfgcheck_e2.log 2>&1 &
wait
echo finished > $O/done