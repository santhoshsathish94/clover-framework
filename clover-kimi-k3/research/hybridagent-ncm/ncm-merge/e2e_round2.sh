#!/bin/bash
# Cycle 32: end-to-end requests. (1) new plans with a names-and-words prompt; (2) run1 on old and new plans with baseline and with the best inference setting.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P; PY=$HOME/ncm/venv/bin/python; CK=$P/out/run1/last.pt; O=$P/out/e2e2; mkdir -p $O
CUDA_VISIBLE_DEVICES=0 $PY plan_from_topic3.py $O/topic_plans_v3.jsonl cuda:0 > $O/plans_v3.log 2>&1
ev() { g=$1; n=$2; pairs=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $PY quality_eval.py --pairs $pairs --ckpt $CK --out $O/$n.json --name $n --n 48 --bs 48 "$@" > $O/$n.log 2>&1 &
}
ev 0 old_base $P/out/topic_plans.jsonl
ev 1 old_best $P/out/topic_plans.jsonl --temp 0.6 --adj 10,4,0.5 --cands 8
ev 2 new_base $O/topic_plans_v3.jsonl
ev 3 new_best $O/topic_plans_v3.jsonl --temp 0.6 --adj 10,4,0.5 --cands 8
wait
echo finished
