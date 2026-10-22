#!/bin/bash
# Cycle 31: evaluation of run1 (step 5000 = last.pt) on the 598 validation pairs, and the end-to-end topic -> plan (HybridAgent) front half. Two cards in parallel.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories HF_HUB_OFFLINE=1
P=$HOME/plan2text; cd $P
PY=$HOME/ncm/venv/bin/python
CUDA_VISIBLE_DEVICES=1 $PY plan_from_topic.py $P/out/topic_plans.jsonl cuda:0 > $P/out/topic_plans.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY ncm_stage2.py eval --pairs $P/data/pairs_valid.jsonl --ckpt $P/out/run1/last.pt --out $P/out/eval_run1.json --n 598 --bs 64 --P 64 --seeds 3 --device cuda:0 > $P/out/eval_run1.log 2>&1
wait
echo finished
