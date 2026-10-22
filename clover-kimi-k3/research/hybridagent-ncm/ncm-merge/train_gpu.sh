#!/bin/bash
# usage: train_gpu.sh NAME STEPS BS_PER_GPU LR [extra args]   (8 cards, torchrun, stage-2 plan-conditioned training)
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories
P=$HOME/plan2text
NAME=$1; STEPS=$2; BS=$3; LR=$4; shift 4
cd $P
$HOME/ncm/venv/bin/torchrun --standalone --nproc_per_node=8 ncm_stage2.py train --pairs $P/data/pairs_train.jsonl --out $P/out/$NAME \
  --steps $STEPS --bs $BS --lr $LR --warmup 300 --P 64 --nval 512 --val-pairs $P/data/pairs_valid.jsonl --eval-every 500 --log-every 50 "$@"
