#!/bin/bash
# usage: train_gpu_n.sh NAME CARDS(comma list) STEPS BS_PER_GPU LR [extra args]   (torchrun on the listed cards)
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories
P=$HOME/plan2text
NAME=$1; CARDS=$2; STEPS=$3; BS=$4; LR=$5; shift 5
NP=$(echo $CARDS | awk -F, '{print NF}')
cd $P
CUDA_VISIBLE_DEVICES=$CARDS $HOME/ncm/venv/bin/torchrun --standalone --nproc_per_node=$NP ncm_stage2.py train --pairs $P/data/pairs_train.jsonl --out $P/out/$NAME \
  --steps $STEPS --bs $BS --lr $LR --warmup 200 --P 64 --nval 512 --val-pairs $P/data/pairs_valid.jsonl --eval-every 500 --log-every 50 "$@"
