#!/bin/bash
# usage: plan_gen.sh INPUT.jsonl OUTPREFIX [N]   -> 8 shards, one per card, OUTPREFIX.<i>.jsonl
export HF_HUB_OFFLINE=1 NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories
P=$HOME/plan2text
IN=$1; OUT=$2; N=${3:-1000000}
cd $P
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i $HOME/ncm/venv/bin/python plan_writer.py --pairs-in $IN --out $OUT.$i.jsonl --n $N --device cuda:0 --bs 64 --max-new 64 --shard $i --nshards 8 > $OUT.$i.log 2>&1 &
done
wait
echo all-shards-done
wc -l $OUT.*.jsonl | tail -1
