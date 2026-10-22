#!/bin/bash
# Cycle 30: story windows on the gpu box (CPU only). Env points at the box's NCM repo and the released TinyStories artifacts.
export NCM_REPO=$HOME/ncm/repo NCM_ART=$HOME/ncm/artifacts/tinystories
P=$HOME/plan2text
cd $P
PY=$HOME/ncm/venv/bin/python
$PY make_pairs_ts.py $P/data/TinyStories-valid.txt 600 $P/data/win_valid.jsonl
$PY make_pairs_ts.py $P/data/TinyStories-train.txt 80000 $P/data/win_train.jsonl
wc -l $P/data/win_valid.jsonl $P/data/win_train.jsonl
