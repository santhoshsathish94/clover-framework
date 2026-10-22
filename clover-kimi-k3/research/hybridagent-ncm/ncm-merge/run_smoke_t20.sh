#!/bin/bash
# Cycle 28 smoke on k3 CPU: keyword-plan pairs -> ncm_stage2 train (timing run) . Detached; log /root/ncm-cpu/smoke/*.log
S=/root/ncm-cpu/smoke
cd /root/ncm-cpu; . venv/bin/activate; mkdir -p $S
python make_pairs_kw.py 3000 10 $S/pairs_kw.jsonl > $S/make.log 2>&1
head -3 $S/pairs_kw.jsonl | cut -c1-200 >> $S/make.log
python ncm_stage2.py train --pairs $S/pairs_kw.jsonl --out $S/t20 --steps 20 --bs 8 --warmup 5 --eval-every 20 --log-every 5 --nval 32 > $S/t20.log 2>&1
touch $S/t20.done
