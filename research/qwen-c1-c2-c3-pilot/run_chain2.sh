#!/bin/bash
B=/opt/clover-k3/qwen-pilot-20261118
export HF_HOME=/root/hf OMP_NUM_THREADS=16
HAV=/root/hybridagent-venv/bin/python
echo "=== $(date '+%F %T') B4 ha";   $HAV $B/scripts/b4_latent_loop.py --model ha   --n 16 --out $B/results/b4_ha.json
echo "=== $(date '+%F %T') B4 base"; $HAV $B/scripts/b4_latent_loop.py --model base --n 16 --out $B/results/b4_base.json
echo "=== $(date '+%F %T') CHAIN2 DONE"
exit 0
