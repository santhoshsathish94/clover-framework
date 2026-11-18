#!/bin/bash
# Chain: waits for the A2 training unit, then A2 eval -> A3 (HybridAgent-8B answers) -> B states -> B link/bridge -> B3 (base answers) -> A3-style render.
B=/opt/clover-k3/qwen-pilot-20261118
S=$B/scripts
NCM=/root/ncm-cpu/venv/bin/python
HAV=/root/hybridagent-venv/bin/python
export HF_HOME=/root/hf OMP_NUM_THREADS=16
step() { echo "=== $(date '+%F %T') $1"; }
while systemctl is-active --quiet qwen-pilot-a2train3-20261118; do sleep 20; done
step "A2 eval trained";  $NCM $S/a2_c3_wmt14.py eval --ckpt $B/runs/a2/last.pt --n 64 --out $B/results/a2_eval_trained.json
step "A3 trained";       $NCM $S/a3_end_to_end.py --n 40 --ckpt $B/runs/a2/last.pt --out $B/results/a3_trained.json
step "A3 zero-shot";     $NCM $S/a3_end_to_end.py --n 40 --out $B/results/a3_zeroshot.json
mkdir -p $B/states
step "B states base held";  $HAV $S/b_states.py --model base --split held  --n 300  --out $B/states/states_base_held.pt
step "B states ha held";    $HAV $S/b_states.py --model ha   --split held  --n 300  --out $B/states/states_ha_held.pt
step "B states base train"; $HAV $S/b_states.py --model base --split train --n 1500 --out $B/states/states_base_train.pt
step "B states ha train";   $HAV $S/b_states.py --model ha   --split train --n 1500 --out $B/states/states_ha_train.pt
step "B link/bridge";       $HAV $S/b_link.py
step "B3 base answers";     $NCM $S/b3_base_answers.py --n 40 --out $B/results/b3_answers.json
step "B3 render";           $NCM $S/a3_end_to_end.py --n 40 --answers $B/results/b3_answers.json --ckpt $B/runs/a2/last.pt --out $B/results/b3_render_trained.json
step "CHAIN DONE"
exit 0
