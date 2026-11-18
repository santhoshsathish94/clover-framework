#!/bin/bash
# A2 step 1: tokenizer reconstruction gate on Dolly windows, then zero-shot eval of the released WMT14 generator with plan prefixes.
cd /opt/clover-k3/qwen-pilot-20261118
PY=/root/ncm-cpu/venv/bin/python
export HF_HOME=/root/hf OMP_NUM_THREADS=24
echo "== recon"; $PY scripts/a2_c3_wmt14.py recon --n 200 --out results/a2_recon.json
echo "== eval zero-shot"; $PY scripts/a2_c3_wmt14.py eval --n 64 --out results/a2_eval_zeroshot.json
echo STEP1DONE
exit 0
