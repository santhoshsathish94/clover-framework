#!/usr/bin/env bash
set -euo pipefail

REPO="/root/dm-study/flashattention/repo/flash-attention"
W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e8"
PY="/root/ncm-cpu/venv/bin/python"

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$W2/e8_fa1_vs_fa2_and_smem.py" "$WORK/e8_fa1_vs_fa2_and_smem.py"

"$PY" -u "$WORK/e8_fa1_vs_fa2_and_smem.py" "$REPO"
