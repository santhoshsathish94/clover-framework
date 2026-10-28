#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e6"
PY="/root/ncm-cpu/venv/bin/python"

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$W2/e6_bwd_recompute.py" "$WORK/e6_bwd_recompute.py"

"$PY" -u "$WORK/e6_bwd_recompute.py"
