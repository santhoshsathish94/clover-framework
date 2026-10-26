#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e7"
PY="/root/ncm-cpu/venv/bin/python"

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$W2/e7_splitkv_combine.py" "$WORK/e7_splitkv_combine.py"

"$PY" -u "$WORK/e7_splitkv_combine.py"
