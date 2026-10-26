#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e5"
PY="/root/ncm-cpu/venv/bin/python"

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$W2/e5_online_softmax.py" "$WORK/e5_online_softmax.py"

"$PY" -u "$WORK/e5_online_softmax.py"
