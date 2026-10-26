#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/flashattention/w2"
WORK="$W2/work_e4"
PY="/root/ncm-cpu/venv/bin/python"

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$W2/e4_alg1_and_traffic.py" "$WORK/e4_alg1_and_traffic.py"

"$PY" -u "$WORK/e4_alg1_and_traffic.py"
