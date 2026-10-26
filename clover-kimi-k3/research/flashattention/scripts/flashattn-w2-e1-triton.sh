#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/flashattention/w2"
REPO="/root/dm-study/flashattention/repo/flash-attention"
PY="/root/ncm-cpu/venv/bin/python"
PIP="/root/ncm-cpu/venv/bin/pip"

mkdir -p "$W2/pylib" "$W2/work"

echo "=== 1. Check existing repo & papers from round 1 ==="
git -C "$REPO" rev-parse HEAD
git -C "$REPO" log -1 --format="%H %ci %s"
echo "--- Triton files in repo ---"
find "$REPO" -maxdepth 4 -name "*triton*" | sort
wc -l "$REPO"/flash_attn/flash_attn_triton*.py "$REPO"/flash_attn/ops/triton/*.py 2>/dev/null || true

echo "=== 2. Install triton into $W2/pylib if not already installed ==="
if [ ! -d "$W2/pylib/triton" ]; then
    "$PIP" install --target "$W2/pylib" triton --quiet
fi
du -sh "$W2/pylib"

echo "=== 3. Negative control: TRITON_INTERPRET=0 on CPU tensor ==="
PYTHONPATH="$W2/pylib" TRITON_INTERPRET=0 "$PY" "$W2/work/e1_triton_probe.py" --neg-control

echo "=== 4. Run E1 probe with TRITON_INTERPRET=1 ==="
PYTHONPATH="$W2/pylib" TRITON_INTERPRET=1 "$PY" "$W2/work/e1_triton_probe.py"

