#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/hopfield-1982"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
mkdir -p "$BASE/work"

echo "=== 1. Read pauljmello/Hopfield-Networks files ==="
cd "$BASE/repo/pauljmello-Hopfield-Networks"
for f in README.md hopfield.py modernhopfield.py main.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done

echo "=== 2. Read & Run 1hubert/hopfield-network in a work copy ==="
rm -rf "$BASE/work/1hubert_run"
cp -r "$BASE/repo/1hubert-hopfield-network" "$BASE/work/1hubert_run"
cd "$BASE/work/1hubert_run"
cat main.py
# Replace interactive matplotlib show() with non-interactive backend in work copy
sed -i 's/plt.show()/# plt.show()/g' main.py
MPLBACKEND=Agg $PY main.py
