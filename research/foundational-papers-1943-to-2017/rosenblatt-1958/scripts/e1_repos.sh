#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"
mkdir -p "$BASE/work"

echo "=== 1. Read & Run vukrosic/rosenblatt-from-scratch in a work copy ==="
rm -rf "$BASE/work/vukrosic_run"
cp -r "$BASE/repo/vukrosic-rosenblatt-from-scratch" "$BASE/work/vukrosic_run"
cd "$BASE/work/vukrosic_run"
for f in perc.py step1_neuron.py step2_learn.py step3_mark1.py step4_converge.py step5_xor.py tests.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
bash run_all.sh

echo "=== 2. Read & Run kaisdukes/rosenblatt-perceptron in a work copy ==="
rm -rf "$BASE/work/kaisdukes_run"
cp -r "$BASE/repo/kaisdukes-rosenblatt-perceptron" "$BASE/work/kaisdukes_run"
cd "$BASE/work/kaisdukes_run"
for f in src/perceptron.py src/train.py src/evaluate.py src/iris.py tests/perceptron_test.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
PYTHONPATH=. $PY -m unittest discover -s tests -p "*test.py" -v

echo "=== 3. Read & Run Honghe/perceptron in a work copy ==="
rm -rf "$BASE/work/honghe_run"
cp -r "$BASE/repo/Honghe-perceptron" "$BASE/work/honghe_run"
cd "$BASE/work/honghe_run"
for f in perceptron/perceptron.py perceptron/generate_data.py perceptron/test.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
MPLBACKEND=Agg PYTHONPATH=. $PY perceptron/test.py || echo "honghe rc=$?"
