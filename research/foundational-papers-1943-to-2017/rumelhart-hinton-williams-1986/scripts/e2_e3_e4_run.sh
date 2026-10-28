#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"

echo "=== 1. Run karpathy/micrograd tests & Kushagra514 scripts ==="
cd "$BASE/work/micrograd_run"
$PY -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'test')
import test_engine
test_engine.test_sanity_check()
test_engine.test_more_ops()
print('karpathy/micrograd test_sanity_check & test_more_ops: OK (rc=0)')
"

rm -rf "$BASE/work/kushagra_run"
cp -r "$BASE/repo/Kushagra514-Backpropagation-from-scratch-Rumelhart-Hinton-Williams-1986-" "$BASE/work/kushagra_run"
cd "$BASE/work/kushagra_run"
for f in 1neuron.py 2neuron_layer.py hidden_neuron_layer.py; do
  echo ">>> Running Kushagra514/$f"
  sed -i 's/import matplotlib.pyplot as plt/# import matplotlib.pyplot as plt/g' "$f"
  sed -i 's/plt\./# plt./g' "$f"
  $PY "$f" | tail -n 8
done

echo "=== 2. Run E2, E3, and E4 (rhw1986_experiments.py) ==="
cd "$BASE/work"
$PY rhw1986_experiments.py
