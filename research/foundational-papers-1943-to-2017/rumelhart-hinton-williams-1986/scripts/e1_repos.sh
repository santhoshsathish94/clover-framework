#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"
mkdir -p "$BASE/work"

echo "=== 1. Read & Run CodeForge-00/rumelhart_backpropagation in a work copy ==="
rm -rf "$BASE/work/codeforge_run"
cp -r "$BASE/repo/CodeForge-00-rumelhart_backpropagation" "$BASE/work/codeforge_run"
cd "$BASE/work/codeforge_run"
for f in README.md backprop.py tasks.py family.py experiments.py demo.py tests/test_backprop.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
$PY -m unittest discover -s tests -p "test_backprop.py" -t . -v
$PY demo.py

echo "=== 2. Read & Run karpathy/micrograd in a work copy ==="
rm -rf "$BASE/work/micrograd_run"
cp -r "$BASE/repo/karpathy-micrograd" "$BASE/work/micrograd_run"
cd "$BASE/work/micrograd_run"
for f in README.md micrograd/engine.py micrograd/nn.py test/test_engine.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
PYTHONPATH=. $PY -c "
from test.test_engine import test_sanity_check, test_more_ops
test_sanity_check()
test_more_ops()
print('micrograd test_sanity_check & test_more_ops: OK (rc=0)')
"

echo "=== 3. Run Kushagra514 scripts in a work copy ==="
rm -rf "$BASE/work/kushagra_run"
cp -r "$BASE/repo/Kushagra514-Backpropagation-from-scratch-Rumelhart-Hinton-Williams-1986-" "$BASE/work/kushagra_run"
cd "$BASE/work/kushagra_run"
for f in 1neuron.py 2neuron_layer.py hidden_neuron_layer.py; do
  echo ">>> Running $f"
  MPLBACKEND=Agg $PY "$f" | tail -n 15
done
