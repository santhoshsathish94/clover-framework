#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/lecun-1998

echo "=== 1. Run pop123-ux/LeNet_5-from-scratch tests & inspect parameter count ==="
cd "$BASE/repo/pop123-ux-LeNet_5-from-scratch"
$PY - << 'EOF'
import sys
sys.path.insert(0, ".")
from tests import test_model
for name in dir(test_model):
    if name.startswith("test_"):
        getattr(test_model, name)()
        print("  pop123-ux passed:", name)
from src import LeNet_5
m = LeNet_5()
print("  pop123-ux LeNet_5.params():", m.params())
for k, p in m.named_parameters():
    print(f"    {k:25s}: {tuple(p.shape)} ({p.numel()})")
EOF

echo "=== 2. Run VTongTV/LeNet-5-scratch tests & inspect parameter count ==="
cd "$BASE/repo/VTongTV-LeNet-5-scratch"
$PY - << 'EOF'
import sys
sys.path.insert(0, ".")
from lenet.network import LeNet5
net = LeNet5()
tot = 0
for i, layer in enumerate(net.layers):
    p = sum(arr.size for arr in getattr(layer, "params", lambda: [])()) if hasattr(layer, "params") else 0
    print(f"  Layer {i} ({type(layer).__name__}):w={getattr(layer, 'weights', None) is not None}")
EOF

echo "=== 3. Run our verified lecun1998_experiments.py (E2, E3, E4) ==="
cd "$BASE/work"
$PY lecun1998_experiments.py
echo "FINISHED"
