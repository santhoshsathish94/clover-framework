#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/krizhevsky-2012

echo "=== 1. Clone & inspect Alex Krizhevsky's original 2012 cuda-convnet (Footnote 1 & 3, pp. 2, 4) ==="
cd "$BASE/repo"
if [ ! -d "ulrichstern-cuda-convnet" ]; then
  git clone --depth 1 "https://github.com/ulrichstern/cuda-convnet.git" ulrichstern-cuda-convnet
fi
echo "--- ulrichstern-cuda-convnet (commit $(git -C ulrichstern-cuda-convnet rev-parse --short HEAD)) ---"
find ulrichstern-cuda-convnet -maxdepth 2 -not -path '*/.*'
echo ">>> CIFAR-10 11% LRN layer config cited in Footnote 3 (p. 4):"
head -n 95 ulrichstern-cuda-convnet/example-layers/layers-18pct.cfg ulrichstern-cuda-convnet/example-layers/layers-80sec.cfg 2>/dev/null || true

echo "=== 2. Inspect matteo-dunnhofer/imagenet/tf/models/alexnet.py ==="
head -n 110 "$BASE/repo/matteo-dunnhofer-imagenet/tf/models/alexnet.py"

echo "=== 3. Run our verified krizhevsky2012_experiments.py (E2, E3, E4) ==="
cd "$BASE/work"
$PY krizhevsky2012_experiments.py
echo "FINISHED"
