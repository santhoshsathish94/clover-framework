#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/lecun-1998
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/repo"

for r in \
  "jmlee8939/LeNet-5_1998_pytorch" \
  "pop123-ux/LeNet_5-from-scratch" \
  "VTongTV/LeNet-5-scratch" \
  "SunnyHaze/LeNet5-MNIST-Pytorch"; do
  name=$(echo "$r" | tr '/' '-')
  if [ ! -d "$name" ]; then
    echo "Cloning $r -> $name"
    git clone --depth 1 "https://github.com/$r.git" "$name"
  fi
done

for d in *; do
  if [ -d "$d" ]; then
    echo "=== $d (commit $(git -C "$d" rev-parse --short HEAD)) ==="
    find "$d" -maxdepth 3 -not -path '*/.*'
    for f in $(find "$d" -maxdepth 3 -name "*.py" | sort); do
      echo "--- $f ---"
      head -n 90 "$f"
    done
  fi
done
echo "FINISHED"
