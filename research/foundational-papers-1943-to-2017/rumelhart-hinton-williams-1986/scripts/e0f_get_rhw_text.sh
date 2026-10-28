#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. Check packages in /root/dm-study/*/pylib ==="
ls /root/dm-study/pebbling/pylib /root/dm-study/eyeriss/pylib 2>/dev/null || true

echo "=== 2. Download DTIC_ADA164453_djvu.txt (Rumelhart, Hinton & Williams 1985/1986) from Archive.org ==="
curl -sL "https://archive.org/download/DTIC_ADA164453/DTIC_ADA164453_djvu.txt" -o rhw_ada164453.txt
wc -l -c rhw_ada164453.txt
head -n 80 rhw_ada164453.txt
