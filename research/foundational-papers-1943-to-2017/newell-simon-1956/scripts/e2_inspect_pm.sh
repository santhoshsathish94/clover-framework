#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"

cd "$BASE/work/dmoews_ipli_run"
echo "=== pm-axioms.txt ==="
cat pm-axioms.txt
echo "=== pm-theorems.txt (count and lines) ==="
wc -l pm-theorems.txt
cat pm-theorems.txt
echo "=== run_logic.py ==="
cat run_logic.py
