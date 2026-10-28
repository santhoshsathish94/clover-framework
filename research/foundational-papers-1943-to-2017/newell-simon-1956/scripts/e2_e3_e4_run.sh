#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
export PATH="/root/ncm-cpu/venv/bin:$PATH"

cd "$BASE/work"
$PY run_e2_e3_e4.py
echo "=== Check work limit in dmoews/logic-theorist/logic.py ==="
grep -n -C 4 "work" "$BASE/work/dmoews_ipli_run/logic.py" || true
