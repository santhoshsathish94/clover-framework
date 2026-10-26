#!/usr/bin/env bash
set -euo pipefail

STUDY="mcculloch-pitts-1943"
BASE="/home/usergpu/foundations-study/$STUDY"
PY="python3"

echo "=== Repo 1 unittest with PYTHONPATH=. ==="
cd "$BASE/work/kaisdukes_run"
PYTHONPATH=. $PY -m unittest discover -s tests -p "*test.py" -v

echo "=== Repo 2 unittest with -t . ==="
cd "$BASE/work/codeforge_run"
$PY -m unittest discover -s tests -t . -v
