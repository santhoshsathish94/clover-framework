#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

curl -sL "http://bitsavers.trailing-edge.com/pdf/mit/ai/aim_index.txt" -o aim_index.txt
grep -i -C 3 "papert\|perceptron" aim_index.txt | head -n 120 || true
