#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== Early MIT AI Lab Memos (1959-1972) by Minsky or Papert ==="
$PY -c "
txt = open('aim_index.txt', errors='ignore').read()
entries = txt.split('\nAIM-')
for e in entries:
    if any(k in e.lower() for k in ('minsky', 'papert', 'perceptron')):
        header = ('AIM-' + e[:400]).replace('\n', ' ')
        if any(y in header for y in ('196', '1970', '1971', '1972', 'AIM-0', 'AIM-1', 'AIM-2')):
            print(header[:220])
"
