#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/turing-1950"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"

cd "$BASE/work/simonbreiter_run"
$PY -c "
import json
with open('assets/instructions/universal.json') as f:
    u = json.load(f)
defined = set(u.keys())
referenced = set()
for st, cases in u.items():
    for sym, act in cases.items():
        referenced.add(act['nextState'])
print('Defined states count:', len(defined))
print('Undefined nextStates:', sorted(referenced - defined))
"
# Run with --end q82 as documented in README.md line 209!
$PY ./universal_turing_machine --instructions assets/instructions/universal.json --begin q1 --end q82 --tape "01000101000100111 " | tail -n 25
