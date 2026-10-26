#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/turing-1950"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
mkdir -p "$BASE/work"

echo "=== Check simonbreiter universal.json on a halting 1-state encoded machine ==="
cd "$BASE/work/simonbreiter_run"
# Encoded transition: state q1 (0), symbol '0' (0) -> nextState q2 (00), write '1' (00), move right (00) => 010100100100
# End of config: 111, followed by input tape: 0
# When the simulated machine transitions to q2 (which has no transition in the encoded table), universal.json halts at q82!
$PY ./universal_turing_machine --instructions assets/instructions/universal.json --begin q1 --end q82 --tape "0101001001001110" | tail -n 22

echo "=== Run E2: Stored-Program Digital Computer & Section 5 Universality ==="
cd "$BASE/work"
$PY e2_stored_program_and_universal.py

echo "=== Run E3 & E4: Imitation Game, Mathematical Objection, Continuity & Learning Machines ==="
$PY e3_e4_imitation_and_learning.py
