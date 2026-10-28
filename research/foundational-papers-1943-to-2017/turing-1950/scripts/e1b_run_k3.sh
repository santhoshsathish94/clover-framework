#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/turing-1950"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
if [ ! -d "vukrosic-turing" ]; then
  git clone --depth 1 https://github.com/vukrosic/turing-from-scratch.git vukrosic-turing
fi
if [ ! -d "simonbreiter-utm" ]; then
  git clone --depth 1 https://github.com/simonbreiter/universal-turing-machine.git simonbreiter-utm
fi

echo "=== Read vukrosic/turing-from-scratch files ==="
cd "$BASE/repo/vukrosic-turing"
for f in README.md step1_machine.py step2_universal.py step4_halting.py step8_imitation.py step9_learning.py tests.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done

echo "=== Run vukrosic/turing-from-scratch in a work copy ==="
rm -rf "$BASE/work/vukrosic_run"
cp -r "$BASE/repo/vukrosic-turing" "$BASE/work/vukrosic_run"
cd "$BASE/work/vukrosic_run"
for f in step1_machine.py step2_universal.py step3_beaver.py step4_halting.py step7_evidence.py step8_imitation.py step9_learning.py tests.py; do
  echo ">>> Running $f"
  $PY "$f"
done

echo "=== Run simonbreiter/universal-turing-machine in a work copy ==="
rm -rf "$BASE/work/simonbreiter_run"
cp -r "$BASE/repo/simonbreiter-utm" "$BASE/work/simonbreiter_run"
cd "$BASE/work/simonbreiter_run"
$PY ./universal_turing_machine --instructions assets/instructions/readme_example.json --begin q0 --end qdone --tape " "
$PY ./universal_turing_machine --instructions assets/instructions/multiplication.json --tape "00 000"
# Run the Universal Turing Machine configuration universal.json on an encoded machine+tape!
$PY ./universal_turing_machine --instructions assets/instructions/universal.json --begin q1 --end qdone --tape "01000010010010000110010000010001001000011101101 00100"
