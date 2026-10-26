#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
mkdir -p "$BASE/work"

echo "=== Read TaraAlcira/PerceptronsRevisited files ==="
cd "$BASE/repo/TaraAlcira-PerceptronsRevisited"
for f in README.md generate_spirals/one_spiral_function.py generate_spirals/two_spirals_function.py generate_spirals/randomize_spirals.py CNN/CNN_best2.py; do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
