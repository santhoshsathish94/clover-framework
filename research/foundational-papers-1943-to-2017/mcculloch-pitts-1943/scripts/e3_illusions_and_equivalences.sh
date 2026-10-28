#!/usr/bin/env bash
set -euo pipefail

STUDY="mcculloch-pitts-1943"
BASE="/home/usergpu/foundations-study/$STUDY"
cd "$BASE/work"
python3 e3_illusions_and_equivalences.py
