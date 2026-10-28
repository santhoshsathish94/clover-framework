#!/usr/bin/env bash
set -euo pipefail

STUDY="mcculloch-pitts-1943"
BASE="/home/usergpu/foundations-study/$STUDY"
mkdir -p "$BASE/work"

python3 "$BASE/work/mp1943_calculus.py"
