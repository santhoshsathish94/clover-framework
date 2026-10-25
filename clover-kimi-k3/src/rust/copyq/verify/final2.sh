#!/bin/bash
cd "$(dirname "$0")"
grep -c '^TIMING' out/timing.log; grep -E 'FOREIGN|ABORT' out/timing.log || echo "no FOREIGN/ABORT lines in timing.log"
python3 tsummary.py
echo "--- raw lines"; grep '^TIMING' out/timing.log | cut -c1-200
echo "--- driver stderr during timing"; wc -c out/timing.err
