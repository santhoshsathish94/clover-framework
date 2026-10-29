#!/bin/bash
# Compares canonical outputs cpp vs rust per scenario/seed and prints the RESULT lines.
cd "$(dirname "$0")"
for f in out/cpp/S*.txt; do
  b=$(basename $f .txt); r=out/rust/$b.txt
  [ -f "$r" ] || { echo "$b: no rust output"; continue; }
  c=$(grep '^RESULT' $f); u=$(grep '^RESULT' $r)
  if cmp -s $f $r; then eq="canonical-output-IDENTICAL(sha256 $(sha256sum < $f | cut -c1-16))"; else eq="canonical-output-DIFFERS"; fi
  echo "$b: cpp=[$c] rust=[$u] $eq lines=$(wc -l < $f)"
done
echo "--- S5 verbatim"
for t in cpp rust; do echo "[$t]"; cat out/$t/S5-1.txt; done
echo "--- diff S5"; diff out/cpp/S5-1.txt out/rust/S5-1.txt
