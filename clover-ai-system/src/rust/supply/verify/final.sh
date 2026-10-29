#!/bin/bash
# The final recorded run: the whole build-and-test cycle (cycle.sh: reference + smoke, cargo build/test, Rust smoke, driver, md5) and then the full differential run three times in a row with the same
# binaries (out/final-1 .. out/final-3). Everything goes to out/final.log; out/final.done appears at the end. Started with nohup from go-final (see RESULTS.md).
cd "$(dirname "$0")"
mkdir -p out
rm -rf out/final-1 out/final-2 out/final-3 out/final.done
{
  date
  ./cycle.sh
  for i in 1 2 3; do
    echo "== differential run final-$i"
    ./run.sh final-$i --keep-going > out/final-$i.stdout 2>&1
    echo "exit status $?"
    grep -v coverage "out/final-$i/summary.txt" | cut -c1-330
  done
  date
} > out/final.log 2>&1
echo done > out/final.done
