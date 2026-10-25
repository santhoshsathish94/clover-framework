#!/bin/bash
# Load test of the thread scenarios (T1, T2: the barrier of T2 has a 10 s timeout) and of the rest: THREE full runs of the unmodified crate at the same time, JOBS=32 each (about 96 units in flight),
# 3 GB address-space cap per process. Any failing unit would show here as a FAIL line. Output: out/v2stress.log, out/v2stress-summary.txt (summary lines of the three runs).
cd "$(dirname "$0")/.."
rm -rf out/v2stress-1 out/v2stress-2 out/v2stress-3 out/v2stress.log
ulimit -v 3145728
for i in 1 2 3; do
  ( JOBS=32 ./run.sh v2stress-$i --keep-going > out/v2stress-$i.stdout 2>&1; echo "stress-$i exit status $?" >> out/v2stress.log ) &
done
wait
for i in 1 2 3; do echo "== v2stress-$i"; grep -v coverage out/v2stress-$i/summary.txt | cut -c1-240; done > out/v2stress-summary.txt
echo "uptime at the end: $(uptime)" >> out/v2stress.log
