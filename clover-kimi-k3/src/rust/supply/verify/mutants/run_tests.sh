#!/bin/bash
# cargo test --release for each named mutant (one at a time, nice 10); the tail of each run goes to mutants/logs/test-<name>.txt, one summary line per mutant to mutants/logs/tests.txt
# (the summary is the FIRST 'test result' line = the crate's unit tests; the last one is the empty doc-test run. The first version of this script printed the last one; tests.txt was regenerated from the logs.)
# usage: run_tests.sh NAME...
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p mutants/logs
for n in "$@"; do
  d=$PWD/work/mut/$n
  [ -d "$d" ] || { echo "$n NO DIR" >> mutants/logs/tests.txt; continue; }
  ( cd "$d" && timeout -k 10 900 nice -n 10 cargo test --release > "$OLDPWD/mutants/logs/test-$n.txt" 2>&1 )
  rc=$?
  res=$(grep -E '^test result' mutants/logs/test-$n.txt | head -1)
  fl=$(grep -E '^test .* FAILED' mutants/logs/test-$n.txt | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
  echo "$n rc=$rc | $res | failed: $fl" >> mutants/logs/tests.txt
done
echo "TESTS DONE" >> mutants/logs/tests.txt
