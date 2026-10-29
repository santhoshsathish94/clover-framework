#!/bin/bash
# `cargo test --release` inside the scratch copy of each named mutant (nice -n 10, one at a time). Output: out/ctest/<name>.txt = the "test ..." lines and "test result" line, rc in the last line.
# usage (box, ~/rustmig/poolinit/verify): mutants/run_ctest.sh NAME...
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
ulimit -v 16000000
mkdir -p out/ctest
for n in "$@"; do
  d=$PWD/work/mut/$n
  [ -d "$d" ] || { echo "$n NO DIR"; continue; }
  ( cd "$d" && nice -n 10 timeout 1200 cargo test --release 2>&1 | grep -E "^test |test result|panicked|error(\[|:)|warning: unused" > "$OLDPWD/out/ctest/$n.txt"; echo "rc=${PIPESTATUS[0]}" >> "$OLDPWD/out/ctest/$n.txt" )
  echo "$n: $(grep -c '^test .* ok$' out/ctest/$n.txt) ok, $(grep -c '^test .* FAILED$' out/ctest/$n.txt) FAILED  $(grep 'test result' out/ctest/$n.txt | head -1 | cut -c1-90) $(grep rc= out/ctest/$n.txt)"
done
