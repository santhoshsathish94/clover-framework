#!/bin/bash
# cargo test of mutant crates: `cargo test --release` (the profile the .so is built with) and, with DEBUG=1, also plain `cargo test` (debug profile: overflow checks on).
# Usage: run_tests.sh M100 M101 ...   Result lines: verify/out/mut/tests-summary.txt, raw output verify/out/mut/tests-<name>.txt (+ -debug)
export PATH=$HOME/.cargo/bin:$PATH
V=$HOME/rustmig/engcfg/verify
W=$V/work/mut
O=$V/out/mut
mkdir -p "$O"
for n in "$@"; do
  ( cd "$W/$n" && nice -n 10 cargo test --release --offline > "$O/tests-$n.txt" 2>&1; echo "exit $?" >> "$O/tests-$n.txt" )
  r=$(grep -E "^test result|FAILED|failed" "$O/tests-$n.txt" | tr '\n' ' ' | cut -c1-300)
  echo "$n release: $(tail -1 $O/tests-$n.txt) | $r" >> "$O/tests-summary.txt"
  if [ -n "$DEBUG" ]; then
    ( cd "$W/$n" && nice -n 10 cargo test --offline > "$O/tests-$n-debug.txt" 2>&1; echo "exit $?" >> "$O/tests-$n-debug.txt" )
    r=$(grep -E "^test result|FAILED|failed" "$O/tests-$n-debug.txt" | tr '\n' ' ' | cut -c1-300)
    echo "$n debug: $(tail -1 $O/tests-$n-debug.txt) | $r" >> "$O/tests-summary.txt"
  fi
  rm -rf "$W/$n/target/debug"
done
echo TESTS-DONE >> "$O/tests-summary.txt"
