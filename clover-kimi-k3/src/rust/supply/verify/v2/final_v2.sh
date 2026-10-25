#!/bin/bash
# Final check of driver v2 on the unmodified crate (run after the mutant campaign): cargo build --release + cargo test, the Rust smoke test, the driver build, md5 of everything (md5.sh prints the
# toolchain versions too), then three full differential runs (all 18 scenarios, 1120 units) with a 3 GB address-space cap per process. Log: out/v2final.log; out/v2final.done at the end.
# The reference library is NOT rebuilt (ref/libsupply_ref.so is used as it is).
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
S=$(cd .. && pwd)
rm -rf out/v2final-1 out/v2final-2 out/v2final-3 out/v2final.done
{
  date -u
  echo "== cargo build";  (cd "$S" && nice -n 10 cargo build --release 2>&1 | tail -3)
  echo "== cargo test";   (cd "$S" && nice -n 10 cargo test 2>&1 | grep -E '^test result|FAILED|panicked|^warning')
  echo "== smoke rust";   (cd "$S" && gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lsupply -Wl,-rpath,$S/target/release && ./smoke_rust 2>&1 | grep -E 'FAIL|checks')
  echo "== driver build"; ./build.sh 2>&1 | tail -5
  echo "== md5";          ./md5.sh
  for i in 1 2 3; do
    echo "== differential run v2final-$i (3 GB cap)"
    ( ulimit -v 3145728; ./run.sh v2final-$i --keep-going > out/v2final-$i.stdout 2>&1; echo "exit status $?" )
    grep -v coverage "out/v2final-$i/summary.txt" | cut -c1-300
  done
  date -u
} > out/v2final.log 2>&1
echo done > out/v2final.done
