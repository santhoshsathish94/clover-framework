#!/bin/bash
# The complete re-run, in order: release build, cargo test, md5 of the final .so, both smoke tests, the differential matrix. Everything is also written to out-final/final.log.
# Run on the box: ~/rustmig/loader/verify/final.sh [N]   (N = seeds of the cheap scenarios, default 200; at most 20 units at a time, all under nice)
cd "$(dirname "$0")"
export PATH=$HOME/.cargo/bin:$PATH
N=${1:-200}
OUT=out-final
rm -rf $OUT; mkdir -p $OUT
{
  echo "== date $(date -Is)  host $(hostname)  kernel $(uname -r)  cargo $(cargo --version)  rustc $(rustc --version)"
  echo "== build (cargo build --release; cargo test does not rebuild the .so)"
  ./build.sh 2>&1 | tail -12
  echo "== cargo test"
  (cd .. && nice -n 10 cargo test 2>&1 | grep -E "^test result|^running|FAILED|panicked|warning" )
  echo "== final md5"
  md5sum ../target/release/libloader.so ../ref/libloader_ref.so ld_driver ../include/loader.h ../src/*.rs ld_driver.cpp ../ref/loader_ref.cpp ../ref/loader_a.inc ../ref/loader_b.inc
  echo "== smoke_ref"
  (cd ../ref && ./run_smoke.sh ./smoke_ref)
  echo "== smoke_rust"
  (cd .. && ./smoke_rust.sh)
  echo "== matrix N=$N"
  ./run.sh $N $OUT
  echo "== coverage statistics of the seeds (stats.sh)"
  ./stats.sh $OUT
  echo "== leftovers of this run in /tmp and /dev/shm (must be empty)"
  ls -d /tmp/ldloader-verify.* /dev/shm/ldloader-verify.* /tmp/ldsmoke* 2>/dev/null
  echo "== done $(date -Is)"
} 2>&1 | tee $OUT/final.log
