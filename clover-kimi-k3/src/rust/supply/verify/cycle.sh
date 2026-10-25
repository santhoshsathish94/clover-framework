#!/bin/bash
# The whole build-and-test cycle on the box, in ~/rustmig/supply (no GPU, no sudo): C++ reference + its smoke test, Rust release build + cargo test + its smoke test, the differential driver.
# Prints only the lines that matter; md5 of the final artefacts at the end.
export PATH=$HOME/.cargo/bin:$PATH
cd "$(dirname "$0")/.."
echo "== reference";   (cd ref && ./build.sh 2>&1 | tail -20 && ./smoke_ref 2>&1 | grep -E 'FAIL|checks')
echo "== cargo build"; nice -n 10 cargo build --release 2>&1 | tail -5
echo "== cargo test";  nice -n 10 cargo test 2>&1 | grep -E '^test result|FAILED|panicked|^warning'
echo "== smoke rust";  gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lsupply -Wl,-rpath,$PWD/target/release && ./smoke_rust 2>&1 | grep -E 'FAIL|checks'
echo "== driver";      (cd verify && ./build.sh 2>&1 | tail -30)
echo "== md5";         verify/md5.sh
