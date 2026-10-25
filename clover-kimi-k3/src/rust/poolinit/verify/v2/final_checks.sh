#!/bin/bash
# Driver v2, last checks (box, ~/rustmig/poolinit/verify; output out/v2/final.txt):
#   1. M117 (field order of pi_expert_stats; the .so is byte-identical to the unmodified crate's): still caught by `cargo test`, as in the study
#   2. a control run under heavy load (JOBS 64, 5 x the default counts + scenario A = 15,060 units) to look for a false alarm of the gates (5 s real-time timeouts)
#   3. the toolchain and `cargo build --release` of the real crate at the very end: rustc --version, md5 of both libraries and of the driver, then a last default control run with that library
export PATH=$HOME/.cargo/bin:$PATH
cd "$(dirname "$0")/.."
O=out/v2/final.txt
{
echo "== 1. cargo test on M117 ($(date -u +%T))"
( cd work/mut/M117 && nice -n 10 cargo test --release 2>&1 | grep -E "^test |test result|panicked|FAILED" | cut -c1-200; rm -rf target )
echo "== 2. control under load: JOBS=64, 5 x counts + A ($(date -u +%T)); load: $(uptime | sed 's/.*load/load/')"
JOBS=64 bash v2/ctl.sh stress 5
echo "== 3. toolchain and final build ($(date -u +%T))"
rustc --version; cargo --version
( cd .. && nice -n 10 cargo build --release 2>&1 | tail -2 && md5sum target/release/libpoolinit.so ref/libpoolinit_ref.so verify/pi_driver && md5sum include/poolinit.h src/*.rs && nm -D --defined-only target/release/libpoolinit.so | awk '{print $3}' | tr '\n' ' '; echo )
echo "== 3b. last default control with the library just built ($(date -u +%T))"
bash v2/ctl.sh last 1
echo "== done $(date -u +%T): $(df -h / | tail -1)"
} > $O 2>&1
