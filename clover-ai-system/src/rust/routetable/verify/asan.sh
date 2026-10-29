#!/bin/bash
# Rebuilds the C++ reference in verify/ref-asan with -fsanitize=address,undefined (sources copied unchanged from ../ref, which is not touched) and runs the whole driver
# against the normal Rust build. The driver itself is not instrumented, so libasan is preloaded (the runtime must come first in the process).
# UBSan is built in recoverable mode (no -fno-sanitize-recover): reports go to the per-unit stderr files and the run continues; they are classified afterwards.
set -e
cd "$(dirname "$0")"
mkdir -p ref-asan
cp ../ref/routetable_ref.cpp ../ref/routetable_extracted.inc ref-asan/
g++ -O1 -g -fno-omit-frame-pointer -std=c++17 -fPIC -shared -fsanitize=address,undefined -I../include ref-asan/routetable_ref.cpp -o ref-asan/libroutetable_ref_asan.so
ASAN_SO=$(g++ -print-file-name=libasan.so)
OUT=${OUT:-out/asan-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
ldd ref-asan/libroutetable_ref_asan.so > "$OUT/ldd.txt" 2>&1 || true
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0
export UBSAN_OPTIONS=print_stacktrace=0:halt_on_error=0
LD_PRELOAD="$ASAN_SO" ./diff_driver ref-asan/libroutetable_ref_asan.so ../target/release/libroutetable.so --out "$OUT" --jobs ${JOBS:-48} "$@" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
echo "exit $?" >> "$OUT/stdout.txt"
echo "$OUT"
