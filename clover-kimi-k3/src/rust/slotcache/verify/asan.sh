#!/bin/bash
# Rebuilds the C++ reference in verify/ref-asan with -fsanitize=address,undefined (sources copied unchanged from ../ref, which is not touched) and runs the whole driver
# against the normal Rust build. The driver itself is not instrumented, so libasan is preloaded (the runtime must come first in the process).
set -e
cd "$(dirname "$0")"
mkdir -p ref-asan
cp ../ref/slotcache_ref.cpp ../ref/slotcache_extracted.inc ref-asan/
g++ -O1 -g -fno-omit-frame-pointer -std=c++17 -fPIC -shared -fsanitize=address,undefined -fno-sanitize-recover=undefined -I../include ref-asan/slotcache_ref.cpp -o ref-asan/libslotcache_ref_asan.so
ASAN_SO=$(g++ -print-file-name=libasan.so)
OUT=${OUT:-out/asan-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
ldd ref-asan/libslotcache_ref_asan.so > "$OUT/ldd.txt" 2>&1 || true
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:verbosity=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
LD_PRELOAD="$ASAN_SO" ./diff_driver ref-asan/libslotcache_ref_asan.so ../target/release/libslotcache.so --out "$OUT" --jobs ${JOBS:-40} "$@" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
echo "exit $?" >> "$OUT/stdout.txt"
echo "$OUT"
