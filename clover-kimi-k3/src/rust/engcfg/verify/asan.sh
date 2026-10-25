#!/bin/bash
# The reference rebuilt with AddressSanitizer + UndefinedBehaviorSanitizer (UBSan in recover mode: report and go on), run against the normal Rust build.
# The driver itself is not instrumented, so libasan is preloaded. Reports go to files (log_path), NOT to stderr, because the children's stderr is part of what is compared.
# Usage: ./asan.sh [driver options, e.g. --scale 0.25]   Output: out/asan-<time>/ (stdout.txt, units.log, san.* report files, summary of UBSan kinds in san-summary.txt)
set -e
cd "$(dirname "$0")"
mkdir -p ref-asan
g++ -O1 -g -std=c++17 -fPIC -shared -Wl,-Bsymbolic -fsanitize=address,undefined -I../include -I../ref -I../ref/.. ../ref/engcfg_ref.cpp -o ref-asan/libengcfg_ref_asan.so 2> ref-asan/build.log || { cat ref-asan/build.log; exit 1; }
OUT=${OUT:-out/asan-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
CAND=$HOME/rustmig/engcfg/target/release/libengcfg.so
export ASAN_OPTIONS=detect_leaks=0:handle_sigfpe=0:handle_abort=0:log_path=$PWD/$OUT/san.asan
export UBSAN_OPTIONS=print_stacktrace=0:log_path=$PWD/$OUT/san.ubsan
LD_PRELOAD=$(gcc -print-file-name=libasan.so) nice -n 10 ./diff_driver "$PWD/ref-asan/libengcfg_ref_asan.so" "$CAND" --out "$OUT" --jobs ${JOBS:-24} --e9-vm-mb 0 "$@" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" || true
echo "exit $?" >> "$OUT/stdout.txt"
rm -rf "$OUT/work"
{ echo "report files (san.*): $(ls "$OUT" | grep -c '^san\.')"
  echo "AddressSanitizer errors (files + stderr): $(cat "$OUT"/san.* "$OUT/stderr.txt" 2>/dev/null | grep -c 'ERROR: AddressSanitizer')"
  echo "UBSan reports by source location (files + stderr; E3 worker processes print to stderr.txt, children's stderr is compared and would show as mismatches):"
  cat "$OUT"/san.* "$OUT/stderr.txt" 2>/dev/null | grep 'runtime error' | sed -E 's/: runtime error.*//' | sort | uniq -c | sort -rn
  echo "UBSan reports by kind:"
  cat "$OUT"/san.* "$OUT/stderr.txt" 2>/dev/null | grep 'runtime error' | sed -E 's/^.*runtime error: //; s/-?[0-9]{4,}/N/g' | sort | uniq -c | sort -rn | head -20; } > "$OUT/san-summary.txt"
echo "$OUT"
