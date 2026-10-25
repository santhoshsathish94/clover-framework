#!/bin/bash
# Rerun of the whole verification on the box. Usage: ./run.sh [candidate.so] [extra driver options]
# The candidate defaults to the normal Rust build; point it at another build to check that the driver notices.
# Env: REF (reference .so), JOBS (default 24, the driver caps it at 24), OUT (output directory). Runs at nice 10.
cd "$(dirname "$0")"
REF=${REF:-$HOME/rustmig/engcfg/ref/libengcfg_ref.so}
CAND=${1:-$HOME/rustmig/engcfg/target/release/libengcfg.so}
shift
OUT=${OUT:-out/run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
nice -n 10 ./diff_driver "$REF" "$CAND" --out "$OUT" --jobs ${JOBS:-24} "$@" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
echo "exit $?" >> "$OUT/stdout.txt"
rm -rf "$OUT/work"
echo "$OUT"
