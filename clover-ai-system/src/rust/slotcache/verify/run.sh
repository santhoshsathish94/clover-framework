#!/bin/bash
# Rerun of the whole verification on the box. Usage: ./run.sh [candidate.so] [extra driver options]
# The candidate defaults to the normal Rust build; point it at a mutated build to check that the driver notices.
cd "$(dirname "$0")"
REF=${REF:-$HOME/rustmig/slotcache/ref/libslotcache_ref.so}
CAND=${1:-$HOME/rustmig/slotcache/target/release/libslotcache.so}
shift
OUT=${OUT:-out/run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
./diff_driver "$REF" "$CAND" --out "$OUT" --jobs ${JOBS:-40} "$@" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
echo "exit $?" >> "$OUT/stdout.txt"
echo "$OUT"
