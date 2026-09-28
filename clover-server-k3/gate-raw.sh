#!/bin/bash
# The correctness gate for clover-server-k3, one trunk mode per invocation.
#
#   ./gate-raw.sh sqlite | stream | ram | map
#
# Every run deletes its own output first and prints the wall time, because a
# harness that can silently run nothing will one day report a stale PASS.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$HERE/build"

BASE5=23d162dcefb18211a7540ef12948f1eb
P5="1008,10484,318,15383,387"
MODE="${1:-sqlite}"

case "$MODE" in
    sqlite) RAWMODE=0 ;;
    stream) RAWMODE=1 ;;
    ram)    RAWMODE=2 ;;
    map)    RAWMODE=3 ;;
    *) echo "modes: sqlite stream ram map" >&2; exit 2 ;;
esac

OUT="$B/raw-$MODE.bin"
LOG="$B/raw-$MODE.log"
rm -f "$OUT" "$LOG"

/usr/bin/time -v env \
    OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    K3_PREFETCH=4 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
    K3_DB=/srv/k3/db K3_RAW=/srv/k3/raw K3_RAWMODE="$RAWMODE" \
    K3_INDEX="$B/eqidx.bin" K3_IDS="$P5" K3_LOGITS="$OUT" \
    ${K3_TRACE:+K3_TRACE="$K3_TRACE"} \
    "$B/csk3" > "$LOG" 2>&1

echo "--- $MODE"
grep -E "trunk ->|trunk pinned|prefetch |emitted token|total wall|trunk  +:|experts|trace  " "$LOG"
grep -E "Maximum resident set size|Elapsed \(wall clock\)" "$LOG"

if [ ! -s "$OUT" ]; then
    echo "FAIL  $OUT is missing or empty - the run did not happen"
    exit 1
fi
m=$(md5sum "$OUT" | cut -d' ' -f1)
if [ "$m" = "$BASE5" ]; then
    echo "PASS  logits md5 $m == preserved baseline"
else
    echo "FAIL  logits md5 $m != $BASE5"
    exit 1
fi
