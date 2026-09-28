#!/bin/bash
# Read-ahead, three ways, three runs each. One run of anything on this box is
# worth about +/- 1 s, so a single number decides nothing.
#   0 = none   1 = reader thread   2 = io_uring
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
B=build
BASE5=23d162dcefb18211a7540ef12948f1eb
P5="1008,10484,318,15383,387"
OUT=$B/ra-reps.tsv
printf 'mode\tbudget\trep\twall_s\tuser_s\tsys_s\thidden_s\twaited_s\tverdict\n' > "$OUT"

sync; echo 3 > /proc/sys/vm/drop_caches
env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 K3_PREFETCH=4 \
    K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 K3_DB=/srv/k3/db \
    K3_RAW=/srv/k3/raw K3_RAWMODE=1 K3_INDEX=$B/eqidx.bin K3_IDS="$P5" \
    K3_LOGITS=$B/prime.bin $B/csk3 > /dev/null 2>&1

one() {   # mode budget rep
    local o=$B/rr.bin; rm -f "$o"
    local l
    l=$( { /usr/bin/time -f "TIME %U %S %e" env \
        OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 K3_PREFETCH=4 \
        K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 K3_DB=/srv/k3/db \
        K3_RAW=/srv/k3/raw K3_RAWMODE=1 K3_RA=$1 K3_RABUDGET=$2 K3_RANT=8 \
        K3_INDEX=$B/eqidx.bin K3_IDS="$P5" K3_LOGITS="$o" $B/csk3; } 2>&1 )
    local w u s e hid wai v
    w=$(printf '%s' "$l" | grep -oP 'total wall time\s+:\s+\K[0-9.]+' | head -1)
    read -r u s e <<< "$(printf '%s' "$l" | grep -oP '^TIME \K.*' | head -1)"
    hid=$(printf '%s' "$l" | grep -oP 'in flight under the arithmetic, \K[0-9.]+' | head -1)
    wai=$(printf '%s' "$l" | grep -oP 'arithmetic, [0-9.]+ s of it waited on' | grep -oP '\K[0-9.]+(?= s of it)')
    [ -z "${wai:-}" ] && wai=$(printf '%s' "$l" | grep -oP '\K[0-9.]+(?= s spent waiting)' | head -1)
    if [ -s "$o" ] && [ "$(md5sum "$o" | cut -d' ' -f1)" = "$BASE5" ]; then v=PASS; else v=FAIL; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "${w:-NORUN}" "${u:-}" "${s:-}" "${hid:-0}" "${wai:-0}" "$v" | tee -a "$OUT"
}

for rep in 1 2 3; do one 0 0    $rep; done
for rep in 1 2 3; do one 2 512  $rep; done
for rep in 1 2 3; do one 2 2048 $rep; done
for rep in 1 2 3; do one 1 1024 $rep; done

echo
column -t "$OUT"
