#!/bin/bash
# Does warming the cache before the operator pay? O_DIRECT DMAs past every
# cache, so the stage is cold when the arithmetic reaches it. Three warms,
# each against the same gate, three reps because one run is not a number.
#   0 off   1 prefetch T2 (L2/L3)   2 prefetch T0 (L1)   3 touch every line
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
B=build
OUT=$B/cpf.tsv
BASE5=23d162dcefb18211a7540ef12948f1eb
P5="1008,10484,318,15383,387"
printf 'cpf\trep\twall_s\twarm_s\twarm_gb\tverdict\n' > "$OUT"

sync; echo 3 > /proc/sys/vm/drop_caches
# one throwaway run so every measured run sees the same warm page cache
env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 K3_PREFETCH=4 \
    K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 K3_DB=/srv/k3/db \
    K3_RAW=/srv/k3/raw K3_RAWMODE=1 K3_INDEX=$B/eqidx.bin K3_IDS="$P5" \
    K3_LOGITS=$B/cpf-prime.bin $B/csk3 > /dev/null 2>&1

for c in 0 1 2 3; do
    for rep in 1 2 3; do
        o=$B/cpf-$c.bin
        rm -f "$o"
        l=$(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 K3_PREFETCH=4 \
            K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 K3_DB=/srv/k3/db \
            K3_RAW=/srv/k3/raw K3_RAWMODE=1 K3_CPF=$c K3_INDEX=$B/eqidx.bin \
            K3_IDS="$P5" K3_LOGITS="$o" $B/csk3 2>&1)
        w=$(printf '%s' "$l" | grep -oP 'total wall time\s+:\s+\K[0-9.]+' | head -1)
        ws=$(printf '%s' "$l" | grep -oP 'cache warm.*GB walked in \K[0-9.]+' | head -1)
        wg=$(printf '%s' "$l" | grep -oP 'cache warm.*mode [0-9]+\s+\K[0-9.]+' | head -1)
        if [ -s "$o" ] && [ "$(md5sum "$o" | cut -d' ' -f1)" = "$BASE5" ]; then v=PASS; else v=FAIL; fi
        printf '%s\t%d\t%s\t%s\t%s\t%s\n' "$c" "$rep" "${w:-NORUN}" "${ws:-0}" "${wg:-0}" "$v" | tee -a "$OUT"
    done
done

echo
column -t "$OUT"
