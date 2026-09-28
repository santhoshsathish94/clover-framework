#!/bin/bash
# Re-run all 34 prompts on the CURRENT build: baseline and cross-layer lookahead.
# Per prompt the logits md5 of both arms must match, or the lookahead changed
# the answer and the timing is worthless.
set -u
cd /opt/clover-k3
. ./config.env

OUT=/root/k3proof
mkdir -p "$OUT" "$OUT/logs"
rm -f "$OUT/DONE" "$OUT/current.tsv"
printf 'idx\tnpos\ttok\twall_base\twall_nx\tnxwait\tsame_md5\tmd5\n' > "$OUT/current.tsv"

BASE="OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 K3_PREFETCH=4 \
K3_TRUNKRAM=0 K3_TRUNKPATH=$K3_TRUNKPATH K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 \
K3_HUGE=1 K3_INDEX=build/eqidx.bin"

while IFS=$'\t' read -r idx npos ids text; do
    [ -z "${idx:-}" ] && continue
    bin="build/ck3_$npos"
    if [ ! -x "$bin" ]; then
        gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS="$npos" \
            -o "$bin" clover-k3.c -lm -lpthread 2>>"$OUT/logs/build.err" || {
            echo "BUILD FAILED npos=$npos" >&2; continue; }
    fi

    lb="$OUT/logs/p${idx}_base.log"
    ln="$OUT/logs/p${idx}_nx.log"
    rt="$OUT/rt_${idx}.bin"
    fb="$OUT/lb_${idx}.bin"
    fn="$OUT/ln_${idx}.bin"
    rm -f "$rt" "$fb" "$fn"

    # arm 1: no lookahead, and write the routing cache the second arm needs
    env $BASE K3_IDS="$ids" K3_LOGITS="$fb" K3_ROUTESAVE="$rt" "./$bin" > "$lb" 2>&1

    # arm 2: read a layer ahead, routing re-checked against the live router
    env $BASE K3_IDS="$ids" K3_LOGITS="$fn" K3_ROUTELOAD="$rt" \
        K3_ARENA2=1 K3_NX=1 K3_NXREAD=14 "./$bin" > "$ln" 2>&1

    wb=$(grep '^total wall' "$lb" | awk '{print $5}')
    wn=$(grep '^total wall' "$ln" | awk '{print $5}')
    nw=$(grep 'waiting for the early' "$ln" | awk '{print $1}')
    tk=$(grep '^emitted token' "$ln" | awk '{print $4}')
    mb=$(md5sum "$fb" 2>/dev/null | cut -d' ' -f1)
    mn=$(md5sum "$fn" 2>/dev/null | cut -d' ' -f1)
    if [ "$mb" = "$mn" ] && [ -n "$mb" ]; then same=yes; else same=NO; fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$idx" "$npos" "${tk:-?}" "${wb:-?}" "${wn:-?}" "${nw:-?}" "$same" "${mb:0:8}" \
        >> "$OUT/current.tsv"
    echo "$(date -Is) p$idx npos=$npos tok=${tk:-?} base=${wb:-?} nx=${wn:-?} $same" \
        >> "$OUT/progress.txt"
    rm -f "$fb" "$fn" "$rt"
done < prompts.tsv

touch "$OUT/DONE"
