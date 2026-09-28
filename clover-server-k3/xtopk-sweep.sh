#!/bin/bash
# Does the answer survive on fewer than 16 experts, or did one easy prompt
# flatter it? Every prompt, four values of k, token compared against that same
# prompt's own k=16 run. NPOS is compile-time, so one binary per length.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
B=build
OUT=$B/xtopk-sweep.tsv
printf 'id\tnpos\tk\twall_s\tx_s\tmean_m\ttoken\tsame\trelL2\ttop8\ttext\n' > "$OUT"

for n in $(awk -F'\t' '{print $2}' prompts.tsv | sort -nu); do
    gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS="$n" \
        -o "$B/csk3-$n" clover-server-k3.c -lm -lsqlite3 -lpthread 2>/dev/null || {
        echo "build failed for NPOS=$n" >&2; continue; }

    awk -F'\t' -v n="$n" '$2==n {print $1"\t"$3"\t"$4}' prompts.tsv | while IFS=$'\t' read -r id ids text; do
        for k in 16 12 8 4; do
            o=$B/xt-$id-$k.bin
            rm -f "$o"
            l=$( { /usr/bin/time -f "TIME %e" env \
                OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=16 \
                K3_PREFETCH=4 K3_NREADER=14 K3_XDEC=2 K3_PLGRAN=1 K3_HUGE=1 \
                K3_DB=/srv/k3/db K3_RAW=/srv/k3/raw K3_RAWMODE=1 K3_RA=2 K3_RABUDGET=512 \
                K3_XTOPK=$k K3_INDEX=$B/eqidx.bin K3_IDS="$ids" K3_LOGITS="$o" \
                "$B/csk3-$n"; } 2>&1 )
            w=$(printf '%s' "$l" | grep -oP 'total wall time\s+:\s+\K[0-9.]+' | head -1)
            x=$(printf '%s' "$l" | grep -oP '^X   mxfp4 expert proj\s+\K[0-9.]+' | head -1)
            mm=$(printf '%s' "$l" | grep -oP 'mean m \K[0-9.]+' | head -1)
            tok=$(printf '%s' "$l" | grep -oP 'emitted token\s+:\s+\K[0-9]+' | head -1)
            if [ "$k" = 16 ]; then
                ref=$tok; rel=0.0000; t8=8; same=REF
                cp -f "$o" "$B/ref-$id.bin"
            else
                ref=$(head -1 "$B/tok-$id.txt" 2>/dev/null || echo "$tok")
                c=$(python3 cmp_logits.py "$B/ref-$id.bin" "$o" 2>/dev/null)
                rel=$(printf '%s' "$c" | grep -oP 'rel L2     : \K[0-9.]+')
                t8=$(printf '%s' "$c" | grep -oP 'top-8 kept : \K[0-9]+')
                [ "$tok" = "$ref" ] && same=SAME || same=CHANGED
            fi
            [ "$k" = 16 ] && echo "$tok" > "$B/tok-$id.txt"
            rm -f "$o"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$id" "$n" "$k" "${w:-}" "${x:-}" "${mm:-}" "${tok:-}" \
                "$same" "${rel:-}" "${t8:-}" "$text" >> "$OUT"
            echo "id=$id npos=$n k=$k wall=${w:-?} tok=${tok:-?} $same relL2=${rel:-?} top8=${t8:-?}"
        done
        rm -f "$B/ref-$id.bin"
    done
done
echo "=== done"
