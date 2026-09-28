#!/bin/bash
# Three consecutive runs in one mode, because a server runs in one mode and
# the second and third runs are the steady state the first one is not.
#   ./trunk-reps.sh stream ram
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
OUT=build/trunk-reps.tsv
printf 'mode\trep\twall_s\trss_kb\tverdict\n' > "$OUT"

for mode in "$@"; do
    sync; echo 3 > /proc/sys/vm/drop_caches
    for rep in 1 2 3; do
        v=$(./gate-raw.sh "$mode" 2>&1)
        w=$(printf '%s' "$v" | grep -oP 'total wall time\s+:\s+\K[0-9.]+' | head -1)
        r=$(printf '%s' "$v" | grep -oP 'Maximum resident set size \(kbytes\):\s+\K[0-9]+' | head -1)
        d=$(printf '%s' "$v" | grep -oE '^(PASS|FAIL)' | head -1)
        printf '%s\t%d\t%s\t%s\t%s\n' "$mode" "$rep" "${w:-}" "${r:-}" "${d:-NORUN}" | tee -a "$OUT"
    done
done

echo
column -t "$OUT"
