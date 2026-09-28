#!/bin/bash
# Every trunk mode, cold then warm, in one pass. Cold means the page cache was
# dropped first, so the number includes reading 54.47 GB of trunk off the disk;
# warm means the run before it left whatever it left.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
OUT=build/trunk-matrix.tsv
printf 'mode\tcache\twall_s\trss_kb\ttrunk_load_s\tverdict\n' > "$OUT"

for mode in sqlite stream ram map; do
    for cache in cold warm; do
        [ "$cache" = cold ] && { sync; echo 3 > /proc/sys/vm/drop_caches; }
        v=$(./gate-raw.sh "$mode" 2>&1)
        echo "=== $mode $cache"
        echo "$v"
        echo
        w=$(printf '%s' "$v" | grep -oP 'total wall time\s+:\s+\K[0-9.]+' | head -1)
        r=$(printf '%s' "$v" | grep -oP 'Maximum resident set size \(kbytes\):\s+\K[0-9]+' | head -1)
        l=$(printf '%s' "$v" | grep -oP 'trunk -> (RAM|mmap):.*in \K[0-9.]+' | head -1)
        d=$(printf '%s' "$v" | grep -oE '^(PASS|FAIL)' | head -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$cache" "${w:-}" "${r:-}" "${l:-0}" "${d:-NORUN}" >> "$OUT"
    done
done

echo "=== matrix"
column -t "$OUT"
