#!/usr/bin/env bash
# Compare a method run against the baseline: token identity, and the cosine of pod 92's
# output. The cosine is the sensitive measure; the token is the one that matters.
# usage: compare.sh <tag>
set -uo pipefail
TAG=${1:?tag required}
B=/tmp/fid-baseline
M=/tmp/fid-$TAG

[ -d "$B" ] || { echo "no baseline at $B"; exit 1; }
[ -d "$M" ] || { echo "no run at $M"; exit 1; }

printf '%-10s %-24s %-24s %-9s %s\n' prompt baseline "$TAG" "cos p92" same
same=0; total=0; worst=2
while IFS=$'\t' read -r name tok; do
    [ -z "$name" ] && continue
    mtok=$(awk -F'\t' -v n="$name" '$1==n{print $2}' "$M/tokens.tsv")
    # Last position differs per prompt, so ask for the highest one both runs recorded.
    best=""
    for p in 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0; do
        c=$(/tmp/vector-compare "$B/$name.92.bin" "$M/$name.92.bin" 0 $p 2>/dev/null | cut -f1)
        case "$c" in ""|NA*) continue ;; esac
        best=$c; break
    done
    mark=no
    if [ "$tok" = "$mtok" ]; then mark=yes; same=$((same+1)); fi
    total=$((total+1))
    case "$best" in ''|NA*) ;; *) awk -v a="$best" -v b="$worst" 'BEGIN{exit !(a<b)}' && worst=$best ;; esac
    printf '%-10s %-24s %-24s %-9s %s\n' "$name" "$tok" "${mtok:-MISSING}" "${best:-NA}" "$mark"
done < "$B/tokens.tsv"

echo
echo "tokens identical: $same of $total   worst pod-92 cosine: $worst"
