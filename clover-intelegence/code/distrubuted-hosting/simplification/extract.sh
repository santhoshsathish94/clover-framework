#!/usr/bin/env bash
# Rebuild tokens.tsv from saved logs. The first extraction used a single-line grep and
# lost two prompts whose answers contain newlines, which looked like a failed run rather
# than a failed regex.
# usage: extract.sh <tag>
set -uo pipefail
TAG=${1:?tag required}
D=/tmp/fid-$TAG
[ -d "$D" ] || { echo "no run at $D"; exit 1; }

: > "$D/tokens.tsv"
for log in "$D"/*.log; do
    name=$(basename "$log" .log)
    token=$(perl -0777 -ne 'if (/OK [\d.]+s \[(.*?)\]\s*END/s) { $t=$1; $t =~ s/\n/\\n/g; $t =~ s/\r/\\r/g; print $t }' "$log")
    printf '%s\t[%s]\n' "$name" "${token}" >> "$D/tokens.tsv"
done
sort -o "$D/tokens.tsv" "$D/tokens.tsv"
cat "$D/tokens.tsv"
