#!/usr/bin/env bash
# One layer holds, sixty-eight collapse. Find where it turns, by substituting a growing
# number of layers. Two orderings, because which layers are chosen may matter as much as
# how many: shallowest-first and most-tolerant-first, the latter taken from the per-layer
# sweep's cosines.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
IDS="1008 10484 318 15383 387"
OUT=/tmp/mc-cum
BASE=/tmp/fid-baseline/france.92.bin

MLA=" 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 92 "
SHALLOW=""
for L in $(seq 1 92); do
    case "$MLA" in *" $L "*) continue ;; esac
    SHALLOW="$SHALLOW $L"
done
# Most tolerant first: highest cosine in the per-layer sweep.
TOLERANT=$(awk '/^ *[0-9]+ \[/ {print $(NF-1), $1}' /tmp/mc-sweep.log | sort -rg | awk '{printf "%s ", $2}')

rm -rf "$OUT"; mkdir -p "$OUT/sel"
cd "$CODE/pipeline" || exit 1

run_set() {
    local label="$1" count="$2"; shift 2
    rm -f "$OUT/sel"/*.f32
    local n=0
    for L in $@; do
        [ $n -ge $count ] && break
        cp "/tmp/ca/t.$L.f32" "$OUT/sel/t.$L.f32" 2>/dev/null && n=$((n+1))
    done
    env CLOVER_VECTORS="$OUT/v" CLOVER_VECTORS_OWNER=92 CLOVER_CONST_ATTN="$OUT/sel/t" \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $IDS' | ./bin/pipeline-serve $CODE serve" > "$OUT/$label-$count.log" 2>&1
    local token cos same
    token=$(perl -0777 -ne 'if (/OK [\d.]+s \[(.*?)\]\s*END/s) { $t=$1; $t =~ s/\n/\\n/g; print $t }' "$OUT/$label-$count.log")
    cos=$(/tmp/vector-compare "$BASE" "$OUT/v.92.bin" 0 4 2>/dev/null | cut -f1)
    same=no; [ "[$token]" = "[ Paris]" ] && same=yes
    printf '%-10s %3d  %-24s %-10s %s\n' "$label" "$n" "[$token]" "${cos:-NA}" "$same"
    rm -f "$OUT/v.92.bin"
}

printf '%-10s %3s  %-24s %-10s %s\n' order n token cos_p92 same
for c in 2 4 8 16 24 32 48 68; do run_set shallow  "$c" $SHALLOW;  done
for c in 2 4 8 16 24 32 48 68; do run_set tolerant "$c" $TOLERANT; done
echo "=== cumulative done ==="
