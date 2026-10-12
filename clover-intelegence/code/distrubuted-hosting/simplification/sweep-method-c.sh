#!/usr/bin/env bash
# Substitute one layer at a time and record what happens. One layer held and all
# sixty-eight collapsed, so the interesting thing is between them: which individual
# layers tolerate a constant and which do not.
#
# One prompt per layer to keep the sweep under an hour; only owner 92 is traced, so the
# cosine against the baseline is available as well as the token. Layers that look
# interesting get the full twelve-prompt treatment afterwards.
set -uo pipefail
ulimit -c 0
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
IDS="1008 10484 318 15383 387"
OUT=/tmp/mc-sweep
BASE=/tmp/fid-baseline/france.92.bin

MLA=" 3 7 11 15 19 23 27 31 35 39 43 47 51 55 59 63 67 71 75 79 83 87 91 92 "
rm -rf "$OUT"; mkdir -p "$OUT/one"
cd "$CODE/pipeline" || exit 1

printf '%4s %-26s %-10s %s\n' pod token cos_p92 same
for L in $(seq 1 92); do
    case "$MLA" in *" $L "*) continue ;; esac
    [ -f "/tmp/ca/t.$L.f32" ] || continue
    rm -f "$OUT/one"/*.f32
    cp "/tmp/ca/t.$L.f32" "$OUT/one/t.$L.f32"

    env CLOVER_VECTORS="$OUT/v$L" CLOVER_VECTORS_OWNER=92 CLOVER_CONST_ATTN="$OUT/one/t" \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=false \
        sh -c "echo '1 $IDS' | ./bin/pipeline-serve $CODE serve" > "$OUT/l$L.log" 2>&1

    token=$(perl -0777 -ne 'if (/OK [\d.]+s \[(.*?)\]\s*END/s) { $t=$1; $t =~ s/\n/\\n/g; print $t }' "$OUT/l$L.log")
    cos=$(/tmp/vector-compare "$BASE" "$OUT/v$L.92.bin" 0 4 2>/dev/null | cut -f1)
    same=no; [ "[$token]" = "[ Paris]" ] && same=yes
    printf '%4s %-26s %-10s %s\n' "$L" "[$token]" "${cos:-NA}" "$same"
    rm -f "$OUT/v$L.92.bin"
done
echo "=== sweep done ==="
