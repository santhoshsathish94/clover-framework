#!/bin/bash
# One pair proves little: a single token can match by chance. Same donor, four
# different targets, each run twice. If every target's answer moves to the donor's,
# the later snapshots decide the output. If some move and some do not, the effect is
# weaker than one pair suggested.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/freeze-many.log
: > "$LOG"
exec >>"$LOG" 2>&1

serve() {
    label=$1; token=$2; vectors=$3; freeze=$4; from=$5
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/fm.in /tmp/fm.out /tmp/fm.err
    mkfifo /tmp/fm.in
    (
        exec 9<>/tmp/fm.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        [ -n "$vectors" ] && export CLOVER_VECTORS="$vectors"
        [ -n "$freeze" ] && export CLOVER_FREEZE="$freeze" CLOVER_FREEZE_FROM="$from"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/fm.in >/tmp/fm.out 2>/tmp/fm.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/fm.out 2>/dev/null && break; sleep 1; done
    printf '1 %s\n' "$token" > /tmp/fm.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/fm.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-36s %s\n' "$label" "$(grep '^REQUEST' /tmp/fm.out | cat -v | sed 's/.*\[/[/' | head -1)"
    rm -f /tmp/fm.in
}

echo "started $(date -u +%FT%TZ)"
rm -f /tmp/fzd.*.bin
serve "donor  token 91019   capture" 91019 /tmp/fzd "" ""
echo
for t in 1008 15383 70000 500; do
    serve "target $t baseline" "$t" "" "" ""
    serve "target $t + donor slots 2+" "$t" "" /tmp/fzd 2
    echo
done
echo "finished $(date -u +%FT%TZ)"
