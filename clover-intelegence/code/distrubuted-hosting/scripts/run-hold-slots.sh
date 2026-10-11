#!/bin/bash
# Slots 1 to 6 are appended at layers 12 to 72 and their cross-prompt cosine runs
# 0.9149 to 0.999999. If they carry nothing the answer depends on, holding one
# stored copy should leave the output alone. Slot 0 (layer 1, cosine 0.0515) and
# slot 7 (0.8654) stay live. Hardmax default throughout, because that is the model.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/hold-slots.log
WANT=${1:-16}
DONOR=/tmp/hold-donor
: > "$LOG"
exec >>"$LOG" 2>&1

serve() {
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/hs.in /tmp/hs.out /tmp/hs.err
    mkfifo /tmp/hs.in
    (
        exec 9<>/tmp/hs.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/hs.in >/tmp/hs.out 2>/tmp/hs.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/hs.out 2>/dev/null && break; sleep 1; done
}

finish() {
    label=$1
    for _ in $(seq 1 3600); do grep -q '^REQUEST' /tmp/hs.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-22s %s\n' "$label" "$(grep '^REQUEST' /tmp/hs.out | cat -v | head -1)"
    rm -f /tmp/hs.in
}

echo "started $(date -u +%FT%TZ), $WANT tokens, holding slots 1-6"
echo

rm -f "$DONOR".*.bin
export CLOVER_VECTORS="$DONOR" CLOVER_VECTORS_KEEP=snapshots
serve
printf '1 1008 10484 318 15383 387\n' > /tmp/hs.in
finish "donor capture"
unset CLOVER_VECTORS CLOVER_VECTORS_KEEP
echo "donor bytes $(du -cb $DONOR.*.bin 2>/dev/null | tail -1 | cut -f1), files $(ls $DONOR.*.bin 2>/dev/null | wc -l)"
echo

pair() {
    name=$1; shift
    echo "########## $name ##########"
    serve;  printf '%s %s\n' "$WANT" "$*" > /tmp/hs.in; finish "$name  live"
    export CLOVER_FREEZE="$DONOR" CLOVER_FREEZE_FROM=1 CLOVER_FREEZE_TO=6
    serve;  printf '%s %s\n' "$WANT" "$*" > /tmp/hs.in; finish "$name  held 1-6"
    unset CLOVER_FREEZE CLOVER_FREEZE_FROM CLOVER_FREEZE_TO
    echo
}

pair france 1008 10484 318 15383 387
pair japan  1008 10484 318 10417 387
pair rain   91019 25528
echo "finished $(date -u +%FT%TZ)"
