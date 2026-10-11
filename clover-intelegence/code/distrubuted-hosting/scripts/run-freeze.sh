#!/bin/bash
# Does the loop need to produce the later snapshots, or would any prompt's do?
#
#   1. run token A alone and keep its snapshots
#   2. run token B alone, unmodified            -> baseline
#   3. run token B with A's snapshots injected  -> does B's answer move?
#
# FROM=2 leaves slots 0 and 1 alone, which are the two that differ most between
# prompts, and replaces the five that were measured at cosine 0.9997 and above.
# FROM=0 replaces everything, as the extreme case.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/freeze.log
: > "$LOG"
exec >>"$LOG" 2>&1

serve() {
    label=$1; token=$2; vectors=$3; freeze=$4; from=$5
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/fz.in /tmp/fz.out /tmp/fz.err
    mkfifo /tmp/fz.in
    (
        exec 9<>/tmp/fz.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        [ -n "$vectors" ] && export CLOVER_VECTORS="$vectors"
        [ -n "$freeze" ] && export CLOVER_FREEZE="$freeze" CLOVER_FREEZE_FROM="$from"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/fz.in >/tmp/fz.out 2>/tmp/fz.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/fz.out 2>/dev/null && break; sleep 1; done
    printf '1 %s\n' "$token" > /tmp/fz.in
    for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/fz.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-34s %s\n' "$label" "$(grep '^REQUEST' /tmp/fz.out | cat -v | head -1)"
    rm -f /tmp/fz.in
}

echo "started $(date -u +%FT%TZ)"
rm -f /tmp/fza.*.bin
serve "donor A  token 1008  capture"   1008  /tmp/fza ""       ""
echo
serve "target B token 70000 baseline"  70000 ""       ""       ""
serve "target B with A slots 2+"       70000 ""       /tmp/fza 2
serve "target B with A slots 1+"       70000 ""       /tmp/fza 1
serve "target B with A slots 0+"       70000 ""       /tmp/fza 0
echo
serve "control A token 1008 baseline"  1008  ""       ""       ""
echo "finished $(date -u +%FT%TZ)"
