#!/bin/bash
# Is the fold doing any work? Run the same prompts with the fold scoring as it is,
# and with the residual asserted at weight 1 everywhere, and read both outputs.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/fold-residual.log
WANT=${1:-16}
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; fold=$2; shift 2
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/fr.in /tmp/fr.out /tmp/fr.err
    mkfifo /tmp/fr.in
    (
        exec 9<>/tmp/fr.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        [ -n "$fold" ] && export CLOVER_FOLD="$fold"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/fr.in >/tmp/fr.out 2>/tmp/fr.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/fr.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/fr.in
    for _ in $(seq 1 3600); do grep -q '^REQUEST' /tmp/fr.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-26s %s\n' "$label" "$(grep '^REQUEST' /tmp/fr.out | cat -v | head -1)"
    rm -f /tmp/fr.in
}

echo "started $(date -u +%FT%TZ), $WANT tokens"
echo
echo "########## the capital of france is ##########"
run "france  fold scored"   ""         1008 10484 318 15383 387
run "france  fold residual" "residual" 1008 10484 318 15383 387
echo
echo "########## the capital of japan is ##########"
run "japan   fold scored"   ""         1008 10484 318 10417 387
run "japan   fold residual" "residual" 1008 10484 318 10417 387
echo
echo "########## rain ##########"
run "rain    fold scored"   ""         91019 25528
run "rain    fold residual" "residual" 91019 25528
echo "finished $(date -u +%FT%TZ)"
