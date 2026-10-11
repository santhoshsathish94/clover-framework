#!/bin/bash
# Which folds should take the winner whole? One layer helps and ninety-one collapse;
# nothing between has been measured. Each run is the same prompt, same build, only
# CLOVER_HARDMAX_LAYERS differs. Writes /tmp/sweep.done at the end so the caller can
# wait on a marker instead of grepping for a process name its own command line
# contains.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/hardmax-sweep.log
WANT=${1:-8}
rm -f /tmp/sweep.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; set_to=$2; shift 2
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/hs.in /tmp/hs.out /tmp/hs.err
    mkfifo /tmp/hs.in
    (
        exec 9<>/tmp/hs.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        case "$set_to" in
            softmax) export CLOVER_SOFTMAX=1 ;;
            default) : ;;
            *)       export CLOVER_HARDMAX_LAYERS="$set_to" ;;
        esac
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/hs.in >/tmp/hs.out 2>/tmp/hs.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/hs.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/hs.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/hs.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-22s %s\n' "$label" "$(grep -A4 '^REQUEST' /tmp/hs.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/hs.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo

run "softmax (none)"       softmax      $FR
run "L46 only (default)"   default      $FR
run "snapshot layers"      12,24,36,48,60,72,84 $FR
run "middle 40-52"         40-52        $FR
run "early 2-24"           2-24         $FR
run "late 70-92"           70-92        $FR
run "all 2-92"             2-92         $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/sweep.done
