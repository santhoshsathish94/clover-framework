#!/bin/bash
# Two explanations fit the first sweep: width (more layers is worse) or position
# (the front is fatal). These separate them.
#
#   36-56   21 layers in the middle, width-matched against the early band that died
#   30-60   31 layers, wider middle
#   25-92   68 layers, everything EXCEPT the front
#   2-52    51 layers, includes the front
#   47-92   46 layers, back half
#
# If 36-56 survives while 2-24 did not, width is not the driver. If 25-92 survives,
# the front is the whole story. Run 1 is an anchor: same build, same prompt, must
# reproduce the previous sweep or the comparison is void.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/band-sweep.log
WANT=${1:-8}
rm -f /tmp/band.done
: > "$LOG"
exec >>"$LOG" 2>&1

run() {
    label=$1; set_to=$2; shift 2
    pkill -f '[b]in/pipeline-serve'; sleep 2
    rm -f /tmp/bs.in /tmp/bs.out /tmp/bs.err
    mkfifo /tmp/bs.in
    (
        exec 9<>/tmp/bs.in
        ulimit -l unlimited
        export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
        export CLOVER_HARDMAX_LAYERS="$set_to"
        cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/bs.in >/tmp/bs.out 2>/tmp/bs.err
    ) &
    for _ in $(seq 1 300); do grep -q READY /tmp/bs.out 2>/dev/null && break; sleep 1; done
    printf '%s %s\n' "$WANT" "$*" > /tmp/bs.in
    for _ in $(seq 1 1800); do grep -q '^REQUEST' /tmp/bs.out 2>/dev/null && break; sleep 2; done
    sleep 2
    pkill -f '[b]in/pipeline-serve'; sleep 1
    printf '%-26s %s\n' "$label" "$(grep -A4 '^REQUEST' /tmp/bs.out | cat -v | head -3 | tr '\n' ' ')"
    rm -f /tmp/bs.in
}

FR='1008 10484 318 15383 387'
echo "started $(date -u +%FT%TZ), $WANT tokens, prompt: the capital of france is"
echo "previous sweep, for comparison:"
echo "  L46 only      [ Paris. The Eiffel Tower is located]"
echo "  middle 40-52  [ Paris. The city is located in the]"
echo "  early 2-24    [   ]"
echo "  late 70-92    [ the address of  thelink The]"
echo "  all 2-92      [<|close|>,.  ,+,,]"
echo

run "anchor: L46 only"       46     $FR
run "middle 36-56 (21)"      36-56  $FR
run "middle 30-60 (31)"      30-60  $FR
run "all but front 25-92"    25-92  $FR
run "with front 2-52 (51)"   2-52   $FR
run "back half 47-92 (46)"   47-92  $FR

echo
echo "finished $(date -u +%FT%TZ)"
touch /tmp/band.done
