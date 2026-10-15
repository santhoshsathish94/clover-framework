#!/usr/bin/env bash
# Compare two environment settings on where CPU time goes, not on wall time.
#
# WHY THIS EXISTS
# ab-pipeline.sh answers "is it faster", and this box cannot resolve under ~0.5 s, so a
# change that moves 40 s of CPU around can come back as "no difference". That is the
# right answer about speed and the wrong answer about mechanism. This reports the
# user/system split, TLB shootdowns and fault counts instead, so a setting that clearly
# changes behaviour without changing wall time is visible as such rather than filed as
# a null.
#
# It exists because an A/B of OMP_WAIT_POLICY came back at 0.9936 against a 0.405 s
# noise floor, and the honest next question was not "is it faster" but "did it do
# anything at all". It had: it moved 37 s from system to user and left total CPU flat.
#
# TWO EARLIER BUGS IN THIS FILE, both worth remembering:
#   local policy=$1 out=/tmp/ctx-$policy      -- bash expands every word on a `local`
#       line before assigning any of them, so $policy was unset and `set -u` aborted.
#   /proc/PID/status voluntary_ctxt_switches  -- that is the MAIN THREAD ONLY, which is
#       why a 16-thread run reported ~1,500 switches. Per-thread counters live in
#       /proc/PID/task/TID/status and have to be summed.
#
# usage: compare-settings.sh LANES 'labelA|ENV=v ...' 'labelB|ENV=v ...'
#
# CMP_LIGHT=1 drops the per-thread sampling and polls once a second instead of five
# times a second. USE IT WHENEVER THE CPU TOTALS ARE THE POINT. The full mode reads
# /proc/PID/task/*/status for every thread every 200 ms, which is ~65 file reads plus
# two awk processes per sample, and that is not free: a 17-thread arm measured 60.838 s
# under full sampling against a 56.612 s median from the untouched A/B harness, a 4.2 s
# perturbation, while a 65-thread arm moved only 0.5 s. The heavy mode distorts sparse
# configurations most, which is exactly backwards from what you want when comparing
# thread counts.
# CMP_REPEAT=n runs each arm n times and prints every run, because a single run of
# anything on this box has already been shown to land 7 % off its own median.
set -u

C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LANES=${1:?lanes}
A=${2:?arm A}
B=${3:?arm B}
LIGHT=${CMP_LIGHT:-0}
REPEAT=${CMP_REPEAT:-1}
INTERVAL=0.2
[ "$LIGHT" = 1 ] && INTERVAL=1

cd "$C/pipeline" || exit 1
: > /tmp/cmp-in.txt
i=0; while [ $i -lt "$LANES" ]; do echo "1 1008 10484 318 15383 387" >> /tmp/cmp-in.txt; i=$((i+1)); done

tlb_total() {
    awk '/^[ \t]*TLB:/ {s=0; for (i=2; i<=NF; i++) if ($i ~ /^[0-9]+$/) s+=$i; print s}' /proc/interrupts
}

probe() {
    local spec=$1
    local name=${spec%%|*}
    local settings=${spec#*|}
    local out=/tmp/cmp-$name

    local tlb0
    tlb0=$(tlb_total)
    # defaults FIRST so an arm can override any of them, including OMP_NUM_THREADS
    # shellcheck disable=SC2086
    env OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores $settings \
        ./bin/pipeline-serve "$C" serve < /tmp/cmp-in.txt > "$out.out" 2>&1 &
    local pid=$!

    local vol=0 nonvol=0 ut=0 st=0 maxth=0 minf=0 majf=0
    : > /tmp/cmp-cpus.$name
    while kill -0 "$pid" 2>/dev/null; do
        local s v n th
        s=$(awk '{print $10" "$12" "$14" "$15" "$20}' "/proc/$pid/stat" 2>/dev/null)
        if [ -n "${s:-}" ]; then
            minf=$(echo "$s" | cut -d' ' -f1); majf=$(echo "$s" | cut -d' ' -f2)
            ut=$(echo "$s" | cut -d' ' -f3);   st=$(echo "$s" | cut -d' ' -f4)
            th=$(echo "$s" | cut -d' ' -f5)
            [ "${th:-0}" -gt "$maxth" ] && maxth=$th
        fi
        # which CPUs are the threads actually ON? field 39 of each task's stat is the
        # last CPU it ran on. If four teams are binding onto overlapping cores, the
        # distinct count stays well below the core count while threads outnumber cores.
        if [ "$LIGHT" != 1 ]; then
            awk '{print $39}' /proc/"$pid"/task/*/stat 2>/dev/null >> /tmp/cmp-cpus."$name"
            # per-thread, summed: /proc/PID/status counts the main thread only
            v=$(cat /proc/"$pid"/task/*/status 2>/dev/null | awk '/^voluntary_ctxt_switches/{s+=$2} END{print s+0}')
            n=$(cat /proc/"$pid"/task/*/status 2>/dev/null | awk '/^nonvoluntary_ctxt_switches/{s+=$2} END{print s+0}')
            [ "${v:-0}" -gt "$vol" ] && vol=$v
            [ "${n:-0}" -gt "$nonvol" ] && nonvol=$n
        fi
        sleep "$INTERVAL"
    done
    wait "$pid"
    local tlb1
    tlb1=$(tlb_total)

    local req
    req=$(awk '/^REQUEST 0 OK/ { sub(/s$/, "", $4); print $4; exit }' "$out.out")
    if [ -z "$req" ]; then
        echo "  $name: *** FAILED, no REQUEST line -- $(tail -1 "$out.out")"
        return
    fi
    local distinct
    distinct=$(sort -u /tmp/cmp-cpus."$name" 2>/dev/null | grep -c '^[0-9]*$')
    printf "  %-10s request %8ss   user %8.2f s   sys %8.2f s   total %8.2f s\n" \
        "$name" "$req" "$(echo "scale=2; $ut/100" | bc)" "$(echo "scale=2; $st/100" | bc)" \
        "$(echo "scale=2; ($ut+$st)/100" | bc)"
    if [ "$LIGHT" = 1 ]; then
        printf "             cores busy %6.2f over the request   peak threads %3d   [light sampling]\n" \
            "$(echo "scale=3; ($ut+$st)/100/$req" | bc)" "$maxth"
    else
        printf "             cores busy %6.2f over the request   peak threads %3d   distinct CPUs used %3d of %d\n" \
            "$(echo "scale=3; ($ut+$st)/100/$req" | bc)" "$maxth" "$distinct" "$(nproc)"
        printf "             ctxt switches: voluntary %11d   nonvoluntary %9d\n" \
            "$vol" "$nonvol"
    fi
    printf "             TLB shootdowns %12d   minflt %10d   majflt %9d\n" \
        "$((tlb1 - tlb0))" "$minf" "$majf"
}

echo "lanes=$LANES, $REPEAT run(s) per arm, light sampling=$LIGHT."
echo "This measures MECHANISM, not speed -- use ab-pipeline.sh for timing, which needs"
echo "four balanced rounds to resolve 0.5 s on this box."
echo
r=1
while [ "$r" -le "$REPEAT" ]; do
    [ "$REPEAT" -gt 1 ] && echo "run $r:"
    probe "$A"
    probe "$B"
    r=$((r + 1))
done
