#!/usr/bin/env bash
# Is 7.6 GB/s the device's ceiling, or only what the pipeline's access pattern reaches?
#
# WHY THIS RUN EXISTS
# Instrumenting the pipeline showed the device is busy ~12.7 s of a 34-57 s request and
# moves 97 GB at 7.6 GB/s while busy -- and that busy time is the SAME at one lane and
# at four. That bounds any I/O-side change: it cannot save time the device is not
# spending. But it does not yet say how much of those 12.7 s are recoverable. If the
# hardware tops out near 7.6 GB/s then 12.7 s is irreducible and an arena has nothing to
# win. If the hardware does far better under deeper queueing, the pipeline is leaving
# bandwidth on the floor.
#
# FIXED AFTER A BAD FIRST RUN. The first version passed iflag=skip_bytes with offsets
# that were not 4096-aligned, so O_DIRECT reads failed with EINVAL -- and stderr was
# sent to /dev/null, so the failures surfaced as "2.00 GB in 0.00 s = 1004 GB/s" rather
# than as errors. Two lessons are baked in below: skip is counted in whole blocks so
# alignment is structural, and bytes moved are taken from what dd REPORTS rather than
# from what the loop assumed it asked for. Failed workers are counted and printed.
# A SECOND bad run followed: root-0 does not exist, and offsets past worker 7 ran off
# the end of the file, so byte totals silently plateaued at 11.64 GB while the rate
# still looked plausible. Both were visible only because the instrument now reports
# failures. An instrument that cannot fail loudly is not an instrument.
set -u

DS=/opt/clover-k3/clover-intelegence/dataset
PER_MB=${PER_MB:-2048}
# dataset dirs are root-1 .. root-92 (there is no root-0), and each experts.direct is
# 15722348544 B = 14994 MiB, so a 2048 MiB read must start below 12946 MiB.
NROOT=92
FILE_MB=14994
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "device: $(lsblk -d -o NAME,MODEL | grep nvme | tr '\n' ' ')"
echo "O_DIRECT reads from $DS/root-N/experts.direct, $PER_MB MiB per worker"
echo "offsets are whole 1 MiB blocks, so every read is 4096-aligned by construction"
echo

run_at() {
    local p=$1 label=$2 stride=$3
    rm -f "$TMP"/w.*
    sync
    local t0 t1 i=0 slots=$(( (FILE_MB - 1024) / PER_MB ))
    t0=$(date +%s.%N)
    while [ $i -lt "$p" ]; do
        dd if="$DS/root-$(( 1 + (i * stride) % NROOT ))/experts.direct" of=/dev/null \
           bs=1M count="$PER_MB" skip=$(( 1024 + (i % slots) * PER_MB )) iflag=direct \
           > "$TMP/w.$i.out" 2> "$TMP/w.$i.err" &
        i=$((i + 1))
    done
    wait
    t1=$(date +%s.%N)

    local bytes=0 failed=0 j=0 n
    while [ $j -lt "$p" ]; do
        n=$(awk '/bytes.*copied/ {print $1; found=1} END {if (!found) print 0}' "$TMP/w.$j.err")
        if [ "${n:-0}" -eq 0 ]; then
            failed=$((failed + 1))
            [ "$failed" -eq 1 ] && echo "    worker $j said: $(head -2 "$TMP/w.$j.err" | tr '\n' ' ')"
        fi
        bytes=$((bytes + ${n:-0}))
        j=$((j + 1))
    done

    local secs rate gb
    secs=$(echo "$t1 - $t0" | bc)
    gb=$(echo "scale=3; $bytes / 1073741824" | bc)
    rate=$(echo "scale=2; $gb / $secs" | bc)
    printf "%-10s parallel %2d   %8.2f GB in %6.2f s = %6.2f GB/s" "$label" "$p" "$gb" "$secs" "$rate"
    if [ "$failed" -gt 0 ]; then printf "   [%d/%d WORKERS FAILED]" "$failed" "$p"; fi
    printf "\n"
}

echo "--- consecutive offsets, one file per worker ---"
for p in 1 2 4 8 16 32; do run_at "$p" "consec" 1; done

echo
echo "--- workers spread across distant files (closer to expert selection) ---"
for p in 1 4 16 32; do run_at "$p" "spread" 7; done

echo
echo "The pipeline moves 97 GB in 12.7 s of device-busy time = 7.6 GB/s."
echo "Any I/O-side change can only shrink those 12.7 s by the ratio of the best rate"
echo "above to 7.6 -- and it cannot touch the rest of the run at all, because the"
echo "device is already idle for 75% of it."
