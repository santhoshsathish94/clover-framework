#!/usr/bin/env bash
# Sends N copies of one identical prompt to an already-running fleet and reports what
# the machine did. The fleet is never restarted between steps, so the only thing that
# changes from one step to the next is how many requests are in flight.
set -uo pipefail
N=${1:?count required}
PROMPT='2 91019 25528'
OUT=/tmp/obs.out

before=$(grep -c '^REQUEST' "$OUT" 2>/dev/null); before=${before:-0}
pid=$(pgrep -f "bin/pipeline-serve" | head -1)
read -r _ _ rd_before _ < <(awk '$3=="md2"{print $3,$4,$6,$7}' /proc/diskstats)
mem_before=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)

cpu_ticks() { awk '{print $14+$15}' "/proc/$1/stat" 2>/dev/null; }
started=$(date +%s.%N)
ticks_before=$(cpu_ticks "$pid")
for _ in $(seq 1 "$N"); do printf '%s\n' "$PROMPT"; done > /tmp/obs.in
while :; do
    done_now=$(grep -c '^REQUEST' "$OUT" 2>/dev/null); done_now=${done_now:-0}
    [ "$done_now" -ge "$((before + N))" ] && break
    sleep 2
done
ticks_after=$(cpu_ticks "$pid")
finished=$(date +%s.%N)

read -r _ _ rd_after _ < <(awk '$3=="md2"{print $3,$4,$6,$7}' /proc/diskstats)
wall=$(echo "$finished - $started" | bc)
mb=$(echo "scale=1; ($rd_after - $rd_before) * 512 / 1048576" | bc)

echo "=== $N concurrent ==="
grep '^REQUEST' "$OUT" | tail -n "$N" | sed 's/^/  /'
printf '  wall %8.2f s   throughput %s req/s\n' \
    "$wall" "$(echo "scale=4; $N / $wall" | bc)"
printf '  cpu %s%% of 3200 during this step   disk read %s MB   mem avail %s -> %s GB\n' \
    "$(echo "scale=0; ($ticks_after - $ticks_before) / $wall" | bc)" \
    "$mb" "$mem_before" "$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo)"
