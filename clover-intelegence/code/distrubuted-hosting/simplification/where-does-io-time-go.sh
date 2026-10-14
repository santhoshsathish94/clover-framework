#!/usr/bin/env bash
# Where does the 26.6 s constant in T = 26.6 + 7.45*lanes actually go?
#
# WHY THIS RUN EXISTS
# The constant was divided into the bytes read (104 GB / 26.6 s = 3.9 GB/s), compared
# against the device's 13.5 GB/s, and the gap called "~1.5x of arena headroom". That
# arithmetic only means what it claims if the 26.6 s IS time the device spent moving
# bytes. It was never checked. The prefetch-touch result then showed that eager
# synchronous fetching HURTS at four lanes, which is hard to reconcile with a run that
# is throughput-bound: if the device were the constraint, fetching earlier and in
# parallel should not cost 2.5 s. So the assumption is now suspect on its own evidence.
#
# WHAT IS SAMPLED, every 200 ms for the whole run
#   /proc/diskstats   sectors read and io_ticks (ms the device was busy) for nvme0n1,
#                     nvme1n1 and md2. io_ticks is the honest answer to "was the disk
#                     actually working", independent of anything the process reports.
#   /proc/PID/io      rchar  = bytes the process asked for, page-cache hits included
#                     read_bytes = bytes actually pulled through the block layer
#                     The ratio of those two is the page-cache hit rate, and this box
#                     is holding 122 GB in buff/cache, so it may be most of the story.
#   /proc/PID/stat    minflt vs majflt: faults served from cache vs faults that waited
#                     on disk. A run dominated by minflt is paying kernel overhead,
#                     not device time, and an arena aimed at throughput misses it.
#                     Also utime+stime against wall x cores: with the device idle 75%
#                     of the run, the run is either compute-bound (CPU pegged) or
#                     latency-bound (neither busy). Those call for opposite work, so
#                     the question has to be settled rather than assumed.
#
# The time series matters as much as the totals: it shows whether I/O is one burst at
# startup or spread across the whole run, which decides whether prefetching can overlap
# with anything at all.
#
# usage: where-does-io-time-go.sh LANES [TAG]
set -u

C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LANES=${1:?lanes}
TAG=${2:-run}
OUT=/tmp/where-io-$TAG-$LANES

cd "$C/pipeline" || exit 1

: > /tmp/where-in.txt
i=0; while [ $i -lt "$LANES" ]; do echo "1 1008 10484 318 15383 387" >> /tmp/where-in.txt; i=$((i+1)); done

echo "=== lanes=$LANES tag=$TAG ===" | tee "$OUT.log"
echo "--- memory before (not dropping caches: warm cache IS the regime this runs in) ---" | tee -a "$OUT.log"
free -m | tee -a "$OUT.log"
echo "--- diskstats before ---" | tee -a "$OUT.log"
grep -E ' (nvme0n1|nvme1n1|md2) ' /proc/diskstats | tee -a "$OUT.log"
echo "--- vmstat before ---" | tee -a "$OUT.log"
grep -E '^(pgfault|pgmajfault|pgpgin) ' /proc/vmstat | tee -a "$OUT.log"

: > "$OUT.series"
printf "t\tsec_nvme0\tsec_nvme1\tsec_md2\ttick_nvme0\ttick_nvme1\trchar\tread_bytes\tminflt\tmajflt\tutime\tstime\tthreads\n" >> "$OUT.series"

env OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./bin/pipeline-serve "$C" serve < /tmp/where-in.txt > "$OUT.out" 2>&1 &
PID=$!
T0=$(date +%s.%N)

while kill -0 "$PID" 2>/dev/null; do
    now=$(date +%s.%N)
    t=$(echo "$now - $T0" | bc)
    ds=$(grep -E ' (nvme0n1|nvme1n1|md2) ' /proc/diskstats)
    s0=$(echo "$ds" | awk '$3=="nvme0n1"{print $6}')
    s1=$(echo "$ds" | awk '$3=="nvme1n1"{print $6}')
    sm=$(echo "$ds" | awk '$3=="md2"{print $6}')
    k0=$(echo "$ds" | awk '$3=="nvme0n1"{print $13}')
    k1=$(echo "$ds" | awk '$3=="nvme1n1"{print $13}')
    rc=$(awk '/^rchar/{print $2}' /proc/$PID/io 2>/dev/null)
    rb=$(awk '/^read_bytes/{print $2}' /proc/$PID/io 2>/dev/null)
    mf=$(awk '{print $10}' /proc/$PID/stat 2>/dev/null)
    Mf=$(awk '{print $12}' /proc/$PID/stat 2>/dev/null)
    ut=$(awk '{print $14}' /proc/$PID/stat 2>/dev/null)
    st=$(awk '{print $15}' /proc/$PID/stat 2>/dev/null)
    th=$(awk '{print $20}' /proc/$PID/stat 2>/dev/null)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$t" "${s0:-}" "${s1:-}" "${sm:-}" "${k0:-}" "${k1:-}" "${rc:-}" "${rb:-}" "${mf:-}" "${Mf:-}" "${ut:-}" "${st:-}" "${th:-}" >> "$OUT.series"
    sleep 0.2
done
wait "$PID"; RC=$?
T1=$(date +%s.%N)

echo "--- diskstats after ---" | tee -a "$OUT.log"
grep -E ' (nvme0n1|nvme1n1|md2) ' /proc/diskstats | tee -a "$OUT.log"
echo "--- vmstat after ---" | tee -a "$OUT.log"
grep -E '^(pgfault|pgmajfault|pgpgin) ' /proc/vmstat | tee -a "$OUT.log"
echo "--- memory after ---" | tee -a "$OUT.log"
free -m | tee -a "$OUT.log"
echo "--- wall $(echo "$T1 - $T0" | bc) ---" | tee -a "$OUT.log"
echo "--- request lines ---" | tee -a "$OUT.log"
grep -E '^REQUEST' "$OUT.out" | tee -a "$OUT.log"

# A run that answers nothing is a failed run, not a measurement. Say so before the
# summary, loudly, because the numbers below will still look perfectly plausible.
NREQ=$(grep -c '^REQUEST' "$OUT.out" 2>/dev/null || echo 0)
if [ "$RC" -ne 0 ] || [ "$NREQ" -lt "$LANES" ]; then
    echo "" | tee -a "$OUT.log"
    echo "*** FAILED RUN: exit=$RC, $NREQ of $LANES REQUEST lines. DO NOT READ THE" | tee -a "$OUT.log"
    echo "*** NUMBERS BELOW AS A MEASUREMENT. Last lines of output:" | tee -a "$OUT.log"
    tail -5 "$OUT.out" | sed 's/^/*** /' | tee -a "$OUT.log"
    echo "" | tee -a "$OUT.log"
fi

python3 - "$OUT.series" "$OUT.log" <<'PYEOF'
import os
import sys
rows = []
for line in open(sys.argv[1]):
    p = line.rstrip("\n").split("\t")
    if p[0] == "t" or len(p) < 13 or not p[1] or not p[7]:
        continue
    try:
        rows.append([float(p[0])] + [int(x) for x in p[1:13]])
    except ValueError:
        continue
if len(rows) < 2:
    print("not enough samples")
    sys.exit()
a, b = rows[0], rows[-1]
span = b[0] - a[0]
SEC = 512
dev_sectors = (b[1] - a[1]) + (b[2] - a[2])
md_sectors = b[3] - a[3]
busy_ms = max(b[4] - a[4], b[5] - a[5])
rchar = b[6] - a[6]
rbytes = b[7] - a[7]
minflt = b[8] - a[8]
majflt = b[9] - a[9]
HZ = 100.0
cpu_s = ((b[10] - a[10]) + (b[11] - a[11])) / HZ
user_s = (b[10] - a[10]) / HZ
sys_s = (b[11] - a[11]) / HZ
try:
    ncpu = os.cpu_count() or 1
except Exception:
    ncpu = 1
G = float(1 << 30)

out = []
out.append("")
out.append("sampled span            %8.2f s  (%d samples, first at %.2f s)" % (span, len(rows), a[0]))
out.append("device sectors read     %8.2f GB  (nvme0+nvme1)" % (dev_sectors * SEC / G))
out.append("md2 sectors read        %8.2f GB" % (md_sectors * SEC / G))
out.append("process read_bytes      %8.2f GB  (through the block layer)" % (rbytes / G))
out.append("process rchar           %8.2f GB  -- read()/pread() only; mmap faults never" % (rchar / G))
out.append("                                   appear here, so a near-zero rchar means the")
out.append("                                   model arrives entirely by page fault.")
out.append("device busy             %8.2f s  of %.2f s sampled = %.1f %% of the run"
           % (busy_ms / 1000.0, span, 100.0 * busy_ms / 1000.0 / span if span else 0))
if busy_ms:
    out.append("device read rate        %8.2f GB/s  while busy  <-- divide by BUSY, not wall;" % (dev_sectors * SEC / G / (busy_ms / 1000.0)))
    out.append("                                   dividing by wall is what produced the bogus")
    out.append("                                   3.9 GB/s and the bogus 1.5x arena headroom.")
if span:
    out.append("same bytes over wall    %8.2f GB/s  (the misleading number, kept for contrast)" % (dev_sectors * SEC / G / span))
out.append("minor faults            %10d  (no disk wait)" % minflt)
out.append("major faults            %10d  (had to wait on disk)" % majflt)
if minflt + majflt:
    out.append("major fault share       %8.3f %%" % (100.0 * majflt / (minflt + majflt)))
if majflt:
    out.append("bytes per major fault   %8.1f KB  (readahead reach, NOT bytes the code touched)"
               % (dev_sectors * SEC / 1024.0 / majflt))
out.append("  caution: fault-around maps up to 16 pages per minor fault, so fault counts do")
out.append("  NOT pin down how many bytes the code actually touched. Do not infer read")
out.append("  amplification from these two numbers alone.")
out.append("")
out.append("IS IT COMPUTE OR IS IT WAITING?")
out.append("cpu time used          %8.2f s  (user %.2f + sys %.2f) across %d cores"
           % (cpu_s, user_s, sys_s, ncpu))
out.append("threads at exit        %10d" % b[12])
if span:
    out.append("cpu busy per core      %8.1f %%  = cpu_time / (wall x cores)"
               % (100.0 * cpu_s / (span * ncpu)))
    out.append("cores busy on average  %8.2f  of %d" % (cpu_s / span, ncpu))
out.append("  the device was busy %.1f %% of the run. If the CPU share above is also low,"
           % (100.0 * busy_ms / 1000.0 / span if span else 0))
out.append("  the run is neither computing nor reading -- it is waiting, and faster bytes")
out.append("  or faster math would both miss the point.")
out.append("")
out.append("when the bytes arrive (cumulative, by elapsed second):")
mark = 0.0
for r in rows:
    if r[0] >= mark:
        out.append("  %6.1f s  read_bytes %8.2f GB   device busy %6.2f s   cpu %7.1f s   majflt %9d"
                   % (r[0], (r[7] - a[7]) / G, max(r[4] - a[4], r[5] - a[5]) / 1000.0,
                      ((r[10] - a[10]) + (r[11] - a[11])) / HZ, r[9] - a[9]))
        mark += 5.0
text = "\n".join(out)
print(text)
open(sys.argv[2], "a").write(text + "\n")
PYEOF

echo "done -> $OUT.log  $OUT.series"
