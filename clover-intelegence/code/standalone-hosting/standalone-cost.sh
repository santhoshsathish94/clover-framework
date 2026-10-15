#!/usr/bin/env bash
# Where does the standalone engine's 3.05 s per token go?
#
# WHY THIS EXISTS
# Two days of instruments were built and every one of them was pointed at the
# distributed pipeline. The number that matters -- 9.9 s prompt pass plus 3.05 s per
# output token, recorded in standalone-hosting/LATENCY-AND-SCALING.md -- belongs to the
# STANDALONE engine, which has never been measured here at all. So there is tooling and
# no evidence about the system being improved. This points the tooling at it.
#
# It reports, for one request:
#   per-token seconds        from STEP_JSON on stderr, so the prompt pass and each
#                            generated token are separated rather than averaged
#   device busy              io_ticks from /proc/diskstats -- the honest answer to
#                            "was the disk working", not bytes divided by wall
#   bytes actually read      sectors from the block layer, and bytes per token
#   user vs system CPU       from /proc/PID/stat
#   cache hits and misses    from CACHE_JSON, which the engine already prints
#
# The cache is the point. standalone has root_cache_open (92 layers x K3_EXPERT_CACHE
# experts x 16.7 MB) and the distributed side has none, which is the leading candidate
# for why distributed costs 5.68 s per token against standalone's 3.05. Before changing
# anything, find out how much of the 3.05 s is disk the cache is failing to absorb and
# how much is compute the cache can never touch.
#
# Sampling is at 1 s, not 0.2 s, and reads only /proc/PID/stat. A 0.2 s loop that walked
# /proc/PID/task/*/status perturbed a 17-thread run by 4.2 s earlier today. Measure
# lightly or measure something else.
set -u

S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
BIN=${BIN:-$S/bin/clover-one}
TOKENS=${TOKENS:-8}
CACHE=${CACHE:-24}
THREADS=${THREADS:-16}
BIND=${BIND:-close}
TAG=${TAG:-base}
OUT=/tmp/standalone-$TAG

cd "$S" || exit 1

echo "{\"input_ids\": [1008, 10484, 318, 15383, 387], \"max_new_tokens\": $TOKENS}" > "$OUT.in"

sectors() { grep -E ' (nvme0n1|nvme1n1) ' /proc/diskstats | awk '{s+=$6} END{print s+0}'; }
ticks()   { grep -E ' (nvme0n1|nvme1n1) ' /proc/diskstats | awk '{if($13>m)m=$13} END{print m+0}'; }

echo "=== standalone: tokens=$TOKENS cache=$CACHE threads=$THREADS bind=$BIND ==="
echo "binary: $BIN  ($(date -r "$BIN" '+%Y-%m-%d %H:%M'))"
free -m | sed -n 2p

s0=$(sectors); k0=$(ticks); t0=$(date +%s.%N)
env -i PATH="$PATH" \
    CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$S/bin/configs/generation.json" \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
    K3_EXPERT_CACHE="$CACHE" \
    OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND="$BIND" OMP_PLACES=cores \
    "$BIN" < "$OUT.in" > "$OUT.out" 2> "$OUT.err" &
PID=$!

ut=0; st=0; maxth=0; minf=0; majf=0; peakrss=0
while kill -0 "$PID" 2>/dev/null; do
    v=$(awk '{print $10" "$12" "$14" "$15" "$20}' "/proc/$PID/stat" 2>/dev/null)
    if [ -n "${v:-}" ]; then
        minf=$(echo "$v" | cut -d' ' -f1); majf=$(echo "$v" | cut -d' ' -f2)
        ut=$(echo "$v" | cut -d' ' -f3);   st=$(echo "$v" | cut -d' ' -f4)
        th=$(echo "$v" | cut -d' ' -f5); [ "${th:-0}" -gt "$maxth" ] && maxth=$th
    fi
    r=$(awk '/^VmRSS:/ {print $2}' "/proc/$PID/status" 2>/dev/null)
    [ -n "${r:-}" ] && [ "$r" -gt "$peakrss" ] && peakrss=$r
    sleep 1
done
wait "$PID"; RC=$?
t1=$(date +%s.%N); s1=$(sectors); k1=$(ticks)

if ! grep -q DONE_JSON "$OUT.out"; then
    echo "*** FAILED (exit $RC). last lines:"; tail -4 "$OUT.out" "$OUT.err"; exit 1
fi

# The answer itself, so a faster engine that says something different is caught.
echo "TOKENS_PRODUCED: $(grep -h '^TOKEN_JSON' "$OUT.out" | sed 's/.*"token"://; s/}//' | tr '\n' ' ')"
echo "peak RSS: $(echo "scale=2; $peakrss/1048576" | bc) GB   (cache at $CACHE should add $(echo "scale=1; 93*$CACHE*17547264/1073741824" | bc) GB if it allocates)"

python3 - "$OUT" "$t0" "$t1" "$s0" "$s1" "$k0" "$k1" "$ut" "$st" "$minf" "$majf" "$TOKENS" <<'PYEOF'
import json
import re
import sys

out, t0, t1, s0, s1, k0, k1, ut, st, minf, majf, tokens = sys.argv[1:]
wall = float(t1) - float(t0)
gb = (int(s1) - int(s0)) * 512 / float(1 << 30)
busy = (int(k1) - int(k0)) / 1000.0
user = int(ut) / 100.0
system = int(st) / 100.0
tokens = int(tokens)

steps = []
for line in open(out + ".err"):
    m = re.search(r'STEP_JSON (\{.*\})', line)
    if m:
        try:
            steps.append(json.loads(m.group(1)))
        except ValueError:
            pass

done = None
cache = None
for line in open(out + ".out"):
    if line.startswith("DONE_JSON "):
        done = json.loads(line[len("DONE_JSON "):])
    if line.startswith("CACHE_JSON "):
        cache = json.loads(line[len("CACHE_JSON "):])

print()
if done:
    print("answer: %s tokens in %.3f s (engine's own figure), stop=%s"
          % (done.get("output_tokens"), done.get("seconds", 0), done.get("stop_reason")))

if steps:
    print("\nper-position seconds from STEP_JSON:")
    for s in steps:
        print("  position %3s  %8.3f s" % (s.get("position"), s.get("seconds", 0)))
    gen = [s.get("seconds", 0) for s in steps[1:]] if len(steps) > 1 else []
    if gen:
        gen_sorted = sorted(gen)
        print("\n  prompt pass          %8.3f s" % steps[0].get("seconds", 0))
        print("  generated tokens     n=%d  median %8.3f s  min %8.3f  max %8.3f"
              % (len(gen), gen_sorted[len(gen) // 2], gen_sorted[0], gen_sorted[-1]))

print("\nwall                   %8.2f s" % wall)
print("bytes read             %8.2f GB   = %.2f GB per generated token"
      % (gb, gb / max(tokens, 1)))
print("device busy            %8.2f s  = %.1f %% of wall" % (busy, 100 * busy / wall if wall else 0))
if busy:
    print("rate while busy        %8.2f GB/s   (ceiling on this box is 12.5)" % (gb / busy))
print("cpu                    %8.2f s  (user %.2f + sys %.2f) = %.2f cores"
      % (user + system, user, system, (user + system) / wall if wall else 0))
print("faults                 minor %d  major %d" % (int(minf), int(majf)))

if cache:
    print("\ncache, from the engine's own CACHE_JSON:")
    for key in ("result_hits", "result_misses", "result_stores", "result_evictions",
                "result_cache_bytes", "cross_layer_hits", "cross_layer_misses",
                "pipeline_reads", "pipeline_bytes", "pipeline_wait_seconds"):
        if key in cache:
            print("  %-24s %s" % (key, cache[key]))
    h, m = cache.get("result_hits", 0), cache.get("result_misses", 0)
    if h + m:
        print("  %-24s %.1f %%" % ("hit rate", 100.0 * h / (h + m)))
PYEOF
