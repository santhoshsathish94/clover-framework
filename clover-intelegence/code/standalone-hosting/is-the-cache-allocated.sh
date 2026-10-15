#!/usr/bin/env bash
# Is the expert cache allocated at all?
#
# WHY THIS EXISTS
# K3_EXPERT_CACHE at 0, 24 and 48 produced byte-for-byte identical disk traffic
# (1647.83 / 1647.83 / 1647.73 GB over 64 tokens). A working cache cannot read exactly
# the same bytes as no cache, so it is not merely failing to hit -- it is not in the
# path. The gate in root_pull_begin is
#
#     unsigned held = (root_cache_store && root->layer < root_cache_layers &&
#                      count <= root_cache_capacity) ? root_cache_capacity : 0;
#
# and if root_cache_store is NULL, held is 0 and the cache disappears with no diagnostic.
# root_cache_hits is counted but never printed, so nothing would ever say so.
#
# root_cache_open allocates layers * capacity * ROOT_EXPERT_RAW of ANONYMOUS memory and
# pre-touches every huge page. At 92 layers that is 38.5 GB for capacity 24 and 77 GB
# for 48. Anonymous memory shows up in RSS, so peak RSS settles the question without
# touching the code: if the cache allocates, capacity 24 must sit ~38.5 GB above
# capacity 0.
#
# Peak, not final. RSS is sampled while the process runs because reading it after exit
# gives nothing.
set -u

S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TOKENS=${TOKENS:-4}

cd "$S" || exit 1
echo '{"input_ids": [1008, 10484, 318, 15383, 387], "max_new_tokens": '"$TOKENS"'}' > /tmp/rss-in.txt

probe() {
    local cache=$1
    env -i PATH="$PATH" \
        CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$S/bin/configs/generation.json" \
        K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 \
        K3_EXPERT_CACHE="$cache" \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
        "$S/bin/clover-one" < /tmp/rss-in.txt > "/tmp/rss-$cache.out" 2>&1 &
    local pid=$! peak=0 rss
    while kill -0 "$pid" 2>/dev/null; do
        rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -n "${rss:-}" ] && [ "$rss" -gt "$peak" ] && peak=$rss
        sleep 0.3
    done
    wait "$pid"
    local ok
    ok=$(grep -c DONE_JSON "/tmp/rss-$cache.out")
    printf "  K3_EXPERT_CACHE=%-3s peak RSS %8.2f GB   completed=%s\n" \
        "$cache" "$(echo "scale=2; $peak/1048576" | bc)" "$ok"
}

echo "expected extra anonymous memory if the cache allocates:"
echo "  capacity 24 -> 92 x 24 x 17547264 B = 38.5 GB"
echo "  capacity 48 -> 92 x 48 x 17547264 B = 77.1 GB"
echo
for c in 0 24 48; do probe "$c"; done
echo
echo "If the three peaks are within a few GB of each other, root_cache_open is failing"
echo "and the documented per-layer expert cache has never been running on this build."
touch /tmp/rss.done
