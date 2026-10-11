#!/usr/bin/env bash
# Runs every layer on its own, inside a memory cgroup, over the expert sequence real
# traffic produced. The host has 124 GB so nothing faults unless the cap forces it;
# major faults here are what a pod with no local disk would fetch over the network.
set -uo pipefail
LIMIT=${1:-16G}
LAYERS=${2:-"1 12 24 36 46 58 70 82 92"}
CG=/sys/fs/cgroup/podtest
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
RESULT=/tmp/walk-$LIMIT.log
: > "$RESULT"; rm -f /tmp/walk.flag

command -v gcc >/dev/null || { echo "no gcc" >> "$RESULT"; exit 1; }
gcc -std=gnu11 -O2 -march=native -w -fopenmp -ffp-contract=off -fno-fast-math -D_GNU_SOURCE \
    -I"$CODE/common" "$CODE/pipeline/expert-walk.c" -lm -o /tmp/expert-walk || {
    echo "build failed" >> "$RESULT"; exit 1; }

# One id file per layer: the experts that layer touched, in the order it touched them.
python3 - <<'EOF'
import json, collections
seq = collections.defaultdict(list)
for line in open('/tmp/routes64.jsonl'):
    if not line.startswith('ROUTE '): continue
    r = json.loads(line[6:])
    seq[r['layer']].extend(r['experts'])
for layer, ids in seq.items():
    with open('/tmp/ids-%d.txt' % layer, 'w') as handle:
        handle.write('\n'.join(str(i) for i in ids))
EOF

grep -q memory /sys/fs/cgroup/cgroup.subtree_control || echo "+memory" > /sys/fs/cgroup/cgroup.subtree_control
for layer in $LAYERS; do
    ids=/tmp/ids-$layer.txt
    [ -s "$ids" ] || { echo "layer=$layer no recorded routes" >> "$RESULT"; continue; }
    rmdir "$CG" 2>/dev/null; mkdir -p "$CG"
    echo "$LIMIT" > "$CG/memory.max"
    echo 0        > "$CG/memory.swap.max"
    echo "400000 100000" > "$CG/cpu.max" 2>/dev/null
    sync; echo 3 > /proc/sys/vm/drop_caches; sleep 1
    ( echo $BASHPID > "$CG/cgroup.procs"
      exec env OMP_NUM_THREADS=4 /tmp/expert-walk \
        "$CODE/tansformers/transformer-$layer/bin/dataset/root-$layer" "$layer" "$ids"
    ) >> "$RESULT" 2>&1
    peak=$(cat "$CG/memory.peak" 2>/dev/null || echo 0)
    printf '          peak_cgroup=%.2f GiB  oom=%s\n' \
        "$(echo "$peak" | awk '{print $1/1073741824}')" \
        "$(awk '/oom_kill /{print $2}' "$CG/memory.events" 2>/dev/null)" >> "$RESULT"
    rmdir "$CG" 2>/dev/null
done
echo DONE > /tmp/walk.flag
