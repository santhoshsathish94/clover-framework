#!/usr/bin/env bash
# Repeat the routing observation across prompts of different input lengths.
#
# Each run seeds a prompt and generates 64 tokens, feeding each one back, so a single
# run sweeps context length from the seed length up to seed+64. Between them the runs
# cover seeds from 5 to 24 tokens, which is the full range the repository's prompts.tsv
# provides -- there is no 2-token or 64-token prompt in it, and I am not inventing
# token ids.
#
# Routing rather than vectors: a full CLOVER_VECTORS capture is ~267 MB per position, so
# 70 positions would be 19 GB for one prompt. The selection per layer per position is a
# few hundred KB and is the thing that actually varies, since of ~120 stages per pod
# only four change the carried state and the expensive one is the MoE.
set -u

S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TSV=/opt/clover-k3/prompts.tsv
OUT=/tmp/sweep
WANT=${WANT:-64}
mkdir -p "$OUT"

# a spread across the available seed lengths rather than the first N rows
ROWS=$(awk -F'\t' '{print $2"\t"$1}' "$TSV" | sort -n | awk '!seen[$1]++ {print $2}' | head -12 | tr '\n' ' ')
echo "rows chosen for length spread: $ROWS"

for row in $ROWS; do
    line=$(awk -F'\t' -v r="$row" '$1==r {print}' "$TSV")
    [ -z "$line" ] && continue
    len=$(echo "$line" | cut -f2)
    ids=$(echo "$line" | cut -f3 | tr ',' ' ')
    text=$(echo "$line" | cut -f4 | cut -c1-44)
    dst="$OUT/route-r$row-len$len.jsonl"
    if [ -s "$dst" ]; then echo "skip row $row, already have it"; continue; fi

    printf "row %-3s len %-3s %s\n" "$row" "$len" "$text"
    start=$(date +%s)
    env -i PATH="$PATH" \
        CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$S/bin/configs/generation.json" \
        K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=24 \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
        /tmp/observe-route-any "$S/bin/dataset" "$WANT" $ids > "$dst" 2>"$dst.err"
    rc=$?
    n=$(grep -c '^ROUTE' "$dst" 2>/dev/null || echo 0)
    t=$(grep -c '^TOKEN' "$dst" 2>/dev/null || echo 0)
    printf "   exit %d, %s ROUTE lines, %s tokens emitted, %ds\n" \
        "$rc" "$n" "$t" "$(( $(date +%s) - start ))"
    [ "$rc" -ne 0 ] && tail -2 "$dst.err"
done
touch /tmp/sweep.done
echo "done"
