#!/bin/sh
# Step 1 of the time accounting: capture everything a run can tell us.
#
# A decode step measured 3.763202 s while its fourteen operator buckets summed
# to 2.519838 s. A third of the token is unattributed, and no proposal is worth
# making until that is explained. Reading files costs CPU too - syscalls, page
# faults, and copies out of page cache - and none of that lands in an operator.
#
# Captures per step: STEP_JSON and OPSTAT_JSON. Captures for the process:
# user and system CPU, major and minor faults, and the kernel's own view of
# bytes read, so operator time, I/O time and stall time can be put in one table
# that has to add up to the wall clock.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TSV=/opt/clover-k3/prompts.tsv
OUT=${OUT:-/tmp/timing}
TOK=${TOK:-6}
ROW=${ROW:-1}
CACHE=${CACHE:-24}
# the deployed binary has no expert cache compiled in, so the build under test
# must be selectable without replacing what is deployed
BIN=${BIN:-$S/bin/clover-one}

mkdir -p "$OUT"
ids=$(awk -F'\t' -v r="$ROW" '$1==r {print $3}' "$TSV")
len=$(awk -F'\t' -v r="$ROW" '$1==r {print $2}' "$TSV")
[ -z "$ids" ] && { echo "no row $ROW"; exit 1; }
echo "binary $BIN"
echo "row $ROW, $len input tokens, generating $TOK, expert cache $CACHE"
echo "cache compiled in: $(strings "$BIN" | grep -c K3_EXPERT_CACHE)"

echo "{\"input_ids\":[$ids],\"max_new_tokens\":$TOK}" > "$OUT/req.json"

# drop caches is not available without root privileges we should not assume;
# instead record whether pages were already warm via the fault counters.
/usr/bin/time -v -o "$OUT/time.txt" \
  env -i PATH="$PATH" \
    CLOVER_DATASET="$S/bin/dataset" CLOVER_CONFIG="$S/bin/configs/generation.json" \
    K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE="$CACHE" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" "$S/bin/dataset" \
    < "$OUT/req.json" > "$OUT/out.txt" 2> "$OUT/err.txt"
rc=$?

grep STEP_JSON  "$OUT/err.txt" > "$OUT/steps.jsonl"  2>/dev/null
grep OPSTAT_JSON "$OUT/err.txt" > "$OUT/opstat.jsonl" 2>/dev/null

echo "exit $rc, $(wc -l < "$OUT/steps.jsonl") steps, $(wc -l < "$OUT/opstat.jsonl") opstat lines"
echo
echo "=== process totals ==="
grep -E 'wall clock|User time|System time|Maximum resident|Major|Minor|File system' "$OUT/time.txt"
echo
echo "=== per-process io ==="
cat "$OUT/err.txt" | grep -E '^trunk ->|^operator values' | head -3
tail -2 "$OUT/err.txt" | cut -c1-200
