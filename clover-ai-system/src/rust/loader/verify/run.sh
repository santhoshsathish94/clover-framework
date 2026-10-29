#!/bin/bash
# The whole matrix: ./run.sh [N] [outdir]   N = seeds of the cheap scenarios (default 30). At most 20 units at a time, all under nice.
cd "$(dirname "$0")"
N=${1:-30}; OUT=${2:-out}
mkdir -p "$OUT"; rm -f "$OUT/units.txt"
{
  for s in $(seq 1 $N); do echo "barrier $s"; echo "wait $s"; echo "stress $s"; echo "race $s"; echo "init2 $s"; done
  for s in $(seq 1 $((5 * N))); do echo "fifo $s"; done
  for s in $(seq 1 $((3 * N))); do echo "open $s"; done
  for s in $(seq 1 $((N / 2 > 4 ? N / 2 : 4))); do echo "shm $s"; done
  for s in $(seq 1 $((N / 5 > 4 ? N / 5 : 4))); do echo "real $s"; echo "fatal $s"; echo "natural $s"; done
  # driver v2 scenarios, after the old ones (the old units keep their place and seeds)
  for s in $(seq 1 $N); do echo "book $s"; echo "hotwait $s"; done
  for s in $(seq 1 $((N / 2 > 4 ? N / 2 : 4))); do echo "race2 $s"; done
  for s in $(seq 1 $((N / 5 > 4 ? N / 5 : 4))); do echo "big $s"; echo "fd0 $s"; echo "stack $s"; echo "guard $s"; done
} > "$OUT/jobs.txt"
start=$(date +%s)
xargs -P 20 -L 1 sh -c './unit.sh $0 $1 '"$OUT" < "$OUT/jobs.txt"
echo "units: $(wc -l < $OUT/units.txt) of $(wc -l < $OUT/jobs.txt), wall $(( $(date +%s) - start )) s"
./summarize.sh "$OUT"
