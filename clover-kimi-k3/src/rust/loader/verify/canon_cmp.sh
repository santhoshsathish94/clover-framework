#!/bin/bash
# Stability of the canonical output: for the old scenarios and seeds 1..N, every "C " line of the v1 driver (v2work/ld_driver_v1, the driver as the mutation controls found it) must be present, in
# the same order, in the output of the current driver; the new driver may only ADD lines. Both libraries (ref, rust). usage: ./canon_cmp.sh [N] [outdir]   (at most 12 processes at a time, under nice)
cd "$(dirname "$0")"
N=${1:-40}; OUT=${2:-v2work/canon}
export V1=${V1:-$PWD/v2work/ld_driver_v1}; export OUT
mkdir -p "$OUT"; rm -f "$OUT"/*
one() {
    scn=$1; seed=$2
    for lab in ref rust; do
        lib=../ref/libloader_ref.so; [ $lab = rust ] && lib=../target/release/libloader.so
        nice -n 10 timeout 120 "$V1" $lab "$lib" $scn $seed 2>/dev/null | grep '^C ' > "$OUT/$scn.$seed.$lab.v1"
        nice -n 10 timeout 120 ./ld_driver $lab "$lib" $scn $seed 2>/dev/null | grep '^C ' > "$OUT/$scn.$seed.$lab.v2"
        removed=$(diff "$OUT/$scn.$seed.$lab.v1" "$OUT/$scn.$seed.$lab.v2" | grep -c '^<')
        added=$(diff "$OUT/$scn.$seed.$lab.v1" "$OUT/$scn.$seed.$lab.v2" | grep -c '^>')
        echo "CANON $scn $seed $lab v1_lines=$(wc -l < $OUT/$scn.$seed.$lab.v1) removed_or_changed=$removed added=$added" >> "$OUT/result.txt"
        [ "$removed" = 0 ] && rm -f "$OUT/$scn.$seed.$lab.v1" "$OUT/$scn.$seed.$lab.v2"
    done
}
export -f one
{
  for s in $(seq 1 $N); do for scn in barrier wait stress race init2; do echo "$scn $s"; done; done
  for s in $(seq 1 $((5 * N))); do echo "fifo $s"; done
  for s in $(seq 1 $((3 * N))); do echo "open $s"; done
  for s in $(seq 1 $((N / 2))); do echo "shm $s"; done
  for s in $(seq 1 $((N / 5))); do echo "real $s"; echo "fatal $s"; echo "natural $s"; done
} | xargs -P 12 -L 1 bash -c 'one $0 $1'
echo "pairs: $(wc -l < $OUT/result.txt)  with removed/changed lines: $(grep -vc 'removed_or_changed=0 ' $OUT/result.txt)  with added lines: $(grep -vc 'added=0$' $OUT/result.txt)"
echo "scenarios with added lines: $(grep -v 'added=0$' $OUT/result.txt | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
