#!/bin/bash
# The original driver (v2/orig/sy_driver, md5 612e76a6...) against the v2 driver on the 11 original scenarios, seeds 1..N, same libraries: for every unit every kept log (reference and Rust arm) must be
# equal once the EXIT and PREAD lines that v2 adds at the end are removed. Output: one line per scenario and a final count; scratch in out/cmp-old, out/cmp-new, out/cmp-data (deleted at the end).
# usage: cmp_old_new.sh [N]
cd "$(dirname "$0")/.."
R=$HOME/rustmig; N=${1:-5}
REF=$R/supply/ref/libsupply_ref.so; RUST=$R/supply/target/release/libsupply.so; SC=$R/slotcache/target/release/libslotcache.so
rm -rf out/cmp-old out/cmp-new out/cmp-data
same=0; diffc=0; missing=0
for scn in Y1 Y2 Y3 Y4 Y5 Y6 Y7 Y8 Y9 YF T1; do
  s_same=0; s_diff=0
  for s in $(seq 1 "$N"); do
    nice -n 10 v2/orig/sy_driver unit --scn $scn --seed $s --ref "$REF" --rust "$RUST" --sc "$SC" --out out/cmp-old --data out/cmp-data --keep-raw 100000 > /dev/null
    nice -n 10 ./sy_driver        unit --scn $scn --seed $s --ref "$REF" --rust "$RUST" --sc "$SC" --out out/cmp-new --data out/cmp-data --keep-raw 100000 > /dev/null
    for f in out/cmp-old/$scn/u$s.*.log.gz; do
      b=$(basename "$f"); g=out/cmp-new/$scn/$b
      if [ ! -f "$g" ]; then missing=$((missing+1)); continue; fi
      if cmp -s <(zcat "$f") <(zcat "$g" | grep -v '^EXIT \|^PREAD '); then s_same=$((s_same+1)); else s_diff=$((s_diff+1)); echo "DIFF $scn $b"; fi
    done
  done
  echo "$scn seeds=1..$N logs equal=$s_same different=$s_diff"
  same=$((same+s_same)); diffc=$((diffc+s_diff))
done
echo "TOTAL logs equal=$same different=$diffc missing=$missing"
rm -rf out/cmp-old out/cmp-new out/cmp-data
