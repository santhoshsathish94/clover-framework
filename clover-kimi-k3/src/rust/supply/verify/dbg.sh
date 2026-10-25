#!/bin/bash
# usage: dbg.sh SCN SEED...   re-runs single units with their logs kept (gzipped) in out/dbg and prints the config line and the last lines of the reference log of each.
cd "$(dirname "$0")"
R=$HOME/rustmig
S=$1; shift
for seed in "$@"; do
  nice -n 10 ./sy_driver unit --scn "$S" --seed "$seed" --ref "$R/supply/ref/libsupply_ref.so" --rust "$R/supply/target/release/libsupply.so" --sc "$R/slotcache/target/release/libslotcache.so" --out out/dbg --data out/dbg/data --keep-raw 100000 | cut -c1-200
  f=out/dbg/$S/u$seed.ref.log.gz
  echo "--- $S $seed config:"; zcat "$f" | grep -m1 '^CFG' | cut -c1-520
  echo "--- last Z line:"; zcat "$f" | grep '^Z ' | tail -1 | cut -c1-300
  echo "--- last lines:"; zcat "$f" | tail -${TAILN:-8} | cut -c1-200
done
