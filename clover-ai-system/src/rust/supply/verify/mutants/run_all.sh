#!/bin/bash
# One full differential-driver run per mutant (one run at a time, nice 10), then a cleanup of the per-unit raw logs.
# Kept per run: units.log, summary.txt, *.res, *.diff, <name>.stdout. Deleted: u<seed>.*.log, *.log.gz, *.trace, *.try1, *.il, *.cov (all units, passing and failing).
# (The first version of this script used the pattern '*.log' and deleted units.log too; the whole pass was repeated with the narrower pattern.)
# The data files the drive reads come from are shared (verify/work/data: the same pure function of (scenario, seed) in every run) and deleted at the end by the caller.
# usage: run_all.sh NAME...   (output: out/mut/<name>/, out/mut/<name>.stdout, out/mut/progress.txt)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p out/mut work/data
k=0
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libsupply.so
  [ -f "$so" ] || { echo "$(date -u +%T) $n NO SO" >> out/mut/progress.txt; continue; }
  free=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $n STOP free=${free}G" >> out/mut/progress.txt; exit 3; fi
  echo "$(date -u +%T) $n start md5=$(md5sum "$so" | cut -c1-32) free=${free}G" >> out/mut/progress.txt
  rm -rf out/mut/$n
  RUST=$so JOBS=${JOBS:-24} timeout -k 10 1500 ./run.sh mut/$n --keep-going --data "$PWD/work/data" > out/mut/$n.stdout 2>&1
  rc=$?
  echo "$(date -u +%T) $n done rc=$rc" >> out/mut/progress.txt
  find out/mut/$n \( -name 'u[0-9]*.log' -o -name 'u[0-9]*.log.gz' -o -name 'u[0-9]*.trace' -o -name 'u[0-9]*.try1' -o -name 'u[0-9]*.il' -o -name 'u[0-9]*.cov' \) -delete
  k=$((k+1))
  if [ $((k % 5)) -eq 0 ]; then
    echo "$(date -u +%T) after $k: out/mut $(du -xsh out/mut | cut -f1), work $(du -xsh work | cut -f1), df: $(df -h / | tail -1 | awk '{print $4" free "$5" used"}')" >> out/mut/progress.txt
  fi
done
echo "$(date -u +%T) ALL DONE out/mut $(du -xsh out/mut | cut -f1) work $(du -xsh work | cut -f1) df: $(df -h / | tail -1 | awk '{print $4" free "$5" used"}')" >> out/mut/progress.txt
