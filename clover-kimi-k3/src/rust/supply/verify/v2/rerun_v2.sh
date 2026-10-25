#!/bin/bash
# Repeats (default 2) the full v2 driver run for the given mutants, one run at a time, 3 GB address-space cap per process, to see whether the number of failing units is stable.
# usage: rerun_v2.sh NAME...   -> v2/logs/reruns-v2.txt ; out/mut/rerun<r>-<name>/ (units.log, summary.txt, *.res, *.diff kept; raw logs deleted). Same disk rules as run_all.sh.
cd "$(dirname "$0")/.."
ulimit -v "${VLIMIT_KB:-3145728}"
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p out/mut work/data v2/logs
for r in $(seq 1 "${REPEATS:-2}"); do
  for n in "$@"; do
    so=$PWD/work/mut/$n/target/release/libsupply.so
    free=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 40 ]; then echo "STOP free=${free}G" >> v2/logs/reruns-v2.txt; exit 3; fi
    d=out/mut/rerun$r-$n
    rm -rf $d
    RUST=$so JOBS=${JOBS:-24} timeout -k 10 1500 ./run.sh mut/rerun$r-$n --keep-going --data "$PWD/work/data" > $d.stdout 2>&1
    find $d \( -name 'u[0-9]*.log' -o -name 'u[0-9]*.log.gz' -o -name 'u[0-9]*.trace' -o -name 'u[0-9]*.try1' -o -name 'u[0-9]*.il' -o -name 'u[0-9]*.cov' \) -delete
    echo "$n rerun$r units=$(grep -c '^UNIT' $d/units.log) fail=$(grep -c '^UNIT .* FAIL ' $d/units.log)" >> v2/logs/reruns-v2.txt
  done
done
echo "RERUNS DONE out/mut $(du -xsh out/mut | cut -f1) df: $(df -h / | tail -1 | awk '{print $4" free "$5" used"}')" >> v2/logs/reruns-v2.txt
