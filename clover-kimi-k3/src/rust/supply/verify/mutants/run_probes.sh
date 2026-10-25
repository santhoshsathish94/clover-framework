#!/bin/bash
# One full driver run per probe build (one at a time, nice 10). SUPPLY_PROBE=<events file> makes the probe lines append "<pid> <tag>" there.
# Afterwards the raw per-unit logs are deleted (as in run_all.sh); the events file and a tag summary are kept in mutants/logs/.
# usage: run_probes.sh NAME...   (output: mutants/logs/probe-<name>.events.txt, mutants/logs/probes.txt)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p out/mut work/data mutants/logs
for p in "$@"; do
  so=$PWD/work/mut/$p/target/release/libsupply.so
  [ -f "$so" ] || { echo "$p NO SO" >> mutants/logs/probes.txt; continue; }
  free=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$p STOP free=${free}G" >> mutants/logs/probes.txt; exit 3; fi
  ev=$PWD/mutants/logs/probe-$p.events.txt
  rm -f "$ev"; rm -rf out/mut/$p
  SUPPLY_PROBE=$ev RUST=$so JOBS=${JOBS:-24} timeout -k 10 1500 ./run.sh mut/$p --keep-going --data "$PWD/work/data" > out/mut/$p.stdout 2>&1
  rc=$?
  find out/mut/$p \( -name 'u[0-9]*.log' -o -name 'u[0-9]*.log.gz' -o -name 'u[0-9]*.trace' -o -name 'u[0-9]*.try1' -o -name 'u[0-9]*.il' -o -name 'u[0-9]*.cov' \) -delete
  nu=$(grep -c '^UNIT' out/mut/$p/units.log); nf=$(grep -c '^UNIT .* FAIL ' out/mut/$p/units.log)
  ne=0; npid=0; [ -f "$ev" ] && { ne=$(wc -l < "$ev"); npid=$(awk '{print $1}' "$ev" | sort -u | wc -l); }
  echo "$p rc=$rc units=$nu fail=$nf events=$ne distinct_processes=$npid events_file_bytes=$( [ -f "$ev" ] && stat -c %s "$ev" || echo 0)" >> mutants/logs/probes.txt
  [ -f "$ev" ] && awk '{print $2}' "$ev" | sort | uniq -c | sort -rn | head -12 | sed "s/^/    $p tag: /" >> mutants/logs/probes.txt
done
echo "PROBES DONE out/mut $(du -xsh out/mut | cut -f1) df: $(df -h / | tail -1 | awk '{print $4" free"}')" >> mutants/logs/probes.txt
