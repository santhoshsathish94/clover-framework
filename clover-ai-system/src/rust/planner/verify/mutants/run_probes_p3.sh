#!/bin/bash
# P3 (real trace) seeds 0..11 only, with the instrumented M06p / M08p copies; counts of the events they appended. Raw and gzipped logs over 20 MB are deleted afterwards (shared disk).
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
for n in M06p M08p; do
  rm -f out/mut/probe_events.txt
  rm -rf out/mut/probe3-$n
  so=$PWD/work/mut/$n/target/release/libplanner.so
  echo "$(date -u +%T) P3 $n start free=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc 0-9)G" >> out/mut/probe_progress.txt
  RUST=$so JOBS=40 ./run.sh mut/probe3-$n --scn P3 --n P3=12 --keep-going > out/mut/probe3-$n.stdout 2>&1
  echo "$(date -u +%T) P3 $n done rc=$?" >> out/mut/probe_progress.txt
  {
    echo "== $n P3 seeds 0-11 events (tag count)"
    [ -f out/mut/probe_events.txt ] && awk '{c[$1" "$2]++} END {for (k in c) print k, c[k]}' out/mut/probe_events.txt | sort
    echo "total event lines: $( [ -f out/mut/probe_events.txt ] && wc -l < out/mut/probe_events.txt || echo 0)"
  } > out/mut/probe3-$n.events.txt
  find out/mut/probe3-$n \( -name '*.log' -o -name '*.log.gz' \) -size +20M -delete
done
echo "$(date -u +%T) P3 PROBES DONE" >> out/mut/probe_progress.txt
