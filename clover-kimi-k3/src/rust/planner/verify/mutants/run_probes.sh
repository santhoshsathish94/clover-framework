#!/bin/bash
# Probe runs of the instrumented M06p / M08p copies: P1,P2,P4,P5,P6 only (P3 omitted: its raw logs are GBs and the disk is shared), counts of the events they appended.
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
for n in M06p M08p; do
  rm -f out/mut/probe_events.txt
  rm -rf out/mut/probe-$n
  so=$PWD/work/mut/$n/target/release/libplanner.so
  echo "$(date -u +%T) $n start md5=$(md5sum "$so" | cut -c1-8)" >> out/mut/probe_progress.txt
  RUST=$so JOBS=40 ./run.sh mut/probe-$n --scn P1,P2,P4,P5,P6 --keep-going > out/mut/probe-$n.stdout 2>&1
  echo "$(date -u +%T) $n done rc=$?" >> out/mut/probe_progress.txt
  {
    echo "== $n events (tag count, units = distinct pids)"
    awk '{c[$1" "$2]++; p[$1" "$2" "$3]=1} END {for (k in c) print k, c[k]}' out/mut/probe_events.txt | sort
    echo "distinct pids per tag:"
    awk '{p[$1" "$2" "$3]=1} END {for (k in p) {split(k,a," "); t[a[1]" "a[2]]++} for (k in t) print k, t[k]}' out/mut/probe_events.txt | sort
    echo "total event lines: $(wc -l < out/mut/probe_events.txt)"
  } > out/mut/probe-$n.events.txt
  cp out/mut/probe_events.txt out/mut/probe-$n.rawevents.txt
done
echo "$(date -u +%T) PROBES DONE" >> out/mut/probe_progress.txt
