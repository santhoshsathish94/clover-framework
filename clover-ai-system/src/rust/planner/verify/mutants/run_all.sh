#!/bin/bash
# One full differential-driver run per mutant (one run at a time), then a cleanup of raw logs over 20 MB (the .diff / .res / units.log stay).
# While a run is going, a reaper deletes the raw logs of FAILED P3 units (GBs each; the .diff keeps the first difference) so that 36 failing real-trace units cannot fill the disk.
# usage: run_all.sh NAME...   (output: out/mut/<name>/, out/mut/<name>.stdout, out/mut/progress.txt)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p out/mut
reaper() {
  while true; do
    for r in out/mut/$1/P3/u*.res; do
      [ -f "$r" ] || continue
      if [ "$(head -c4 "$r")" = FAIL ]; then rm -f "${r%.res}".ref.log "${r%.res}".rust.log; fi
    done
    sleep 15
  done
}
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libplanner.so
  [ -f "$so" ] || { echo "$(date -u +%T) $n NO SO" >> out/mut/progress.txt; continue; }
  free=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $n STOP free=${free}G" >> out/mut/progress.txt; exit 3; fi
  echo "$(date -u +%T) $n start md5=$(md5sum "$so" | cut -c1-8) free=${free}G" >> out/mut/progress.txt
  rm -rf out/mut/$n
  reaper $n & rp=$!
  RUST=$so JOBS=40 ./run.sh mut/$n --keep-going > out/mut/$n.stdout 2>&1
  rc=$?
  kill $rp 2>/dev/null
  echo "$(date -u +%T) $n done rc=$rc" >> out/mut/progress.txt
  find out/mut/$n -name '*.log' -size +20M -printf '%s %p\n' > out/mut/$n.bigdel.txt
  find out/mut/$n -name '*.log' -size +20M -delete
  find out/mut/$n -name '*.trace' -size +20M -delete
done
echo "$(date -u +%T) ALL DONE" >> out/mut/progress.txt
