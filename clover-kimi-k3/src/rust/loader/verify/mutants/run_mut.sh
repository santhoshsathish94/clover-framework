#!/bin/bash
# One full matrix (the counts of verify/run.sh N, default 200) per mutant, one mutant at a time, all under nice, at most 20 units at a time.
# usage: [N=200] [TAG=r1] [SODIR=...] run_mut.sh NAME...      output: verify/out/mut/<NAME>.<TAG>/ (units.log, summary.txt, *.fail.txt of FAIL units), progress in verify/out/mut/progress.txt
# After each mutant: raw leftovers of the driver in /tmp and /dev/shm that this run created are removed; stop if free space on / is below 40 GB.
cd "$(dirname "$0")/.."
N=${N:-200}; TAG=${TAG:-r1}; SODIR=${SODIR:-$PWD/work/so}
mkdir -p out/mut
for name in "$@"; do
    so=$SODIR/$name.so
    [ -f "$so" ] || { echo "$(date -u +%T) $name NO SO" >> out/mut/progress.txt; continue; }
    free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $name STOP free=${free}G" >> out/mut/progress.txt; exit 3; fi
    out=out/mut/$name.$TAG
    rm -rf "$out"; mkdir -p "$out"
    touch "$out/.start"
    {
      for s in $(seq 1 $N); do echo "barrier $s"; echo "wait $s"; echo "stress $s"; echo "race $s"; echo "init2 $s"; done
      for s in $(seq 1 $((5 * N))); do echo "fifo $s"; done
      for s in $(seq 1 $((3 * N))); do echo "open $s"; done
      for s in $(seq 1 $((N / 2 > 4 ? N / 2 : 4))); do echo "shm $s"; done
      for s in $(seq 1 $((N / 5 > 4 ? N / 5 : 4))); do echo "real $s"; echo "fatal $s"; echo "natural $s"; done
    } > "$out/jobs.txt"
    echo "$(date -u +%T) $name.$TAG start md5=$(md5sum "$so" | cut -c1-8) free=${free}G jobs=$(wc -l < $out/jobs.txt)" >> out/mut/progress.txt
    start=$(date +%s)
    xargs -P 20 -L 1 sh -c "mutants/mut_unit.sh \$0 \$1 $PWD/$out $so" < "$out/jobs.txt"
    wall=$(( $(date +%s) - start ))
    U=$out/units.log
    {
      echo "mutant $name tag $TAG md5 $(md5sum "$so" | cut -d' ' -f1) jobs $(wc -l < $out/jobs.txt) unit_lines $(grep -c '^UNIT' $U) maxfail ${MAXFAIL:-none} wall ${wall}s"
      echo "PASS $(grep -c '^UNIT [a-z0-9]* [0-9]* PASS ' $U) FAIL $(grep -c '^UNIT [a-z0-9]* [0-9]* FAIL ' $U) REFBAD $(grep -c '^UNIT [a-z0-9]* [0-9]* REFBAD ' $U)"
      for s in fifo barrier wait stress race init2 open shm natural real fatal; do
        echo "SCENARIO $s units=$(grep -c "^UNIT $s " $U) fail=$(grep -c "^UNIT $s [0-9]* FAIL " $U)"
      done
    } > "$out/summary.txt"
    echo "$(date -u +%T) $name.$TAG done $(sed -n 2p $out/summary.txt) wall=${wall}s" >> out/mut/progress.txt
    for d in /tmp/ldloader-verify.* /dev/shm/ldloader-verify.* /tmp/ldsmoke.*; do
        [ -e "$d" ] || continue
        [ "$d" -nt "$out/.start" ] || continue
        pid=${d##*.}
        kill -0 "$pid" 2>/dev/null && continue
        rm -rf "$d"
    done
    rm -f "$out/.start"
done
echo "$(date -u +%T) ALL DONE ($*)" >> out/mut/progress.txt
