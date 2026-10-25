#!/bin/bash
# Driver v2 re-run of the mutation matrix: the same runner as mutants/run_mut.sh (one mutant at a time, at most 20 units at a time, all under nice, MAXFAIL stops a run after K failed units) with the
# units of the v2 scenarios added after the old ones, and with the audit trail kept: the units.log of every run is gzipped into out/mut2/units/, the driver outputs of failing units (*.fail.txt) are
# deleted unless the mutant is in KEEP (they are first-30-KB copies; the first failure line is in units.log anyway), everything else raw is deleted at once by mutants/mut_unit.sh.
# usage: [N=200] [TAG=v2a] [MAXFAIL=100] [KEEP="M05 M82"] [SODIR=...] mutants/run_mut2.sh NAME...        progress: verify/out/mut2/progress.txt
cd "$(dirname "$0")/.."
N=${N:-200}; TAG=${TAG:-v2a}; SODIR=${SODIR:-$PWD/work/so}; KEEP=${KEEP:-}
mkdir -p out/mut2/units
for name in "$@"; do
    so=$SODIR/$name.so
    [ -f "$so" ] || { echo "$(date -u +%T) $name NO SO" >> out/mut2/progress.txt; continue; }
    free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $name STOP free=${free}G" >> out/mut2/progress.txt; exit 3; fi
    out=out/mut2/$name.$TAG
    rm -rf "$out"; mkdir -p "$out"
    touch "$out/.start"
    {
      for s in $(seq 1 $N); do echo "barrier $s"; echo "wait $s"; echo "stress $s"; echo "race $s"; echo "init2 $s"; done
      for s in $(seq 1 $((5 * N))); do echo "fifo $s"; done
      for s in $(seq 1 $((3 * N))); do echo "open $s"; done
      for s in $(seq 1 $((N / 2 > 4 ? N / 2 : 4))); do echo "shm $s"; done
      for s in $(seq 1 $((N / 5 > 4 ? N / 5 : 4))); do echo "real $s"; echo "fatal $s"; echo "natural $s"; done
      for s in $(seq 1 $N); do echo "book $s"; echo "hotwait $s"; done
      for s in $(seq 1 $((N / 2 > 4 ? N / 2 : 4))); do echo "race2 $s"; done
      for s in $(seq 1 $((N / 5 > 4 ? N / 5 : 4))); do echo "big $s"; echo "fd0 $s"; echo "stack $s"; echo "guard $s"; done
    } > "$out/jobs.txt"
    echo "$(date -u +%T) $name.$TAG start md5=$(md5sum "$so" | cut -c1-8) free=${free}G jobs=$(wc -l < $out/jobs.txt)" >> out/mut2/progress.txt
    start=$(date +%s)
    xargs -P 20 -L 1 sh -c "mutants/mut_unit.sh \$0 \$1 $PWD/$out $so" < "$out/jobs.txt"
    wall=$(( $(date +%s) - start ))
    U=$out/units.log
    {
      echo "mutant $name tag $TAG md5 $(md5sum "$so" | cut -d' ' -f1) jobs $(wc -l < $out/jobs.txt) unit_lines $(grep -c '^UNIT' $U) maxfail ${MAXFAIL:-none} wall ${wall}s"
      echo "PASS $(grep -c '^UNIT [a-z0-9]* [0-9]* PASS ' $U) FAIL $(grep -c '^UNIT [a-z0-9]* [0-9]* FAIL ' $U) REFBAD $(grep -c '^UNIT [a-z0-9]* [0-9]* REFBAD ' $U)"
      for s in fifo barrier wait stress race init2 open shm natural real fatal book hotwait race2 big fd0 stack guard; do
        echo "SCENARIO $s units=$(grep -c "^UNIT $s " $U) fail=$(grep -c "^UNIT $s [0-9]* FAIL " $U)"
      done
    } > "$out/summary.txt"
    echo "$(date -u +%T) $name.$TAG done $(sed -n 2p $out/summary.txt) wall=${wall}s" >> out/mut2/progress.txt
    gzip -c "$U" > "out/mut2/units/$name.$TAG.units.log.gz"
    cp "$out/summary.txt" "out/mut2/units/$name.$TAG.summary.txt"
    case " $KEEP " in *" $name "*) mkdir -p out/mut2/fail; tar -czf "out/mut2/fail/$name.$TAG.fail.tgz" -C "$out" $(cd "$out"; ls *.fail.txt 2>/dev/null | head -20) 2>/dev/null ;; esac
    for d in /tmp/ldloader-verify.* /dev/shm/ldloader-verify.* /tmp/ldsmoke.*; do
        [ -e "$d" ] || continue
        [ "$d" -nt "$out/.start" ] || continue
        pid=${d##*.}
        kill -0 "$pid" 2>/dev/null && continue
        rm -rf "$d"
    done
    rm -rf "$out"
done
echo "$(date -u +%T) ALL DONE ($*)" >> out/mut2/progress.txt
