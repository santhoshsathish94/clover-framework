#!/bin/bash
# Driver v2, step 2: one full differential run per mutant (default counts H 600, HX 400, P 600, PX 800, PR 200, M 400 = 3,000 units, plus scenario A = 60 units of refused arguments), one run at
# a time, JOBS (default 8) units in parallel, nice -n 10, every process capped at 3 GB of virtual memory (ulimit -v). Mutants come from work/v2so/<name>.so (v2/build_all.sh); M00 = the unmodified crate.
# After each run: v2/analyze2.py (counts, first failure, logs of the first failing unit to v2/logs/<name>/), then the raw per-unit logs are deleted (units.log, summary.txt, .res stay).
# df / du every 5 runs; the loop stops if the free space on / drops below 40 GB.
# usage (box, ~/rustmig/poolinit/verify): v2/run_mut.sh NAME...     env SUB (output dir out/$SUB, default v2mut), MULT (default 1), JOBS, VMEM_KB, NOA=1 (no scenario A)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
ulimit -v ${VMEM_KB:-3000000}
SUB=${SUB:-v2mut}; OUTD=out/$SUB; mkdir -p $OUTD
MULT=${MULT:-1}
N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))"
SCN="H,HX,P,PX,PR,M"
if [ -z "$NOA" ]; then N="$N,A=60"; SCN="$SCN,A"; fi
export SUB MULT NOA
k=0
for n in "$@"; do
  so=$PWD/work/v2so/$n.so
  [ -f "$so" ] || { echo "$(date -u +%T) $n NO SO" >> $OUTD/progress.txt; continue; }
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $n STOP free=${free}G" >> $OUTD/progress.txt; exit 3; fi
  md5=$(md5sum "$so" | cut -c1-32)
  echo "$(date -u +%T) $n start md5=$md5 free=${free}G mult=$MULT" >> $OUTD/progress.txt
  rm -rf $OUTD/$n
  RUST=$so JOBS=${JOBS:-8} timeout ${TMO:-1500} ./run.sh $SUB/$n --keep-raw 0 --keep-going --scn $SCN --n "$N" > $OUTD/$n.stdout 2>&1
  rc=$?
  if [ $rc -eq 124 ]; then pkill -u "$USER" -f "pi_driver .*out/$SUB/$n( |/)"; fi
  echo "$(date -u +%T) $n done rc=$rc recount: UNIT=$(grep -c '^UNIT' $OUTD/$n/units.log) PASS=$(grep -c '^UNIT .* PASS ' $OUTD/$n/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' $OUTD/$n/units.log) distinct=$(grep '^UNIT' $OUTD/$n/units.log | awk '{print $2,$3}' | sort -u | wc -l)" >> $OUTD/progress.txt
  python3 v2/analyze2.py $n $md5 >> $OUTD/analysis.log 2>&1
  find $OUTD/$n -type f \( -name '*.log' -o -name '*.log.gz' -o -name '*.trace' -o -name '*.err' \) ! -name units.log -delete
  [ "$MULT" -gt 1 ] && find $OUTD/$n -type f -name '*.res' -delete   # 60,060 .res files are 240 MB of blocks per run; units.log + summary.txt keep the result (the 20x run of the first campaign kept them: 1.9 GB)
  k=$((k+1))
  if [ $((k % 5)) -eq 0 ]; then echo "$(date -u +%T) after $k: $(du -xsh $OUTD work/v2so 2>/dev/null | tr '\n' ' ') $(df -h / | tail -1)" >> $OUTD/progress.txt; fi
done
echo "$(date -u +%T) ALL DONE ($*) $(du -xsh $OUTD 2>/dev/null | tr '\n' ' ') $(df -h / | tail -1)" >> $OUTD/progress.txt
