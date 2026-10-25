#!/bin/bash
# One full differential-driver run (default counts H 600, HX 400, P 600, PX 800, PR 200, M 400 = 3000 units) per mutant, one run at a time, nice -n 10.
# After each run: analyze.py (counts, first failure, copies the first failing unit's logs to mutants/logs/), then the raw per-unit logs (*.log except units.log, *.log.gz, *.trace, *.err) are deleted;
# units.log, summary.txt, *.res, *.diff and the .stdout stay. df / du every 5 mutants; stops if free space on / drops below 40 GB.
# usage (box, ~/rustmig/poolinit/verify): mutants/run_all.sh NAME...     env MULT scales the counts (default 1), JOBS (default 8), KEEP raw logs of the first N seeds (default 0), VMEM_KB
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
# Virtual-memory cap for every process of a run (default 3 GB): a mutant with an absurd slot count makes the driver's own event recorder grow without bound
# (first M69 / M70 run, no cap: children above 14 GB each before they were killed by hand). Hitting the cap kills the child -> the unit FAILs (caught as a crash).
ulimit -v ${VMEM_KB:-3000000}
# crashing mutants: no crash-handler (apport) per crash, see nodump.c (gcc -shared -fPIC -o mutants/nodump.so mutants/nodump.c)
[ -f "$PWD/mutants/nodump.so" ] && export LD_PRELOAD=$PWD/mutants/nodump.so
mkdir -p out/mut
MULT=${MULT:-1}
N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))"
k=0
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libpoolinit.so
  [ -f "$so" ] || { echo "$(date -u +%T) $n NO SO" >> out/mut/progress.txt; continue; }
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $n STOP free=${free}G" >> out/mut/progress.txt; exit 3; fi
  md5=$(md5sum "$so" | cut -c1-32)
  echo "$(date -u +%T) $n start md5=$md5 free=${free}G mult=$MULT" >> out/mut/progress.txt
  rm -rf out/mut/$n
  RUST=$so JOBS=${JOBS:-8} timeout 1500 ./run.sh mut/$n --keep-raw ${KEEP:-0} --keep-going --n "$N" > out/mut/$n.stdout 2>&1
  rc=$?
  if [ $rc -eq 124 ]; then pkill -u "$USER" -f "pi_driver .*out/mut/$n( |/)"; fi
  echo "$(date -u +%T) $n done rc=$rc" >> out/mut/progress.txt
  echo "$n recount: UNIT=$(grep -c '^UNIT' out/mut/$n/units.log) PASS=$(grep -c '^UNIT .* PASS ' out/mut/$n/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' out/mut/$n/units.log)" >> out/mut/progress.txt
  python3 mutants/analyze.py $n $md5 >> out/mut/analysis.log 2>&1
  find out/mut/$n -type f \( -name '*.log' -o -name '*.log.gz' -o -name '*.trace' -o -name '*.err' \) ! -name units.log -delete
  k=$((k+1))
  if [ $((k % 5)) -eq 0 ]; then echo "$(date -u +%T) after $k: $(du -xsh out/mut work/mut 2>/dev/null | tr '\n' ' ') $(df -h / | tail -1)" >> out/mut/progress.txt; fi
done
echo "$(date -u +%T) ALL DONE $(du -xsh out/mut work/mut 2>/dev/null | tr '\n' ' ') $(df -h / | tail -1)" >> out/mut/progress.txt
