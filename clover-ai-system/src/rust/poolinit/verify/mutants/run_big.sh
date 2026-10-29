#!/bin/bash
# The big run for mutants that the 3,000-unit run did not catch: 20 times the default counts (60,000 units: H 12000, HX 8000, P 12000, PX 16000, PR 4000, M 8000), one mutant at a time,
# output in out/big/<name>/ (units.log, summary.txt, .res; raw logs deleted), results in out/big/results.tsv, progress in out/big/progress.txt. Same caps as run_all.sh.
# usage (box, ~/rustmig/poolinit/verify): mutants/run_big.sh NAME...
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
ulimit -v ${VMEM_KB:-3000000}
[ -f "$PWD/mutants/nodump.so" ] && export LD_PRELOAD=$PWD/mutants/nodump.so
mkdir -p out/big
export SUB=big MULT=${MULT:-20}
N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))"
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libpoolinit.so
  [ -f "$so" ] || { echo "$(date -u +%T) $n NO SO" >> out/big/progress.txt; continue; }
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$(date -u +%T) $n STOP free=${free}G" >> out/big/progress.txt; exit 3; fi
  md5=$(md5sum "$so" | cut -c1-32)
  echo "$(date -u +%T) $n start md5=$md5 free=${free}G mult=$MULT" >> out/big/progress.txt
  rm -rf out/big/$n
  RUST=$so JOBS=${JOBS:-12} timeout 3000 ./run.sh big/$n --keep-raw 0 --keep-going --n "$N" > out/big/$n.stdout 2>&1
  rc=$?
  if [ $rc -eq 124 ]; then pkill -u "$USER" -f "pi_driver .*out/big/$n( |/)"; fi
  echo "$(date -u +%T) $n done rc=$rc recount: UNIT=$(grep -c '^UNIT' out/big/$n/units.log) PASS=$(grep -c '^UNIT .* PASS ' out/big/$n/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' out/big/$n/units.log) distinct=$(grep '^UNIT' out/big/$n/units.log | awk '{print $2,$3}' | sort -u | wc -l) res=$(find out/big/$n -name '*.res' | wc -l)" >> out/big/progress.txt
  python3 mutants/analyze.py $n $md5 >> out/big/analysis.log 2>&1
  find out/big/$n -type f \( -name '*.log' -o -name '*.log.gz' -o -name '*.trace' -o -name '*.err' -o -name '*.res' \) ! -name units.log -delete   # 60000 .res files = 240 MB of blocks per run; units.log + summary.txt keep the result
  echo "$(date -u +%T) $n cleaned: $(du -xsh out/big | cut -f1) $(df -h / | tail -1)" >> out/big/progress.txt
done
echo "$(date -u +%T) ALL DONE $(du -xsh out/big | cut -f1) $(df -h / | tail -1)" >> out/big/progress.txt
