#!/bin/bash
# Restart helper (verifier): kills the running campaign (run_mut.sh, its xargs and the units of THIS verifier's work/so libraries, by explicit pids) and starts a new one.
# usage: restart.sh <MAXFAIL> <first-mutant-name>   (continues with every mutant after it in the list order, M11w/M05w/probes excluded)
cd "$(dirname "$0")/.."
MF=$1; FIRST=$2
for p in $(ps -u usergpu -o pid=,args= | grep -E "mutants/run_mut.sh|xargs -P 20 -L 1 sh -c mutants/mut_unit.sh|mutants/mut_unit.sh|verify/work/so/" | grep -v -E "grep|restart.sh" | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 2
cur=$(grep -E "start md5" out/mut/progress.txt | tail -1 | awk '{print $2}' | sed 's/\.r1//')
if grep -q "^[0-9:]* $cur.r1 done" out/mut/progress.txt; then echo "$cur is already done: nothing aborted, FIRST must be the next one"; cur=NONE; fi
echo "$(date -u +%T) $cur ABORTED by restart.sh (new MAXFAIL=$MF)" >> out/mut/progress.txt
rm -rf out/mut/$cur.r1
names=$(python3 mutants/mk_mutants.py --list | awk '{print $1}' | grep -v -E "^(M11w|M05w)$" | awk -v f="$FIRST" '$1==f{go=1} go{print}')
MAXFAIL=$MF N=200 TAG=r1 nohup mutants/run_mut.sh $names > /dev/null 2>&1 &
echo "restarted from $FIRST with MAXFAIL=$MF; aborted $cur"
