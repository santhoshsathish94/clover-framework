#!/bin/bash
# Driver v2, the whole proof in one detached job (box, ~/rustmig/poolinit/verify; log out/v2/campaign.log, one line per step):
#   1. control: the unmodified crate against the reference, default counts (3,000 units) + scenario A (60), three times, then 20 x the counts (60,000 + 60) once
#   2. every mutant of mutants/md5list.txt (123 = M00 + M01..M122) on the new driver, 3,000 + 60 units each, JOBS 8, 3 GB cap (v2/run_mut.sh)
#   3. the mutants argued equivalent / out of domain / byte-identical (M47 M48 M49 M59 M67 M68 M86 M117) at 20 x the counts
# usage: (setsid nohup bash v2/campaign.sh > out/v2/campaign.out 2>&1 < /dev/null &)
cd "$(dirname "$0")/.."
mkdir -p out/v2
L=out/v2/campaign.log
say() { echo "$(date -u +%T) $*" >> $L; }
say "start: $(df -h / | tail -1)"
for i in 1 2 3; do say "ctl-$i"; bash v2/ctl.sh $i 1 > out/v2/ctl-$i.txt 2>&1; say "ctl-$i done: $(grep -c ' PASS$' out/v2/ctl-$i.txt) scenario lines PASS, $(grep recount out/v2/ctl-$i.txt)"; done
say "ctl-20x"; bash v2/ctl.sh 20x 20 > out/v2/ctl-20x.txt 2>&1; say "ctl-20x done: $(grep -c ' PASS$' out/v2/ctl-20x.txt) scenario lines PASS, $(grep recount out/v2/ctl-20x.txt)"
say "mutants: $(df -h / | tail -1)"
bash v2/run_mut.sh $(awk '{print $1}' mutants/md5list.txt | sort -V)
say "mutants done"
SUB=v2big MULT=20 JOBS=12 bash v2/run_mut.sh M47 M48 M49 M59 M67 M68 M86 M117
say "big done: $(du -xsh out/v2big | cut -f1); $(df -h / | tail -1)"
