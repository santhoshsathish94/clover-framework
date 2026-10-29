#!/bin/bash
# Phase 3 (verifier): the probe matrix again with the one-write probe logger, the probe sums, lw_probe on the PM05 probe (does the counter wrap happen?), the 4 MB stack probe on M00 and M27, md5 list of every .so.
# usage (box, ~/rustmig/loader/verify): nohup mutants/phase3.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.."
python3 mutants/mk_probes.py > out/mut/probe-build.txt 2>&1
rm -f /dev/shm/ldprobe-*.txt
N=200 TAG=probe mutants/run_mut.sh PM05 PM14 PM25 PM50 PM58 PM59 PM51 PM87 PM92
python3 mutants/probe_sum.py PM05 PM14 PM25 PM50 PM58 PM59 PM51 PM87 PM92 > out/mut/probe-sum.txt 2>&1
cp /dev/shm/ldprobe-PM25.txt out/mut/ldprobe-PM25.sample.txt 2>/dev/null
rm -f /dev/shm/ldprobe-PM05.txt out/mut/lw-probe-pm05.txt
for run in 1 2 3; do printf 'PM05 lw run%s: ' $run >> out/mut/lw-probe-pm05.txt; nice -n 10 timeout 150 mutants/lw_probe work/so/PM05.so 20000 4 $run >> out/mut/lw-probe-pm05.txt 2>&1 || printf 'rc=%s\n' $? >> out/mut/lw-probe-pm05.txt; done
python3 mutants/probe_sum.py PM05 >> out/mut/lw-probe-pm05.txt 2>&1
rm -f out/mut/stack-probe.txt
for n in M00 M27; do printf '%s: ' $n >> out/mut/stack-probe.txt; nice -n 10 timeout 60 mutants/lw_probe work/so/$n.so 1 1 1 stack >> out/mut/stack-probe.txt 2>&1; echo "rc=$?" >> out/mut/stack-probe.txt; done
md5sum work/so/*.so > mutants/md5-so.txt
rm -f /dev/shm/ldprobe-*.txt
echo "$(date -u +%T) PHASE3 done" >> out/mut/progress.txt
