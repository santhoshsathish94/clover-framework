#!/bin/bash
# phase 2 of the poolinit mutant study (box, ~/rustmig/poolinit/verify): control raw logs, probes, 20x runs and cargo tests of the mutants the 3,000-unit run did not catch.
cd ~/rustmig/poolinit/verify || exit 1
UNC="M03 M11 M12 M13 M14 M33 M46 M47 M48 M49 M57 M58 M59 M67 M68 M86 M117 M120 M121 M122"
PROBES="M00c M03p M11p M12p M13p M14p M33p M46p M47p M48p M49p M57p M58p M59p M67p M68p M86p"
mkdir -p out/probe out/ctl
echo "== control raw logs $(date -u +%T)" > out/phase2.log
( ulimit -v 3000000; export LD_PRELOAD=$PWD/mutants/nodump.so; rm -rf out/ctl/*; nice -n 10 ./run.sh ctl --keep-raw 4 --n H=3,HX=3,P=3,PX=3,PR=3,M=3 > out/ctl.stdout 2>&1 ) 
echo "== probes $(date -u +%T)" >> out/phase2.log
mutants/run_probes.sh $PROBES >> out/phase2.log 2>&1
python3 mutants/probe_count.py out/probe/*.events > out/probe/summary.txt 2>&1
echo "== big runs $(date -u +%T)" >> out/phase2.log
mutants/run_big.sh $UNC >> out/phase2.log 2>&1
echo "== cargo test $(date -u +%T)" >> out/phase2.log
mutants/run_ctest.sh $UNC >> out/phase2.log 2>&1
echo "== PHASE2 DONE $(date -u +%T)" >> out/phase2.log
