#!/bin/bash
# Phase 2 of the mutation campaign (verifier), one run at a time: r1 of the two widened twins, r2 and r3 of the racy mutants, the probe builds' matrix, the N=1000 soak of the uncaught mutants.
# usage (box, ~/rustmig/loader/verify): nohup mutants/phase2.sh > /dev/null 2>&1 &     progress in out/mut/progress.txt
cd "$(dirname "$0")/.."
RACY="M02 M05 M05w M06 M07 M09 M10 M11 M11w M13 M14 M15 M16 M17 M22 M25 M26 M82 M83"
UNC="M05 M06 M17 M27 M50 M51 M58 M59 M75 M76 M82 M83 M84 M87 M92"
echo "$(date -u +%T) PHASE2 start" >> out/mut/progress.txt
MAXFAIL=100 N=200 TAG=r1 mutants/run_mut.sh M05w M11w
MAXFAIL=100 N=200 TAG=r2 mutants/run_mut.sh $RACY
MAXFAIL=100 N=200 TAG=r3 mutants/run_mut.sh $RACY
rm -f /dev/shm/ldprobe-*.txt
N=200 TAG=probe mutants/run_mut.sh PM05 PM14 PM25 PM50 PM58 PM59 PM51 PM87 PM92
N=1000 TAG=soak mutants/run_mut.sh M00 $UNC
echo "$(date -u +%T) PHASE2 done" >> out/mut/progress.txt
