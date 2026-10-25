#!/bin/bash
cd ~/rustmig/planner/verify
rm -rf out/smoke1
for s in P1 P2; do
  ./run.sh smoke1 --scn $s --n $s=3 --jobs 4 2>&1 | tail -8
done
ls out/smoke1/P1 | head -20
cat out/smoke1/units.log | head -20
