#!/bin/bash
cd ~/rustmig/planner/verify/out/smoke1
for s in P1 P2; do
  echo "== $s u1 cfg/end"; head -1 $s/u1.ref.log; tail -2 $s/u1.ref.log
  echo "== op codes"; zcat -f $s/u*.ref.log* | awk '{c[$1]++} END{for(k in c) printf "%s=%d ", k, c[k]; print ""}'
done
echo "== sample of P2 u1 hook region"; grep -n -m1 '^dK 5 ' P2/u1.ref.log; sed -n '/^dK 5 /,/^Z /p' P2/u1.ref.log | head -40
