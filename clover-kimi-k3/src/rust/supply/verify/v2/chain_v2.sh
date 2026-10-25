#!/bin/bash
# The follow-up steps of the driver-v2 check, one after the other, started once the campaign (v2/run_set.sh M00..M97 X1..X3) has written "ALL DONE": compare with the first pass, the probes,
# the final three runs of the unmodified crate (cargo build/test, md5, 3 x 1120 units), the three-at-once load test, and the repeat runs. Output: v2/logs/chain-*.out, v2/logs/chain.done.
cd "$(dirname "$0")/.."
while ! grep -q "ALL DONE" out/mut/progress.txt 2>/dev/null; do sleep 5; done
python3 v2/compare_v2.py 1120 > v2/logs/chain-compare.out 2>&1
v2/probes_v2.sh > v2/logs/chain-probes.out 2>&1
v2/final_v2.sh > v2/logs/chain-final.out 2>&1
v2/stress_v2.sh > v2/logs/chain-stress.out 2>&1
REPEATS=2 v2/rerun_v2.sh M00 M42 M43 M45 M52 M62 M63 M64 M65 M67 M74 M75 M76 M91 M93 M16 M26 M41 M57 M81 M90 M92 M97 X1 X2 X3 > v2/logs/chain-rerun.out 2>&1
date -u > v2/logs/chain.done
