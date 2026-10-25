#!/bin/bash
# phase 4: after phase 3, rebuild the proposal driver with the exact lower bound and re-run it against the mutants it is meant to kill (+ control)
cd ~/rustmig/poolinit/verify || exit 1
while ! grep -q "PHASE3 DONE" out/phase3.log 2>/dev/null; do sleep 10; done
python3 mutants/proto_driver.py > out/phase4.log 2>&1
mutants/proto_run.sh M00 M03 M11 M12 M13 M14 M15 >> out/phase4.log 2>&1
echo "== PHASE4 DONE $(date -u +%T)" >> out/phase4.log
