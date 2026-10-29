#!/bin/bash
# phase 3: after phase 2 has finished, run the corrected probes (one write() per line), summarise them, and run the proposal driver against the mutants it is meant to kill.
cd ~/rustmig/poolinit/verify || exit 1
while ! grep -q "PHASE2 DONE" out/phase2.log 2>/dev/null; do sleep 10; done
PROBES="M00c M03p M11p M12p M13p M14p M33p M46p M47p M48p M49p M57p M58p M59p M67p M68p M86p"
echo "== probes $(date -u +%T)" > out/phase3.log
rm -rf out/probe
mutants/run_probes.sh $PROBES >> out/phase3.log 2>&1
python3 mutants/probe_count.py out/probe/*.events > out/probe/summary.txt 2>&1
echo "== proposal driver $(date -u +%T)" >> out/phase3.log
mutants/proto_run.sh M00 M03 M11 M12 M13 M14 M15 M33 M46 M47 M48 M49 M57 M58 M59 M67 M68 M86 >> out/phase3.log 2>&1
echo "== PHASE3 DONE $(date -u +%T)" >> out/phase3.log
