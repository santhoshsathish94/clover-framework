#!/bin/bash
# Every acceptance gate for the stage machine, in one detached run.
# Each gate asserts on what it produced. An exit code of 0 is not evidence here:
# test-ends.sh writes its answer to /tmp/t.out and returns 0 whatever came out.
set -uo pipefail
ROOT=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/verify-stages.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="

echo
echo "--- gate 1: layer 1 against its recorded fixtures ---"
( cd "$ROOT/tansformers/transformer-1" && bash test-reference.sh bin/dataset )
echo "layer-1 exit=$?"

echo
echo "--- gate 2: layers 2..92 against /tmp/layers.bin ---"
bash /tmp/verify-all.sh
echo "layers-2-92 exit=$?"
PASSED=$(grep -c '^PASS' /tmp/verify-all.log 2>/dev/null)
FAILED=$(grep -cE 'COMPILEFAIL|RUNFAIL|FAIL' /tmp/verify-all.log 2>/dev/null)
echo "layers-2-92 PASS lines=${PASSED:-0} FAIL lines=${FAILED:-0} (expect 91 and 0)"

echo
echo "--- gate 3: end to end, client free ---"
( cd "$ROOT/pipeline" && bash test-ends.sh )
echo "--- /tmp/t.out ---"
cat /tmp/t.out 2>/dev/null || echo "(no /tmp/t.out)"
echo "--- /tmp/t.err tail ---"
tail -5 /tmp/t.err 2>/dev/null || echo "(no /tmp/t.err)"
if grep -q 'REQUEST 0 OK' /tmp/t.out 2>/dev/null && grep -q 'REQUEST 1 OK' /tmp/t.out 2>/dev/null; then
    echo "end-to-end: both requests OK"
else
    echo "end-to-end: MISSING an OK request"
fi

echo
echo "=== finished $(date -u +%FT%TZ) ==="
