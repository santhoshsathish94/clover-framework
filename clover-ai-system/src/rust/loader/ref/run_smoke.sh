#!/bin/bash
# Runs the smoke test in all modes against whatever is linked into the given binary. Usage: ./run_smoke.sh ./smoke_ref   (run in the directory of the binary)
set -u
bin=${1:?binary}
rc=0
for m in basic noprefill prefill missing eof badcard; do
    "$bin" "$m" || rc=1
done
exit $rc
