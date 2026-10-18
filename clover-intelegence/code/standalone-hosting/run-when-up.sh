#!/bin/bash
# One command once the host answers: build the detokeniser, then run the four
# configurations and print the text each one produced.
set -uo pipefail
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd "$D/tansformers" || exit 1
gcc -O2 -o /tmp/detokenise detokenise.c -lm || { echo "detokenise did not build"; exit 1; }
echo "round trip: [$(echo '1008 10484 318 15383 387' | /tmp/detokenise "$D")]"
exec bash /tmp/run-snapshot-scheme.sh "${1:-16}"
