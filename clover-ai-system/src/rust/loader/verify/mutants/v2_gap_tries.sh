#!/bin/bash
# Catch rates of the v2 scenarios against the mutants of the real gaps (and the control M00 for false positives): mutants/v2_gap_tries.sh [P]
# Each line: mutant scenario N. Output: out/try/summary.txt (one TRY line each; units.log of each in out/try/NAME.SCENARIO.log)
cd "$(dirname "$0")/.."
P=${1:-10}
mkdir -p out/try
while read -r m s n; do
    [ -z "$m" ] && continue
    mutants/try_mut.sh $m $s $n $P >> out/try/summary.txt 2>&1
done <<'LIST'
M00 book 100
M00 hotwait 100
M00 race2 100
M00 big 40
M00 fd0 40
M00 stack 40
M00 guard 40
M82 book 100
M83 book 50
M83 open 50
M84 open 100
M33 open 50
M46 open 100
M51 fd0 40
M58 big 40
M59 big 40
M27 stack 40
M87 guard 40
M92 guard 40
M25 race2 100
M05 hotwait 100
M11 hotwait 100
M14 hotwait 100
M15 hotwait 20
M05w hotwait 20
M11w hotwait 20
LIST
echo "TRIES DONE $(date -u +%T)" >> out/try/summary.txt
