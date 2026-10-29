#!/bin/bash
# Recounts a run from units.log with awk (independent of the driver's own summary). Usage: ./count.sh OUTDIR
# Fields of "UNIT scen seed cmp chk cases PASS|FAIL|CRASH": $2 scen, $3 seed, $4 cmp, $5 chk, $6 cases, $7 verdict.
d=${1:?usage: count.sh OUTDIR}
echo "BEGIN lines: $(grep -c '^BEGIN' "$d/units.log")   UNIT lines: $(grep -c '^UNIT' "$d/units.log")"
awk '$1=="UNIT" { u[$2]++; c[$2]+=$4; k[$2]+=$5; n[$2]+=$6; if ($7=="PASS") p[$2]++; else f[$2]++; tc+=$4; tn+=$6; tu++ }
     END { for (s in u) printf "%s units=%d pass=%d notpass=%d cmp=%d chk=%d cases=%d\n", s, u[s], p[s], f[s], c[s], k[s], n[s];
           printf "TOTAL units=%d cmp=%d cases=%d\n", tu, tc, tn }' "$d/units.log" | sort
ls "$d/fail" 2>/dev/null | head -5
