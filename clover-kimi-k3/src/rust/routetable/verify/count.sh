#!/bin/bash
# Recount of a run from the raw log only (independent of the driver's own summary). Usage: ./count.sh OUTDIR
L=${1:?outdir}/units.log
for s in R1 R2 R3 R4; do
  awk -v s=$s '$1=="BEGIN" && $2==s {b++} $1=="UNIT" && $2==s {u++; c+=$4; k+=$5; n+=$6; if ($9=="PASS") p++; else f++} $1=="CRASH" && $2==s {x++}
    END { if (b+u>0) printf "%s begin=%d unit=%d pass=%d fail=%d crash=%d comparisons=%d checks=%d next_use=%d\n", s, b, u, p, f, x, c, k, n }' "$L"
done
