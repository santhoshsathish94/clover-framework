#!/bin/bash
# Catch rate of one scenario against one mutant: mutants/try_mut.sh NAME SCENARIO N [P]  (seeds 1..N, at most P units at a time, default 10; the units.log is kept in out/try/NAME.SCENARIO.log)
cd "$(dirname "$0")/.."
name=$1; scn=$2; N=${3:-50}; P=${4:-10}
so=${SODIR:-$PWD/work/so}/$name.so
out=$PWD/out/try/$name.$scn; rm -rf "$out"; mkdir -p "$out"
seq 1 $N | sed "s/^/$scn /" | xargs -P $P -L 1 sh -c "mutants/mut_unit.sh \$0 \$1 $out $so"
fail=$(grep -c ' FAIL ' "$out/units.log"); total=$(grep -c '^UNIT' "$out/units.log"); ref=$(grep -c ' REFBAD ' "$out/units.log")
echo "TRY $name $scn: FAIL $fail of $total (REFBAD $ref)  first: $(grep -m1 ' FAIL ' "$out/units.log" | sed 's/.*| //' | cut -c1-200)"
mv "$out/units.log" "$out/../$name.$scn.log"; rm -rf "$out"
