#!/bin/bash
# One unit against one mutant: ./mut_unit.sh <scenario> <seed> <outdir> <mutant.so>
# Same as verify/unit.sh (ref and the library, canonical "C " lines compared) with three differences: the Rust library is the mutant given as $4, every process gets
# `timeout 60` and a 2.5 GB address-space limit (a mutant that loops on pread would otherwise log reads without end), and raw files are removed at once (kept only for a FAIL).
scn=$1; seed=$2; out=$3; so=$4
# MAXFAIL=K (optional): once K units of this run have failed, the remaining units are not started (a mutant that hangs most units would cost hours); the run is then partial and says so.
if [ -n "$MAXFAIL" ] && [ -f "$out/units.log" ] && [ "$(grep -c ' FAIL ' "$out/units.log")" -ge "$MAXFAIL" ]; then exit 0; fi
here=$(cd "$(dirname "$0")/.." && pwd)
ref=$here/../ref/libloader_ref.so
ulimit -v 2500000
for lab in ref rust; do
    lib=$ref; [ $lab = rust ] && lib=$so
    nice -n 10 timeout 60 "$here/ld_driver" $lab "$lib" $scn $seed > "$out/$scn.$seed.$lab.txt" 2> "$out/$scn.$seed.$lab.err"
    eval "rc_$lab=\$?"
done
cref=$(grep '^C ' "$out/$scn.$seed.ref.txt" | md5sum | cut -c1-8)
crust=$(grep '^C ' "$out/$scn.$seed.rust.txt" | md5sum | cut -c1-8)
res_ref=$(grep '^RESULT' "$out/$scn.$seed.ref.txt" | awk '{print $4}'); res_rust=$(grep '^RESULT' "$out/$scn.$seed.rust.txt" | awk '{print $4}')
same=DIFF; [ "$cref" = "$crust" ] && [ -n "$cref" ] && same=SAME
st=PASS
{ [ "$res_rust" = PASS ] && [ "$rc_rust" = 0 ] && [ $same = SAME ]; } || st=FAIL
[ "$res_ref" = PASS ] && [ "$rc_ref" = 0 ] || st=REFBAD
first=""
if [ $st != PASS ]; then
    first=$(grep -m1 '^F ' "$out/$scn.$seed.rust.txt" | cut -c1-230)
    [ -z "$first" ] && [ "$same" = DIFF ] && [ "$res_rust" = PASS ] && first="canonical lines differ from the reference"
    [ -z "$first" ] && first="no F line (rc=$rc_rust result=${res_rust:-NONE}); stderr: $(head -c 150 "$out/$scn.$seed.rust.err" | tr '\n' ' ')"
    head -c 30000 "$out/$scn.$seed.rust.txt" > "$out/$scn.$seed.fail.txt"
fi
rm -f "$out/$scn.$seed".ref.txt "$out/$scn.$seed".rust.txt "$out/$scn.$seed".ref.err "$out/$scn.$seed".rust.err
printf 'UNIT %s %s %s ref_rc=%s ref=%s rust_rc=%s rust=%s canon=%s | %s\n' $scn $seed $st $rc_ref "${res_ref:-NONE}" $rc_rust "${res_rust:-NONE}" $same "$first" >> "$out/units.log"
