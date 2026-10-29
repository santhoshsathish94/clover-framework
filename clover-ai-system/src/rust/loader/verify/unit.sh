#!/bin/bash
# One unit: scenario $1, seed $2, output dir $3: runs the driver against the reference and against the Rust library (one after the other), compares the canonical ("C ") lines.
scn=$1; seed=$2; out=$3
here=$(cd "$(dirname "$0")" && pwd)
ref=$here/../ref/libloader_ref.so
rust=$here/../target/release/libloader.so
for lab in ref rust; do
    lib=$ref; [ $lab = rust ] && lib=$rust
    nice -n 10 timeout 300 "$here/ld_driver" $lab "$lib" $scn $seed > "$out/$scn.$seed.$lab.txt" 2> "$out/$scn.$seed.$lab.err"
    eval "rc_$lab=\$?"
    [ -s "$out/$scn.$seed.$lab.err" ] || rm -f "$out/$scn.$seed.$lab.err"
done
cref=$(grep '^C ' "$out/$scn.$seed.ref.txt" | md5sum | cut -c1-8)
crust=$(grep '^C ' "$out/$scn.$seed.rust.txt" | md5sum | cut -c1-8)
res_ref=$(grep '^RESULT' "$out/$scn.$seed.ref.txt" | awk '{print $4}'); res_rust=$(grep '^RESULT' "$out/$scn.$seed.rust.txt" | awk '{print $4}')
same=DIFF; [ "$cref" = "$crust" ] && [ -n "$cref" ] && same=SAME
printf 'UNIT %s %s ref_rc=%s ref=%s rust_rc=%s rust=%s canon=%s %s\n' $scn $seed $rc_ref "${res_ref:-NONE}" $rc_rust "${res_rust:-NONE}" $same $cref >> "$out/units.txt"
