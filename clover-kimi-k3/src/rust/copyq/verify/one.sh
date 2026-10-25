#!/bin/bash
# usage: one.sh TAG SCEN [SEED]   TAG = cpp | rust | mutant name (mutants/libTAG.so). Output: out/TAG/SCEN-SEED.{txt,err}; prints one summary line.
cd "$(dirname "$0")"; tag=$1; scen=$2; seed=${3:-1}
case $tag in cpp) lib=../ref/libcopyq_ref.so;; rust) lib=../target/release/libcopyq.so;; *) lib=mutants/lib$tag.so;; esac
mkdir -p out/$tag
f=out/$tag/$scen-$seed
t0=$(date +%s.%N)
timeout 1500 ./diff_driver $lib $scen --seed $seed --tag $tag > $f.txt 2> $f.err
rc=$?
t1=$(date +%s.%N)
echo "$tag $scen seed=$seed rc=$rc secs=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") | $(grep -E '^(SUMMARY|TIMING)' $f.txt | head -1) | $(grep -E '^RESULT' $f.txt)"
