#!/bin/bash
# Per-seed comparison for the PM25 probe: for each race seed, does the driver FAIL exactly when the probe says a double start happened? (verifier; run in ~/rustmig/loader/verify)
cd "$(dirname "$0")/.."
f=/dev/shm/ldprobe-PM25.txt
agree=0; hit_pass=0; nohit_fail=0; hit_fail=0; nohit_pass=0
for s in $(seq 1 400); do
    rm -f $f
    out=$(ulimit -v 2500000; nice -n 10 timeout 60 ./ld_driver rust work/so/PM25.so race $s 2>&1)
    echo "$out" | grep -q '^RESULT .* PASS' && res=PASS || res=FAIL
    hits=$( [ -f $f ] && grep -c "double start" $f || echo 0 )
    if [ $hits -gt 0 ] && [ $res = PASS ]; then hit_pass=$((hit_pass+1)); echo "seed $s: probe hits=$hits but the driver PASSED" >> out/mut/pm25-seeds.txt; fi
    if [ $hits -eq 0 ] && [ $res = FAIL ]; then nohit_fail=$((nohit_fail+1)); echo "seed $s: no probe hit but the driver FAILED: $(echo "$out" | grep -m1 '^F ' | cut -c1-160)" >> out/mut/pm25-seeds.txt; fi
    [ $hits -gt 0 ] && [ $res = FAIL ] && hit_fail=$((hit_fail+1))
    [ $hits -eq 0 ] && [ $res = PASS ] && nohit_pass=$((nohit_pass+1))
done
rm -f $f
echo "PM25 race seeds 1-400 (one process each): probe hit & driver FAIL $hit_fail; probe hit & driver PASS $hit_pass; no hit & FAIL $nohit_fail; no hit & PASS $nohit_pass" >> out/mut/pm25-seeds.txt
