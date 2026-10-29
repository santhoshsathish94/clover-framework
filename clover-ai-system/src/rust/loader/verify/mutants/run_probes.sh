#!/bin/bash
# Extra probes for the racy / weak-window mutants (verifier's own, the crate's driver is untouched). Run on the box in ~/rustmig/loader/verify.
#   mutants/run_probes.sh lw  NAME... : lw_probe (counter / ld_wait stress) against work/so/NAME.so, 3 runs each (rounds 20000, 4 waiters), output out/mut/lw-probe.txt
#   mutants/run_probes.sh bk  NAME... : the matrix of the open scenarios (open/shm) with the patched driver copy (mk_driver_bk.py), 1 run each, output out/mut/bk-probe.txt
cd "$(dirname "$0")/.."
mode=$1; shift
case $mode in
lw)
  gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I../include -o mutants/lw_probe mutants/lw_probe.c -ldl || exit 1
  for n in "$@"; do
    so=$PWD/work/so/$n.so
    for run in 1 2 3; do
      printf '%s run%s: ' $n $run >> out/mut/lw-probe.txt
      nice -n 10 timeout 150 mutants/lw_probe $so 20000 4 $run >> out/mut/lw-probe.txt 2>&1 || printf 'rc=%s ' $? >> out/mut/lw-probe.txt
      echo >> out/mut/lw-probe.txt
    done
  done ;;
bk)
  python3 mutants/mk_driver_bk.py || exit 1
  g++ -O1 -std=c++17 -D_FORTIFY_SOURCE=0 -Wall -Wextra -pthread -rdynamic -I../include -o mutants/probe/ld_driver_bk mutants/probe/ld_driver_bk.cpp -ldl || exit 1
  for n in "$@"; do
    so=$PWD/work/so/$n.so
    for scn in open shm; do
      f=0; t=0
      for s in $(seq 1 200); do
        out=$(ulimit -v 2500000; nice -n 10 timeout 60 mutants/probe/ld_driver_bk rust $so $scn $s 2>&1)
        t=$((t+1)); echo "$out" | grep -q "^RESULT .* PASS" || { f=$((f+1)); [ $f = 1 ] && echo "$n $scn $s: $(echo "$out" | grep -m1 '^F ' | cut -c1-200)" >> out/mut/bk-probe.txt; }
      done
      echo "$n $scn fail=$f of $t (patched driver, seeds 1-200)" >> out/mut/bk-probe.txt
    done
  done
  for d in /tmp/ldloader-verify.* /dev/shm/ldloader-verify.*; do [ -e "$d" ] || continue; pid=${d##*.}; kill -0 $pid 2>/dev/null || rm -rf "$d"; done ;;
esac
