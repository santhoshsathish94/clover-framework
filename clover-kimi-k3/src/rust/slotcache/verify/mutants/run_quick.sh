#!/bin/bash
# Quick runs: REAL (control) and M1..M16, two driver runs at a time, 40 workers each. Output: verify/out/mut/<name>-quick/ (summary.txt, fail-*.txt, cases/), stdout/stderr/exit in <name>-quick.log
# Usage: run_quick.sh [names...]   (default: REAL M1 .. M16)
V=$HOME/rustmig/slotcache/verify
REF=$HOME/rustmig/slotcache/ref/libslotcache_ref.so
cd $V
one() {
  n=$1
  if [ "$n" = REAL ]; then cand=$HOME/rustmig/slotcache/target/release/libslotcache.so; else cand=$V/work/mut/$n/target/release/libslotcache.so; fi
  out=$V/out/mut/$n-quick
  rm -rf $out; mkdir -p $out
  s=$(date +%s)
  ./diff_driver $REF $cand --opsmul 0.1 --jobs 40 --out $out > $out.stdout 2> $out.stderr
  echo "exit $? wall $(( $(date +%s) - s )) s" > $out.exit
}
export -f one; export V REF
if [ $# -eq 0 ]; then set -- REAL M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14 M15 M16; fi
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c 'one {}'
echo ALLDONE > $V/out/mut/quick.done
