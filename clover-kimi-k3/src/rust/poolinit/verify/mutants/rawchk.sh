#!/bin/bash
# Raw stderr of the first host units of some not-caught mutants next to the original's, to read what the library really printed (the driver masks the seconds in its logs).
# usage (box, ~/rustmig/poolinit/verify): mutants/rawchk.sh NAME...      output: out/rawchk/<name>/ (deleted by the caller afterwards)
cd "$(dirname "$0")/.."
ulimit -v 3000000
[ -f "$PWD/mutants/nodump.so" ] && export LD_PRELOAD=$PWD/mutants/nodump.so
R=$HOME/rustmig
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libpoolinit.so
  rm -rf out/rawchk/$n
  nice -n 10 ./pi_driver run --ref $R/poolinit/ref/libpoolinit_ref.so --rust "$so" --sc $R/slotcache/target/release/libslotcache.so --out out/rawchk/$n --jobs 1 --keep-raw 12 --keep-going --scn H --n H=11 > /dev/null 2>&1
  echo "=== $n (md5 $(md5sum $so | cut -c1-8)): unit result lines, then stderr of ref | rust for seeds 1..11"
  for s in 1 2 3 4 5 6 7 8 9 10 11; do
    echo "seed $s: $(head -c 300 out/rawchk/$n/H/u$s.res | tr '\n' ' ')"
    echo "   ref : $(cat out/rawchk/$n/H/u$s.ref.err)"
    echo "   rust: $(cat out/rawchk/$n/H/u$s.rust.err)"
  done
done
