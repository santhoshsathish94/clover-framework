#!/bin/bash
# Driver v2: runs ONE unit of one scenario against one mutant (work/v2so/<name>.so) and prints the unit line and the violation, to show a first failure of a given gate mode.
# usage (box, ~/rustmig/poolinit/verify): bash v2/unit_demo.sh NAME SCN GATE [after-seed]      the first seed > after-seed (default 0) of scenario SCN whose first host step has gate GATE (0, 1 or 2)
cd "$(dirname "$0")/.."
NAME=${1:?name}; SCN=${2:?scn}; GATE=${3:?gate}; AFTER=${4:-0}
R=$HOME/rustmig
ulimit -v 3000000
seed=$((AFTER+1))
while [ $seed -lt 5000 ]; do
  if ./pi_driver cfg --scn $SCN --seed $seed | grep -q " gate=$GATE\$"; then break; fi
  seed=$((seed+1))
done
echo "scenario $SCN seed $seed: $(./pi_driver cfg --scn $SCN --seed $seed | cut -c1-200)"
D=out/demo-$NAME; rm -rf $D; mkdir -p $D
./pi_driver unit --scn $SCN --seed $seed --ref $R/poolinit/ref/libpoolinit_ref.so --rust $PWD/work/v2so/$NAME.so --sc $R/slotcache/target/release/libslotcache.so --out $D --keep-raw 0
grep -h VIOLATION $D/$SCN/u$seed.rust.log | cut -c1-400
rm -rf $D
