#!/bin/bash
# like run.sh, but with the PROPOSAL driver (work/proto/pi_driver_proto, see proto_driver.py). usage: mutants/proto_run.sh NAME... (mutant names; M00 = the unmodified crate); output out/proto/<name>/
cd "$(dirname "$0")/.."
R=$HOME/rustmig
ulimit -v ${VMEM_KB:-3000000}
[ -f "$PWD/mutants/nodump.so" ] && export LD_PRELOAD=$PWD/mutants/nodump.so
mkdir -p out/proto
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libpoolinit.so
  rm -rf out/proto/$n
  nice -n 10 work/proto/pi_driver_proto run --ref $R/poolinit/ref/libpoolinit_ref.so --rust "$so" --sc $R/slotcache/target/release/libslotcache.so --out out/proto/$n --jobs 8 --keep-raw 0 --keep-going > out/proto/$n.stdout 2>&1
  echo "$n: UNIT=$(grep -c '^UNIT' out/proto/$n/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' out/proto/$n/units.log) $(grep '^H ' out/proto/$n/summary.txt | cut -c1-60) first: $(grep -h 'PROTO' out/proto/$n/*/u*.rust.log 2>/dev/null | head -1 | cut -c1-220) PROTO-violations: $(grep -l 'PROTO' out/proto/$n/*/u*.rust.log 2>/dev/null | wc -l)"
  find out/proto/$n -type f \( -name '*.err' -o -name '*.res' -o -name '*.log' ! -name units.log \) -delete
done
