#!/bin/bash
# Builds every mutant copy (cargo build --release), then reports: md5 of .so, md5 of the real .so, whether Cargo.toml/Cargo.lock/tests.rs are identical to the real ones, number of diff hunks of lib.rs.
export PATH=$HOME/.cargo/bin:$PATH
REAL=$HOME/rustmig/slotcache
W=$REAL/verify/work/mut
L=$REAL/verify/out/mut
mkdir -p $L
echo "real .so md5: $(md5sum < $REAL/target/release/libslotcache.so)"
for n in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  m=M$n
  cd $W/$m || exit 1
  cargo build --release --offline > $L/build-$m.log 2>&1
  rc=$?
  so=$W/$m/target/release/libslotcache.so
  sm=$( [ -f $so ] && md5sum < $so | cut -d' ' -f1 || echo NOSO )
  same=ok
  for f in Cargo.toml Cargo.lock src/tests.rs; do cmp -s $f $REAL/$f || same="DIFFERS:$f"; done
  hunks=$(diff -u $REAL/src/lib.rs $W/$m/src/lib.rs | grep -c '^@@')
  warn=$(grep -c '^warning' $L/build-$m.log)
  echo "$m build_rc=$rc so_md5=$sm other_files=$same lib_rs_hunks=$hunks warnings=$warn lib_rs_md5=$(md5sum < src/lib.rs | cut -d' ' -f1)"
done
