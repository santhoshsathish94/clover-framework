#!/bin/bash
# md5 of everything the verification depends on (run on the box inside ~/rustmig/routetable/verify).
cd "$(dirname "$0")/.."
md5sum include/routetable.h src/lib.rs src/tests.rs Cargo.toml ref/routetable_ref.cpp ref/routetable_extracted.inc ref/libroutetable_ref.so target/release/libroutetable.so \
  verify/diff_driver.cpp verify/build.sh verify/run.sh verify/asan.sh verify/md5.sh verify/count.sh verify/diff_driver 2>/dev/null
ls ref | sed 's/^/ref-file: /'
md5sum ~/k3gpu/results/p1d.trace; stat -c '%n %s bytes mtime %y' ~/k3gpu/results/p1d.trace
