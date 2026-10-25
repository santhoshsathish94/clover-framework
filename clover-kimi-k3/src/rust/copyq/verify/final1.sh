#!/bin/bash
cd "$(dirname "$0")"
bash compare.sh
echo "=== matrix.log (final run)"; cut -c1-200 out/matrix.log
echo "=== run1 (driver with the wrong S3 expectation) S3 lines"; grep -E " S3 " out/run1/matrix.log | cut -c1-220
echo "=== md5 untouchable files now vs before"
md5sum ../ref/copyq_ref.cu ../ref/libcopyq_ref.so ../target/release/libcopyq.so ../include/copyq.h ../src/*.rs | diff - out/md5-before.txt && echo UNCHANGED
echo "=== stderr of the runs (non-empty only)"; find out -name '*.err' -size +0 | head
