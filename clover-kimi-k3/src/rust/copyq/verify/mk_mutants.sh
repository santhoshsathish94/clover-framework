#!/bin/bash
# Builds the mutant libraries from mutated copies of the reference (ref/ files are only read; md5 checked afterwards).
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.6/bin:$PATH
python3 make_mutants.py || exit 1
for m in M1 M2 M2v M3 M4; do
  nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -fPIC -Xcompiler -pthread -shared -I../include mutants/copyq_ref_$m.cu -o mutants/lib$m.so 2>&1 | head -5
  ls -la mutants/lib$m.so
done
cat mutants/M1.diff mutants/M2.diff mutants/M3.diff
echo "--- md5 of untouchable files now vs before"
md5sum ../ref/copyq_ref.cu ../ref/libcopyq_ref.so ../target/release/libcopyq.so ../include/copyq.h ../src/*.rs | diff - out/md5-before.txt && echo UNCHANGED
