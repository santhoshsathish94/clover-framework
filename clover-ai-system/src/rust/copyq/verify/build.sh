#!/bin/bash
# Builds diff_driver (box, in ~/rustmig/copyq/verify). Records md5 of the files this verifier must not modify.
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.6/bin:$PATH
mkdir -p out data mutants
[ -f out/md5-before.txt ] || md5sum ../ref/copyq_ref.cu ../ref/libcopyq_ref.so ../target/release/libcopyq.so ../include/copyq.h ../src/*.rs > out/md5-before.txt
nvcc -O2 -std=c++17 -I../include diff_driver.cpp -o diff_driver -ldl -lpthread 2>&1 | head -60
ls -la diff_driver
cat out/md5-before.txt
