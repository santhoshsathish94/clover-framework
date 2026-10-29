#!/bin/bash
# Builds libcopyq_ref.so from copyq_ref.cu (made by extract.py). Run on the box inside ~/rustmig/copyq/ref with ../include present.
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.6/bin:$PATH
nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -fPIC -Xcompiler -pthread -shared -I../include copyq_ref.cu -o libcopyq_ref.so
ls -la libcopyq_ref.so
