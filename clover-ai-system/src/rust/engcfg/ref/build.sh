#!/bin/bash
# Builds libengcfg_ref.so (and the smoke test) from engcfg_ref.cpp + x/*.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/engcfg/ref with ../include/engcfg.h present.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared -Wl,-Bsymbolic -Wall -Wextra -Wno-unused-variable -Wno-unused-function -I../include -I. engcfg_ref.cpp -o libengcfg_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -I../include -o smoke_ref smoke_ref.c -L. -lengcfg_ref -Wl,-rpath,'$ORIGIN'
ls -la libengcfg_ref.so smoke_ref
