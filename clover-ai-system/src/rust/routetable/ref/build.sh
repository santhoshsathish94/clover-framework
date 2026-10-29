#!/bin/bash
# Builds libroutetable_ref.so (and the smoke test) from routetable_ref.cpp + routetable_extracted.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/routetable/ref with ../include/routetable.h present.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared -I../include routetable_ref.cpp -o libroutetable_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -o smoke_ref smoke_ref.c -L. -lroutetable_ref -Wl,-rpath,'$ORIGIN'
ls -la libroutetable_ref.so smoke_ref
