#!/bin/bash
# Builds libslotcache_ref.so (and the smoke test) from slotcache_ref.cpp + slotcache_extracted.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/slotcache/ref with ../include/slotcache.h present.
set -e
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -fPIC -shared -I../include slotcache_ref.cpp -o libslotcache_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -o smoke_ref smoke_ref.c -L. -lslotcache_ref -Wl,-rpath,'$ORIGIN'
ls -la libslotcache_ref.so smoke_ref
