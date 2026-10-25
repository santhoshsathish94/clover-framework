#!/bin/bash
# Builds the C++ reference library and its smoke test on the box (no GPU, no sudo). The sibling include dir is the existing read-only ~/rustmig/slotcache/include.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -fwrapv -pthread -I../include -I../../slotcache/include supply_ref.cpp -o libsupply_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I../include -I../../slotcache/include smoke_ref.c -o smoke_ref -L. -lsupply_ref -Wl,-rpath,"$PWD"
echo "built $(pwd)/libsupply_ref.so"
nm -D libsupply_ref.so | grep ' T ' | awk '{print $3}' | sort | tr '\n' ' '; echo
md5sum libsupply_ref.so
