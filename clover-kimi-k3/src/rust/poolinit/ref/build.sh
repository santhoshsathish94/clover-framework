#!/bin/bash
# Builds libpoolinit_ref.so (and the smoke test) from poolinit_ref.cpp + poolinit_wall.inc + poolinit_extracted.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/poolinit/ref with ../include/poolinit.h and the sibling include dir ../../slotcache/include present.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -pthread -I../include -I../../slotcache/include poolinit_ref.cpp -o libpoolinit_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I../include -I../../slotcache/include -o smoke_ref smoke_ref.c -L. -lpoolinit_ref -Wl,-rpath,'$ORIGIN'
ls -la libpoolinit_ref.so smoke_ref
md5sum libpoolinit_ref.so
