#!/bin/bash
# Builds libloader_ref.so and the smoke test from loader_ref.cpp + loader_a.inc + loader_b.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/loader/ref with ../include/loader.h present. -D_FORTIFY_SOURCE=0: the Ubuntu default would turn open() into __open_2(), which a test could not interpose.
set -e
cd "$(dirname "$0")"
nice -n 10 g++ -O2 -std=c++17 -fPIC -shared -pthread -Wall -Wextra -D_FORTIFY_SOURCE=0 -I../include loader_ref.cpp -o libloader_ref.so
nice -n 10 gcc -O1 -std=gnu11 -Wall -Wextra -pthread -D_FORTIFY_SOURCE=0 -I../include -o smoke_ref smoke_ref.c -L. -lloader_ref -Wl,-rpath,'$ORIGIN'
ls -la libloader_ref.so smoke_ref
md5sum libloader_ref.so
