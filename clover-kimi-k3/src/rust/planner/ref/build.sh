#!/bin/bash
# Builds libplanner_ref.so (and the smoke test) from planner_ref.cpp + planner_extracted.inc (made by extract.py on Windows).
# Run on the box inside ~/rustmig/planner/ref with ../include/planner.h and the sibling include dirs ../../{copyq,slotcache,routetable}/include present.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -I../include -I../../copyq/include -I../../slotcache/include -I../../routetable/include planner_ref.cpp -o libplanner_ref.so
gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I../include -I../../copyq/include -I../../slotcache/include -I../../routetable/include -o smoke_ref smoke_ref.c -L. -lplanner_ref -Wl,-rpath,'$ORIGIN'
ls -la libplanner_ref.so smoke_ref
