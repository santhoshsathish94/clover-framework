#!/bin/bash
# Builds the differential driver on the box (no GPU, no sudo). The sibling include dirs are the existing read-only ~/rustmig/{copyq,slotcache,routetable}/include.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -pthread -I../include -I../../copyq/include -I../../slotcache/include -I../../routetable/include pl_driver.cpp -o pl_driver -ldl
echo "built $(pwd)/pl_driver"
