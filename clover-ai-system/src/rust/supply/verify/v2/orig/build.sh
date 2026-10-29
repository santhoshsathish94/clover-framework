#!/bin/bash
# Builds the differential driver on the box (no GPU, no sudo). The sibling include dir is the existing read-only ~/rustmig/slotcache/include.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -pthread -I../include -I../../slotcache/include sy_driver.cpp -o sy_driver -ldl
echo "built $(pwd)/sy_driver"
