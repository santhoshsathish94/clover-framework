#!/bin/bash
# Builds the differential driver on the box (no GPU, no sudo). -rdynamic: the executable's mmap / madvise / pthread_create must be visible to the two libraries.
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -pthread -rdynamic -I../include -I../../slotcache/include pi_driver.cpp -o pi_driver -ldl
echo "built $(pwd)/pi_driver"
