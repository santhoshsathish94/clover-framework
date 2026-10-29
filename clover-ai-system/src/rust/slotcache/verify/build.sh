#!/bin/bash
# Builds the differential driver (run on the box inside ~/rustmig/slotcache/verify).
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -Wall -Wextra -I../include -o diff_driver diff_driver.cpp -ldl
ls -la diff_driver
