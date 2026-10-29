#!/bin/bash
cd "$(dirname "$0")"
bash one.sh cpp S1 1
bash one.sh rust S1 1
head -3 out/cpp/S1-1.txt; tail -4 out/cpp/S1-1.txt; head -c 600 out/cpp/S1-1.err
cmp out/cpp/S1-1.txt out/rust/S1-1.txt && echo S1-seed1-cpp-rust-IDENTICAL
