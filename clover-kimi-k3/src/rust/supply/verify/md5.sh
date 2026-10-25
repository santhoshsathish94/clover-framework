#!/bin/bash
# md5 of every file that defines what was run (on the box, after sync).
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
echo "rustc: $(rustc --version)   cargo: $(cargo --version)   g++: $(g++ --version | head -1)"
md5sum include/supply.h ref/supply_ref.cpp ref/supply_types.inc ref/supply_extracted.inc ref/libsupply_ref.so src/*.rs Cargo.toml target/release/libsupply.so verify/sy_driver.cpp verify/sy_driver ../slotcache/target/release/libslotcache.so 2>&1
