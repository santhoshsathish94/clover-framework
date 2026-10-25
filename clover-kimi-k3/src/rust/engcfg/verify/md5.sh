#!/bin/bash
# Hashes everything a verification run depends on (box, inside ~/rustmig/engcfg/verify). Usage: ./md5.sh
cd "$(dirname "$0")/.."
md5sum include/engcfg.h Cargo.toml src/lib.rs src/tests.rs smoke_rust.c ref/extract.py ref/engcfg_ref.cpp ref/ecref.h ref/build.sh ref/smoke_ref.c ref/x/MANIFEST.txt \
       ref/libengcfg_ref.so target/release/libengcfg.so verify/diff_driver.cpp verify/diff_driver
echo "extracted slices (md5 of the concatenation of ref/x/*.inc in name order): $(cat ref/x/*.inc | md5sum | cut -d' ' -f1)"
