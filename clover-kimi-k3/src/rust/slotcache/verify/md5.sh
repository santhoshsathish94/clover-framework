#!/bin/bash
# md5 of the two libraries, the frozen inputs (header, Rust source, ref/) and every file of the verifier.
cd ~/rustmig/slotcache
md5sum ref/libslotcache_ref.so target/release/libslotcache.so include/slotcache.h src/lib.rs src/tests.rs Cargo.toml Cargo.lock ref/slotcache_ref.cpp ref/slotcache_extracted.inc ref/extract.py ref/build.sh ref/smoke_ref.c 2>&1
cd verify
md5sum diff_driver.cpp dd_engine.inc dd_cases.inc build.sh run.sh md5.sh diff_driver 2>&1
