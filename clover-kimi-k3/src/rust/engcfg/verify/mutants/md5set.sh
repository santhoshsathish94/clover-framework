#!/bin/bash
# md5 of everything the campaign depends on (run on the box); run before and after the campaign and compare.
cd ~/rustmig/engcfg || exit 1
md5sum src/lib.rs src/tests.rs include/engcfg.h Cargo.toml Cargo.lock verify/diff_driver.cpp verify/diff_driver verify/build.sh verify/run.sh verify/all.sh \
  ref/engcfg_ref.cpp ref/ecref.h ref/libengcfg_ref.so target/release/libengcfg.so
