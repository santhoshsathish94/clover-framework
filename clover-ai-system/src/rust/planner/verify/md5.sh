#!/bin/bash
# md5 of everything the verification depends on: the libraries, the contract, the crate and ref sources (read-only for this work) and the driver's own files.
cd "$(dirname "$0")"
R=$HOME/rustmig
md5sum $R/planner/ref/libplanner_ref.so $R/planner/target/release/libplanner.so $R/slotcache/target/release/libslotcache.so $R/routetable/target/release/libroutetable.so \
       $R/planner/include/planner.h $R/planner/Cargo.toml $R/planner/src/*.rs $R/planner/ref/planner_ref.cpp $R/planner/ref/planner_extracted.inc \
       $HOME/k3gpu/results/p1d.trace pl_driver.cpp build.sh run.sh asan.sh md5.sh count.sh pl_driver 2>&1
