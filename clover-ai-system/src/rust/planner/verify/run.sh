#!/bin/bash
# usage: run.sh NAME [pl_driver run options...]   (REF / RUST / JOBS may be set in the environment; RUST is where a later agent points a mutated build)
# Runs every scenario against the two planner libraries; results in out/NAME/ (summary.txt, units.log, <scenario>/u<seed>.*).
cd "$(dirname "$0")"
R=$HOME/rustmig
REF=${REF:-$R/planner/ref/libplanner_ref.so}
RUST=${RUST:-$R/planner/target/release/libplanner.so}
OUT=out/$1; shift
./pl_driver run --ref "$REF" --rust "$RUST" --sc "$R/slotcache/target/release/libslotcache.so" --rt "$R/routetable/target/release/libroutetable.so" --out "$OUT" --jobs "${JOBS:-40}" "$@"
