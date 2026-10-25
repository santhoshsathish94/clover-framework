#!/bin/bash
# usage: run.sh NAME [sy_driver run options...]   (REF / RUST / JOBS may be set in the environment; RUST is where a later agent points a mutated build)
# Runs every scenario against the two supply libraries; results in out/NAME/ (summary.txt, units.log, <scenario>/u<seed>.*, data/ the files the drive reads come from).
cd "$(dirname "$0")"
R=$HOME/rustmig
REF=${REF:-$R/supply/ref/libsupply_ref.so}
RUST=${RUST:-$R/supply/target/release/libsupply.so}
OUT=out/$1; shift
nice -n 10 ./sy_driver run --ref "$REF" --rust "$RUST" --sc "$R/slotcache/target/release/libslotcache.so" --out "$OUT" --jobs "${JOBS:-20}" "$@"
