#!/bin/bash
# usage: run.sh NAME [pi_driver run options...]   (REF / RUST / JOBS may be set in the environment; RUST is where a later agent points a mutated build)
# Runs every scenario against the two libraries; results in out/NAME/ (summary.txt, units.log, <scenario>/u<seed>.*). nice -n 10.
cd "$(dirname "$0")"
R=$HOME/rustmig
REF=${REF:-$R/poolinit/ref/libpoolinit_ref.so}
RUST=${RUST:-$R/poolinit/target/release/libpoolinit.so}
OUT=out/$1; shift
nice -n 10 ./pi_driver run --ref "$REF" --rust "$RUST" --sc "$R/slotcache/target/release/libslotcache.so" --out "$OUT" --jobs "${JOBS:-16}" "$@"
