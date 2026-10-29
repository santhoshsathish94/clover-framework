#!/bin/bash
# Rebuilds everything on the box and re-runs everything: reference library + smoke, Rust library + unit tests + smoke, the driver, and a differential run.
# usage (box, ~/rustmig/poolinit/verify): ./final.sh NAME [MULT]     MULT scales the default unit counts (H 600, HX 400, P 600, PX 800, PR 200, M 400); default 1.
# Prints md5 of both libraries and the run's summary lines. nice -n 10, JOBS defaults to 16. Output of the run: out/NAME/ (units.log, summary.txt, kept raw logs of units 1 and the failing ones).
set -e
export PATH=$HOME/.cargo/bin:$PATH
cd "$(dirname "$0")/.."
NAME=${1:?name} ; MULT=${2:-1}
( cd ref && nice -n 10 ./build.sh >/dev/null && ./smoke_ref | tail -1 )
nice -n 10 cargo build --release 2>&1 | tail -2
nice -n 10 cargo test 2>&1 | grep -E "^test result|FAILED|panicked" | head
gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lpoolinit -Wl,-rpath,"$PWD/target/release" && ./smoke_rust | tail -1
md5sum target/release/libpoolinit.so ref/libpoolinit_ref.so
cd verify && nice -n 10 ./build.sh >/dev/null
./run.sh "$NAME" --keep-raw 2 --n "H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))" | grep -v '^\['
# driver v2: scenario A (the library's refusals of NULL pointers / NULL ops / expert_raw 0) is not in the default list
./run.sh "$NAME-A" --keep-raw 2 --scn A --n A=60 | grep -v '^\['
