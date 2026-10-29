#!/bin/bash
# Builds smoke_rust (ref/smoke_ref.c against target/release/libloader.so) and runs all modes. Run on the box in ~/rustmig/loader after `cargo build --release`.
set -e
cd "$(dirname "$0")"
nice -n 10 gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -o smoke_rust smoke_rust.c -L target/release -lloader -Wl,-rpath,"$PWD/target/release"
ref/run_smoke.sh ./smoke_rust
