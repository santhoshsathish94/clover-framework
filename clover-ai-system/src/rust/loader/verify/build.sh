#!/bin/bash
# Builds the differential driver (ld_driver) and, if needed, both libraries. Run on the box in ~/rustmig/loader/verify.
# -D_FORTIFY_SOURCE=0 and -rdynamic: the driver defines open/pread itself and the libraries must resolve to them.
set -e
cd "$(dirname "$0")"
export PATH=$HOME/.cargo/bin:$PATH
(cd .. && nice -n 10 cargo build --release 2>&1 | tail -3)
(cd ../ref && ./build.sh >/dev/null)
nice -n 10 g++ -O1 -std=c++17 -D_FORTIFY_SOURCE=0 -Wall -Wextra -pthread -rdynamic -I../include -o ld_driver ld_driver.cpp -ldl
md5sum ld_driver ../ref/libloader_ref.so ../target/release/libloader.so
