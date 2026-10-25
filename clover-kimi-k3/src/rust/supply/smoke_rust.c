/* The smoke test of the Rust library: the same hand-derived scenarios as ref/smoke_ref.c (one source, so the two libraries are held to identical expectations), linked against
   target/release/libsupply.so instead of ref/libsupply_ref.so.
   Build and run on the box, in ~/rustmig/supply (after cargo build --release):
     gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lsupply -Wl,-rpath,$PWD/target/release && ./smoke_rust */
#include "ref/smoke_ref.c"
