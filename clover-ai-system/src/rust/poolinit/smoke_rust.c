/* Smoke test of the Rust library: the same program as ref/smoke_ref.c (hand-derived expectations, fake ops, real memory for the host cache), linked against target/release/libpoolinit.so.
   Build and run (box, ~/rustmig/poolinit, after `cargo build --release`):
     gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lpoolinit -Wl,-rpath,$PWD/target/release && ./smoke_rust */
#include "ref/smoke_ref.c"
