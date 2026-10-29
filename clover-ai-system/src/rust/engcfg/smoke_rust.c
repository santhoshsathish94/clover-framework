/* Smoke test of the Rust library: the same hand-derived scenario as ref/smoke_ref.c (included below), linked against target/release/libengcfg.so.
   Build and run (box, in ~/rustmig/engcfg):  gcc -O1 -std=gnu11 -Wall -Wextra -I include -o smoke_rust smoke_rust.c -L target/release -lengcfg -Wl,-rpath,$PWD/target/release && ./smoke_rust */
#include "ref/smoke_ref.c"
