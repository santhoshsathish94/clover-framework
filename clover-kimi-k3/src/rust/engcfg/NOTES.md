# engcfg (Rust) — notes

Implements `include/engcfg.h` (fixed contract; not edited after the reference was written). Behaviour reference: `backend-gpu.cu` and `decode.cuh` **as committed at 4d8d891** (`git show`; the working tree was not read). Lines quoted below are lines of that blob.

Layout: `src/lib.rs` (implementation and C ABI), `src/tests.rs` (unit tests, `#[cfg(test)]`), `smoke_rust.c` (includes `ref/smoke_ref.c`, the same hand-derived scenario for both libraries), `ref/` (C++ reference built from the engine's own text, see `ref/NOTES.md`), `verify/` (differential driver, see `verify/NOTES.md` and `verify/RESULTS.md`). Crate `engcfg`, `cdylib` + `rlib`, release profile `panic = "abort"`, no dependencies; the `.so` exports exactly the 13 `ec_*` functions of the header and needs glibc >= 2.34 symbols (`objdump -T`).

## What it is

The small pure-CPU parts of the GPU backend's start-up and exit code, split out of `configure()` / `print_summary()` / `card_of()` / `force_routes()` so that everything touching CUDA, dlopen or the loaders stays in C++ and enters as an argument. The call order the C++ adapter must keep is in the header comment (it is also what `verify` scenario E7 runs against the original `configure()` text).

## Design

- **Environment.** A trait `EnvSource { get(&CStr) -> *const c_char }`; the C ABI uses libc `getenv` (nothing is cached), the unit tests use a map. `routecache` in `ec_settings` is the pointer `getenv` returned, not a copy.
- **`atoi`.** Re-implemented byte-wise as glibc's `(int)strtol(s, NULL, 10)`: C-locale whitespace (space `\t \n \v \f \r`), one sign, decimal digits, saturation at `i64::MAX` / `i64::MIN` (including `-9223372036854775808` exactly), no digits gives 0, then `as i32` (low 32 bits). Not delegated to libc so that the logic is visible and unit-tested; the differential driver compares it with glibc's own `atoi` (the reference) on millions of generated strings.
- **Integer widening.** `(size_t)env_int(..) << 20` / `<< 30` and `(size_t)env_int(..)` for `vq_cap` are `v as i64 as usize` (sign extension) followed by a shift whose lost bits are discarded; `vq_from` is `i32 -> c_long`. `pend_cap = cap_slots.wrapping_mul(6) / 10`.
- **Messages.** Each line is built as bytes (`Line`): literal pieces, decimal integers (`Display`, equal to `%d %zu %lu`), `%s` of a C string as raw bytes (NULL prints `(null)` like glibc), and doubles through **libc `snprintf`** (`%.0f %.1f %.3f`, called twice: once to measure, once to fill). Reason for libc instead of a Rust formatter: the original uses printf, so the same function gives the same digits, tie rounding (round-half-even on the exact binary value), `inf` / `nan` / `-nan` spelling, negative zero and the all-digits expansion of huge values by construction; no independent formatter had to be proven equal. One `write_all` on `std::io::stderr()` per original `fprintf` line (stderr is unbuffered in both; write errors are ignored; Rust's std additionally treats EBADF as success, which has the same visible effect).
- **Fatal paths.** Message written, then `std::process::exit(1)` (libc `exit`: atexit handlers and stdio flushing run, as the original's `exit(1)`). The trace message and the selector messages carry the value's raw bytes.
- **`ec_open_trace`.** Reads `K3GPU_TRACE`, `fopen(value, "w")` through libc, returns the `FILE *`; on failure the message and `exit(1)`. The stream is the caller's (the original keeps it open for the whole run).
- **`force_routes`.** The once-only warning flag is a process-global `AtomicBool` (`swap(true)`: the first caller prints). The dealing loop reproduces the original's unchecked write `sel[(size_t)t * top + fill[t]++]` with wrapping `usize` arithmetic and `sel.wrapping_add(idx)`; `fill` is a `Vec<i32>` of `n` zeros (n < 0 aborts, see below). `total` is `i64`, `(long)n * top` is `i64 * i64`.
- **`card_of`.** `layout == 1` selects the interleaved formula, any other value the grouped one. `wrapping_mul` / `wrapping_sub` where the original's `int` arithmetic would overflow. The two inputs on which the original traps (`% 0`, `INT_MIN % -1`) abort the process explicitly.
- **`print_summary`.** `summary_lines` returns the lines; `ec_print_summary` writes them. `u64` sums use `wrapping_add` (the original's `unsigned long`), the floating-point expressions keep the original's order: `100.0 * hc / need`, `dr * (double)EXPERT_RAW / 1e9`, `1000 * t / n`. The summation of `t_read` / `t_copy` runs over the ready cards in card order. `n_calls[i] != 0.0` is the C truth test (true for NaN, false for +0 and -0).
- **Not thread-safe** except the warning flag, like the original (the main thread is the only caller). Panics abort (release profile); there are no `unwrap`s on external input.

## Behaviour map (original -> Rust)

| Original (4d8d891) | Rust |
|---|---|
| `env_int` (backend-gpu.cu:50) | `env_int`, `atoi`, `strtol10`; `ec_env_int` |
| `configure()` 492-495, 497, 499, 501, 514-515 (cards, readers, verbose, reserve, policy, chunks, host, layout, prefill) | `settings_from`; `ec_settings_from_env` |
| `configure()` 502-511 (route cache: path, look-ahead, `g_vq`, `g_vq_cap`, `g_vq_from`) | `settings_from`, the `if !unset_or_empty(rc)` block |
| `configure()` 506 `g_pend_cap = g_host.cap_slots * 6 / 10` | `ec_pend_cap` |
| `configure()` 500 trace | `ec_open_trace`, `trace_error_line` |
| `configure()` 512 route cache line | `route_cache_line`; `ec_print_route_cache` |
| `configure()` 516-520 prefill check and ON line | `ec_prefill_start`, `PREFILL_NEEDS`, `PREFILL_ON` |
| `configure()` 523 start line | `start_line`; `ec_print_start_line` |
| `copyq_select` / `slotcache_select` / `routetable_select`, variable parsing (183-188, 224-228, 243-247, 258-264, 441-447, 456-462) and the announce lines (200, 256, 454) | `SELS`, `selector_decide`, `selector_cpp_line`, `selector_lib_from`, `selector_announce_line`; `ec_selector_mode`, `ec_selector_lib`, `ec_selector_announce_rust` |
| `card_of` (526) | `card_of`; `ec_card_of` |
| `force_routes` (374-387) | `force_routes`, `warn_line`; `ec_force_routes` |
| `print_summary` (465-485) | `summary_lines`; `ec_print_summary` |

## Deliberate differences from the original (all in the header)

1. **Functions instead of globals**: the values the original takes from engine globals (`g_cards`, `g_layout`, `g_rc.size()`, `g_rc_pos`, `g_host.cap_slots`, the counters) are arguments; `force_routes` receives only the one `g_rc[g_rc_pos]` entry it reads.
2. **`TOP` and `EXPERT_RAW`** (compile-time in the engine) are arguments (`top`, `expert_raw`).
3. **`K3GPU_HOST_GB`** is read once (the original evaluates `env_int` at lines 501 and 523); the start line prints the stored value.
4. **`ec_selector_mode` prints the `cpp` line itself** (as the original's select functions do) and returns the mode; for `rust` it returns 1 and prints nothing, the caller loads the library and calls `ec_selector_announce_rust`.
5. **Out of contract, both die (signals differ):** `card_of(1, 0, x)`, `card_of(1, -1, INT_MIN + 1)` (original SIGFPE, Rust SIGABRT by an explicit check); `force_routes` with n < 0 reaching the dealing (original `std::length_error` -> terminate, Rust abort without the libstdc++ message). `ec_print_summary` with `cards > 8` aborts in Rust; the original reads past its array.
6. The warning flag of `force_routes` is atomic (the original's is a plain `static bool`).

## Choices where the header or the original is silent

- **Negative counts.** A route-cache line can parse to a negative count (`3:-2`, see the routetable notes). `force_routes` then passes steps 1-3 if the others compensate, and the dealing writes more than `top` experts for one token, i.e. **past the end of `sel`** in the original. The Rust code does the same writes (the driver compares the written words, including 512 words past `n * top`). Whether to bounds-check is a Direction question (a check would change the behaviour from "writes past the buffer" to "stops"); not done.
- **`cards <= 0`.** `K3GPU_CARDS=0` or negative devices give `cards <= 0`; `ec_settings_from_env` returns it as is (the original does). `card_of` layout 0 is defined arithmetic there (`cards = 0`: layer 0 -> 0, else -1); layout 1 traps for 0.
- **`layout`** keeps any value read; only 1 means interleaved (`g_layout == 1`), `2`, `-1`, ... behave as 0.
- **`K3GPU_PREFILL`** with an empty parsed table stops the run even when `K3GPU_ROUTECACHE` is set (the original tests `g_rc.empty()`); the header says so.
- **`ec_open_trace` with a pre-existing file** truncates it (mode `"w"`).
- **The NULL `%s` cases** (`routecache == NULL` in `ec_print_route_cache`, `lib == NULL` in `ec_selector_announce_rust`) print `(null)`: glibc's behaviour, not a C-standard promise; the reference prints the same because it runs glibc's printf.
- **Locale.** Nothing sets a locale; `atoi` whitespace and the printf decimal point are those of the C locale. If the engine ever calls `setlocale`, the Rust `atoi` would differ from glibc's for locales with other whitespace; the printf part would still follow libc.
- **Floating point of the engine build.** The reference was built with plain `g++ -O2` (no `-march`, no fast-math); the engine host code is compiled by nvcc with `-O3` (`scripts/run-model-gpu.sh`); no `-ffast-math`, no FMA contraction is visible in that command, and the expressions have no multiply-add pattern, but the engine's own object was not compared.

## Tests (`cargo test`, 56 unit tests, expectations derived by hand from the original text)

`atoi_*` (whitespace set, signs, `0x10`, low-32-bit cast at both ends, saturation at LONG_MAX/LONG_MIN and the digit after it, leading zeros), `env_int_*`, settings (`defaults`, `cards_*`, `reserve_and_host_bytes_*` with hand-computed powers of two, `plain_ints`, `prefill_is_nonzero`, `route_cache_*`, `vqueue_*`), `pend_cap_*`, message texts (`route_cache_line_*` incl. non-UTF-8 bytes and NULL, `start_line_text`, `fixed_prefill_texts`, `warn_text`), selectors (`selector_built`, `selector_not_built`, `selector_lib_and_announce`), `card_of_layout_0/1` (including negative layers, the 93 boundary, wrapping, cards 0 and negative), `force_routes_*` (three hand-dealt cases, full width 16 x 5, exits leave `sel` alone, n = 0, negative count writing into the next slot), `libc_formats_halves_and_specials` (ties, near-ties, NaN, -NaN, inf, -0, 1e22, 1e300, DBL_MAX), `summary_*` (a small case by hand, three-lines-only, ready/not-ready/cards bounds, `need` wrapping to 0, route cache and queue lines, call classes with NaN/inf/-0, large values), `abi_layout` (sizes and offsets of the four structs computed by hand for LP64; `smoke_ref.c` asserts the same numbers from the C side). Child-process tests (the test binary re-run with `ENGCFG_CHILD`): `trace_open_and_failure`, `prefill_start_exit_and_line`, `selector_exit_through_the_c_abi`, `warns_once_per_process`, `out_of_contract_inputs_abort_with_sigabrt` (exit status, raw stderr bytes, SIGABRT).

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/engcfg`)

    cargo build --release                       # target/release/libengcfg.so (cargo test does NOT rebuild it)
    cargo test
    gcc -O1 -std=gnu11 -Wall -Wextra -I include -o smoke_rust smoke_rust.c -L target/release -lengcfg -Wl,-rpath,$PWD/target/release && ./smoke_rust
    verify/all.sh [OUTDIR]                      # everything, incl. the differential run (see verify/RESULTS.md)

## Not covered

- **The engine.** No GPU, no CUDA call, no engine build or run was possible or attempted; the C++ adapter that would call these functions does not exist yet. The order of calls is validated only against the original `configure()` text with stand-ins (E7), not against the real `configure()` with CUDA, the loaders and `dlopen`.
- Equality with the original on **inputs the generators do not produce** (see the lists in `verify/NOTES.md` and `verify/RESULTS.md`), other libcs than glibc 2.39 (box: Ubuntu 24.04), other locales, big-endian or 32-bit targets, threads.
- `K3GPU_ROUTECACHE` files (the loader is the routetable crate's), the real `g_host.init`, `rc_load_sel` failure paths, `dlopen` messages.
- The working-tree engine sources (being edited by someone else): if their `configure()` / `print_summary()` text has changed since 4d8d891, `ref/extract.py` fails on its pinned hashes by design; it must be re-pinned for the new commit and the differential run repeated.
- Performance (none of this is on a hot path).
