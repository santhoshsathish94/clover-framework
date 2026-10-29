# poolinit (Rust) - notes

Implements `include/poolinit.h` (fixed contract written for this component; md5 c92f900f3e2f0a52120b091ace850705, box copy with LF). Behaviour reference (read 2026-10-25): `HostCache::init` with `CHUNK_SLOTS`, `struct ExpertStats`, `enum { MAX_READERS = 32 }` and `CardPool::init` (decode.cuh lines 403-471 **as committed at 4d8d891**, not the working tree), the `CHECK` macro (kernels.cuh:17), the constants `EXPERT_RAW = 17547264` and `EXPERTS = 896` (moe.cuh:5); callers: `g_host.init((size_t)env_int("K3GPU_HOST_GB", 150) << 30)` (backend-gpu.cu:501, in `configure`) and `g_pool[h.card].init(h.card, g_reserve, g_readers)` (backend-gpu.cu:621, in `forward`, first use of a card).

Layout: `src/lib.rs` (the two exported C functions: argument checks, then the work), `src/abi.rs` (`#[repr(C)]` structs), `src/host.rs` (HostCache::init), `src/pool.rs` (CardPool::init), `src/sys.rs` (mmap, madvise, clock_gettime, exit, message helpers, the CHECK macro), `src/tests.rs`, `smoke_rust.c` (includes `ref/smoke_ref.c`: the same hand-derived program as for the reference). No dependencies; `panic = "abort"`; crate-type cdylib + rlib. No global state: every call copies `*ops` and `*cfg` at its start.

## Design

- **Ops and threads.** Every CUDA effect goes through the `pi_ops` function pointers; all of them, and every `sc_add_spare`, are called on the caller's thread, after the helper threads have been joined. The ops get the address of the out field itself (`addr_of_mut!((*out).base)`, `.copy`, `.rstream[i]`, `.bounce[i]`, `.written_back`, `.d_eptr`), as the original passes `&member`.
- **Host cache.** `plan()` = `cap_slots = bytes / raw`, `chunk_bytes = 32 * raw`, `n_chunks = cap_slots / 32` (wrapping `usize`). 16 workers: 15 helpers in `std::thread::scope` plus the calling thread, a shared `AtomicUsize` counter (`fetch_add(1, SeqCst)`), the table `base` is `Vec<AtomicUsize>`. A worker calls the real `mmap(NULL, chunk_bytes, 3, 0x22, -1, 0)` (declared `extern "C"`, so the dynamic linker resolves it and an interposer sees it), on failure writes `host expert cache: cannot map <chunk_bytes> bytes\n` and calls the C `exit(1)`, then `madvise(p, chunk_bytes, 14)` (result ignored), then `ptr::write_bytes(p, 0, chunk_bytes)`. After the scope: `set_device(0)`, `host_register(base[i], chunk_bytes, 1)` for i ascending, `sc_add_spare(cache, base[i] + s * raw)` for i descending, s from 31 down to 0; `out.cap_slots = n_chunks * 32`; the summary line with `{:.1}` formatting. The seconds come from `clock_gettime(CLOCK_MONOTONIC)` as `tv_sec + 1e-9 * tv_nsec` like `wall_now_()`.
- **Card pool.** `slots_for()` = `usable = f > reserve ? f - reserve : 0`, `(usable / raw) as i32` (wraps as the C cast); `< 64` -> the original's message and `exit(1)`; the slab `device_malloc(slots * raw)`; spares for i from slots-1 down to 0; copy stream; per reader (`0..readers`, readers = `min(nreaders, 32)`, empty for a negative count) a stream and a bounce `host_alloc(raw, 0)`; the event; `device_malloc(experts * 8)` with `experts` sign-extended like C's int -> `size_t`; `ready = 1` last.
- **CHECK.** `check(code, error_string, file, line)`: a nonzero code writes `<file>:<line>: <error_string(code)>\n` (file = `src_file` or `decode.cuh`; a NULL string from `error_string` prints `(null)` like glibc's `%s`) with one `write_all` on stderr and calls `exit(1)`. The ten line numbers are constants (428 429 454 455 459 461 463 464 466 467), asserted against the commit by `ref/extract.py`.
- **Out struct progression.** Host: `cap_slots = bytes / raw` and `ready = 1` first (as the original assigns the members first), `cap_slots = n_chunks * 32` at the end. Pool: `dev`, `readers` first; `slots` before the `< 64` test; each pointer when its op stored it; `ready` last. A process that exits in the middle therefore shows what the original object would show to an `atexit` handler. Entries of `rstream` / `bounce` at index >= readers, and `st`, are never written.
- **Messages** are written with `write_all` on `stderr()`, errors ignored, no `eprintln!` (it panics on a failed write). Texts of this crate's own checks: `poolinit: ops is NULL`, `cfg is NULL`, `out is NULL`, `ops->set_device is NULL` (and the other function names, in the order of `lib.rs`), `expert_raw is 0`; checked before anything else is done.

## Behaviour map (original -> Rust)

| Original (decode.cuh@4d8d891) | Rust |
|---|---|
| 408 `cap_slots = bytes / EXPERT_RAW; ready = true;` | `host_init`: store `cap0`, `ready = 1` into `out` |
| 409 `chunk_bytes = (size_t)CHUNK_SLOTS * EXPERT_RAW, n_chunks = cap_slots / CHUNK_SLOTS` | `host::plan` |
| 410-423 `base`, `next`, `t0`, the `touch` lambda (mmap, failure message + exit(1), madvise, memset, store) | `touch` closure in `host_init` |
| 424-427 15 `std::thread`s, then `touch()` on the caller, then `join` | `thread::scope`, 15 `spawn_scoped`, `touch()`; a spawn failure writes `poolinit: cannot start a helper thread` and aborts (the original: `std::terminate`) |
| 428 `CHECK(cudaSetDevice(0))` | `check(set_device(0), ..., 428)` |
| 429 `CHECK(cudaHostRegister(base[i], chunk_bytes, cudaHostRegisterPortable))` | `check(host_register(..., 1), ..., 429)` |
| 430-431 spares, `i` descending, `s` descending | `base.iter().rev()`, `(0..32).rev()` |
| 432-433 `cap_slots = n_chunks * CHUNK_SLOTS;` the line | `out.cap_slots`, `host::summary_line` |
| 453 `dev = device; readers = std::min(nreaders, (int)MAX_READERS);` | `out.dev`, `out.readers = min(.., 32)` |
| 454-458 set_device, mem_get_info, `usable`, `slots`, the `< 64` exit | `pool_init`, `pool::slots_for`, `pool::refusal_message` |
| 459-460 slab and spares (`i = slots; i-- > 0`) | `device_malloc`, `(0..slots).rev()` |
| 461-465 copy stream, per reader stream + `cudaHostAlloc(.., EXPERT_RAW, cudaHostAllocDefault)` | `stream_create(.., 1)`, `host_alloc(.., raw, 0)` |
| 466-467 event (`cudaEventDisableTiming`), `cudaMalloc(&d_eptr, EXPERTS * sizeof(void *))` | `event_create(.., 2)`, `device_malloc(.., experts * 8)` |
| 468 `h_eptr.assign(EXPERTS, nullptr);` | the caller's (not here) |
| 469 `ready = true;` | `out.ready = 1` |
| `ExpertStats st` | `pi_expert_stats st` in the out struct, not written |

## Choices where the header or the original is silent (the first four are decisions for the owner; see "Open questions")

1. **Source = committed blob (LF), not the working tree (CRLF).** See `ref/NOTES.md`.
2. **`__FILE__` of the CHECK message** is configuration (`src_file`, NULL = `decode.cuh`); line numbers are decode.cuh@4d8d891's.
3. **`cache` is the caller's handle**, passed to `ops.sc_add_spare` unchanged (NULL included: the Rust side does not look at it). The engine's `SlotCacheSel::add_spare` (cpp branch: `spare.push_back`; rust branch: create the cache lazily with `R()`, then `sc_add_spare`) is not reproduced; the integrator's `sc_add_spare` op must do what `SlotCacheSel::add_spare` does.
4. **`h_eptr` and `ready` of the engine's `CardPool`** stay the caller's / are copied from `out`.
5. Argument checks and NULL function pointers (messages above): additions, the original would crash.
6. `readers` negative: kept negative in `out.readers`, no reader is made (the original does the same: `min(-2, 32)`, the loop does not run). `nreaders = INT_MIN` likewise.
7. `slots` is a wrapping cast: usable / raw >= 2^31 gives a negative or small count (tested: 2^31 + 5 -> -2147483643, refused with that number printed; 2^32 + k -> k).
8. `expert_raw >= 2^31`: the original's enum is an `int` (a different arithmetic for such values); not a case that exists (17547264).
9. Helper threads' default stack: 2 MiB (Rust) against the C++ default (the `ulimit -s` size, 8 MiB typically). Not observable.
10. The helpers and the caller take chunk numbers from one counter, so which thread maps which chunk, and the address order of the chunks, are free; `base[i]` is chunk number i of the counter, as in the original. A different mapping of "chunk number" to address cannot be observed through the contract.
11. The out struct fields are written through raw pointers (never a `&mut` to the whole struct while an op can see it).
12. Memory of the chunks is never unmapped or unregistered (as the original).

## Tests (`cargo test`, 13 tests; expectations derived by hand from decode.cuh, the derivations are in the comments of `tests.rs`; fake ops record every call into one ordered log; the host memory is real with 64 KB "experts")

`pool_calls_and_out_fields` (100 slots, 3 readers: the 112 calls in order, arena addresses, spares base+99*raw down to base, streams 0x1000.., bounce slots, event, d_eptr, untouched entries and `st`), `pool_slot_count_boundaries` (63 / 64 slots, remainders, free below the reserve, the int wrap, both message texts), `pool_readers_clamped_and_negative` (40 -> 32 readers = 134 calls; -2 readers = 70 calls), `pool_with_real_constants` (17547264 / 896 / 2500 MiB / 22,000,000,000 free -> 1104 slots, slab 19372179456 bytes, 1142 calls, d_eptr 7168 bytes, through fake ops: nothing is touched), `pool_refusals_and_failed_checks_end_the_process` (child processes: 63 slots, 0 slots, the k-th CUDA call failing for k = 1..12 with the CHECK lines, events / spares / `slots` seen by an `atexit` handler, no `src_file`), `host_calls_spare_order_and_memory` (196 slots -> 6 chunks of 2 MiB, 192 spares in order, regions page aligned, non-overlapping, zero), `host_zero_chunks_and_tiny_budgets` (0 / 31 / 32*raw-1 slots-worth of bytes: only `set_device 0`; exactly one chunk), `host_plan_and_summary_with_real_constants` (150 GiB / 17547264 -> 9178 slots, 286 chunks of 561512448 bytes, 9152 slots, `160.6 GB`; plus the real engine with an empty budget through the ops), `host_failures_end_the_process` (set_device failing: line 428 with `cap_slots` 196 and `ready` 1 at the exit; the third register failing: line 429; no `src_file`), `null_arguments_and_zero_expert_raw_are_refused_first` (ten cases, each a child with the exact text), `one_decimal_formatting_equals_c_printf` (Rust `{:.1}` against the C library's `snprintf("%.1f")` for 60,000 values, including exact ties such as 0.25 and multiples of 1/8 and 1/64), `struct_layout_equals_the_c_header` (a `cc` compile of the header prints sizes and offsets of all six structs and the constants; compared with `size_of` / `offset_of!`), and `child_entry` (the helper the child tests re-run; passes trivially without its environment variable).

Run (2026-10-25): 13 passed, 0 failed. They all passed at the first run: the tests have not been shown to fail by a mutation of the crate here (the differential driver's liveness probes below exercise the same code through other paths).

## Smoke (`smoke_rust.c`)

The program of `ref/smoke_ref.c` against `libpoolinit.so`: `PASS (45 checks, 0 failed)`, the same 45 checks as for the reference.

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/poolinit`)

    cargo build --release                       # target/release/libpoolinit.so  (cargo test does NOT rebuild it)
    md5sum target/release/libpoolinit.so        # c40ad1bf36e39bd77fd4f3566fb6ad69 (final)
    cargo test
    gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lpoolinit -Wl,-rpath,$PWD/target/release && ./smoke_rust
    verify/final.sh NAME [MULT]                 # everything above for the reference too, then the differential run (see verify/RESULTS.md)

`nm -D` exports exactly `pi_host_init` and `pi_pool_init`. The build printed no warning. File md5 (box): `host.rs` 774d1c18, `pool.rs` f3694062, `lib.rs` 380b4630, `sys.rs` 42073bfa, `abi.rs` f565f6ad, `tests.rs` 117e0d2b.

## Not covered

- **Not run on a GPU or against a real CUDA runtime** (the GPUs were fully used by another job). No call to any CUDA function was made by anything in this directory. The engine was not run with this library; there is no engine integration.
- **The real `HostCache::init` with the engine's constants** (561,512,448 bytes per chunk, 150 GB): only the arithmetic (`host::plan`, the summary line, 9178 / 286 / 9152 / `160.6 GB`) and the zero-chunk case through the ops are checked; a real mapping was never made with those sizes (the rule was at most 200 MB). The path "n_chunks >= 1" is exercised with `expert_raw` between 2 KB and 128 KB. The card pool is exercised with the real constants through fake ops.
- Equality with the C++ original in the engine: the reference is the original text behind a harness with the adaptations in `ref/NOTES.md`; not the engine build.
- Timing: how long the real mmap + zeroing of 150 GB takes in Rust against C++ (no measurement; the helper count, the work split and the system calls are the same, the stack size of the helpers is not).
- Concurrency tools: no ThreadSanitizer, Miri or loom run; the only shared state of the helpers is an atomic counter and a table of atomics, written once per slot.
- Memory limits: a `base` table that cannot be allocated (`n_chunks` huge), failure of `thread::Builder::spawn_scoped` (abort), a failing `madvise` (ignored), `experts < 0`.
- `exit(1)` from a helper while other helpers still run: reproduced only by the driver's injected mmap failure (and reasoned for the real ENOMEM).
- 32-bit or non-Linux targets (`MADV_HUGEPAGE`, `MAP_ANONYMOUS`, `timespec`, `size_t` sizes are those of LP64 Linux).

## Open questions for the owner

1. Is the committed blob (LF) the intended source of truth for the extracted text, given that the Windows working-tree files are CRLF? (Earlier crates extracted the CRLF working tree. The two differ only in line ends.)
2. What `src_file` should the engine pass? If the wrapper lives in decode.cuh, `__FILE__` there is exactly the original macro's value; if it lives elsewhere, the messages differ from the original by the path.
3. Are the hard-coded CHECK line numbers (428 ...) wanted (identical messages at 4d8d891, stale after the next edit of decode.cuh), or should the messages carry the library's own file / line?
4. What is `cache` in K3GPU_SLOTCACHE=cpp mode: the integrator needs an `sc_add_spare` op that pushes into the C++ `SlotCache::spare` of `g_host.c` / `g_pool[c].c`.
5. May `out->st` stay untouched (the contract's reading of "init does not assign it"), or should init zero it?
