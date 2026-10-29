# copyq (Rust) — notes

Implements `include/copyq.h` (not modified). Behaviour reference: `pf_enqueue` in `src/code/gpu/backend-gpu.cu`. The header says lines 61-129; in the file as read on 2026-10-21 the function is at lines 76-129 (the header's start line is about 15 early). Line numbers below are from that reading. `decode.cuh`: `act_gate` (line ~392), `g_slab[93]`, `g_prefill`.

Layout: `src/abi.rs` (repr(C) structs), `src/engine.rs` (queues, workers, `Backend` trait, unit tests with a stub backend), `src/cuda.rs` (real backend: cudarc raw `result`/`sys` driver calls + pread), `src/lib.rs` (`cq_init`, `cq_enqueue`, layout test). `smoke_rust.c` is the C smoke program.

## Behaviour map (original -> Rust)

| Original (backend-gpu.cu) | Behaviour | Rust |
|---|---|---|
| 78 `qi = kind==0 ? MAX_CARDS : dev` | queue 8 for kind 0, else queue `dev` (any kind != 0) | engine.rs:110 |
| 80-82 lock, `if (!Q.started)` | first job of a queue starts its threads, under the queue lock | engine.rs:115-117 |
| 83 `nthreads` | queue 8: `readers`; card queue: `max(2, readers / max(1, cards))` (C int division) | engine.rs:104-106 (`threads_for`) |
| 84 detached `std::thread` per worker | std thread, handle dropped, never stopped | engine.rs:118-123 |
| 85 per-thread `bounce`, `stream[MAX_CARDS]` | per-thread `CudaWs { stream[8], bounce }`, created inside the thread | cuda.rs:20-24, 43-45 |
| 88 wait on condvar, pop front | FIFO `VecDeque`, `Condvar` wait while empty | engine.rs:129-143 |
| 128-129 push_back + notify_one under lock | same | engine.rs:125-126 |
| 89-98 kind 0: pread loop to EXPERT_RAW, skipped when prefill, `p->state=1` release | `read_full` loop (partial reads continue at `off+got`), skip if prefill, `store1(p_state)` | engine.rs:188-193, 146-160 |
| 93 error text `read-ahead: pread failed: <strerror|end of file>` then `exit(1)` | same text, exit(1) | engine.rs:146-160 |
| 99 `cudaSetDevice` | `ctx.bind_to_thread()` of the primary context (retained once per device) | cuda.rs:58-71 |
| 100 stream per (thread, dev), non-blocking, lazy | `result::stream::create(StreamKind::NonBlocking)` lazily | cuda.rs:68-70 |
| 102-104 victim: async D2H, sync, then `wb->state=1` | `d2h` (async + stream sync), then `store1(wb_state)`, before any upload | engine.rs:196-200, cuda.rs:74-80 |
| 106-120 kind 1: prefill -> slab source; else lazy pinned bounce (`cudaHostAlloc` default flags), pread loop (text `card read-ahead: pread failed: ...`), then `act_gate`, async H2D | same order: bounce alloc after the victim copy, only when not prefill | engine.rs:201-210, cuda.rs:82-89 |
| 108, 121 `g_slab[layer] + ((off/EXPERT_RAW)&3)*EXPERT_RAW` | same arithmetic (signed i64 division, `&3`), slab array read at job time (late-bound, as `g_slab` is filled after start-up) | engine.rs:175-184 |
| 121 kind != 1: `act_gate`, then source (slab or `host_src`) evaluated after the gate | same order | engine.rs:211-214 |
| 119/121 `act_gate`: yield loop while `g_act_hold` != 0 (acquire load) | `thread::yield_now()` loop on `act_hold`, acquire load | engine.rs:163-173 |
| 122 `cudaStreamSynchronize` | `sync` | engine.rs:216, cuda.rs:96-99 |
| 123-124 `rel` (if set) then `arr`, release stores | `store1(rel_state)` (skipped if NULL) then `store1(arr_state)`, release stores | engine.rs:217-218, 79-86 |
| queues never destroyed, workers never stopped | engine is in a `static OnceLock`, workers loop forever | lib.rs:10, engine.rs:129 |

## `unsafe` blocks (each has a one-line SAFETY comment in the source)

| Where | Argument |
|---|---|
| engine.rs:33 `unsafe impl Send for Job` | The header makes the caller keep every pointer in a job valid until its flag is set; pointers are only dereferenced by the worker that owns the job. |
| engine.rs:85 `AtomicI32::from_ptr` (stores) | `*_state` is the address of a live, aligned 4-byte atomic int owned by the caller (header). |
| engine.rs:168 `AtomicI32::from_ptr` (act_hold load) | `act_hold` is the address of a live atomic int for the whole run (header). |
| engine.rs:180 read `slab[layer]` | `slab` points to 93 pointers (header); `layer` is range-checked first (outside the range: message + exit(1)). |
| cuda.rs:54 `pread` | The engine passes a writable buffer of at least `len` bytes (slot, or the EXPERT_RAW-byte bounce buffer). |
| cuda.rs:77 `cuMemcpyDtoHAsync_v2` | `dev_ptr` is a live device allocation of n bytes on `dev`, `host` writable for n bytes (caller contract); context current (set_device). |
| cuda.rs:79, 98 `stream::synchronize` | The stream was created by this thread in `set_device` and is never destroyed. |
| cuda.rs:85 `malloc_host` | Plain pinned allocation, context current; never freed (as the original). |
| cuda.rs:93 `cuMemcpyHtoDAsync_v2` | `dst` live device allocation of n bytes on `dev`, `src` readable for n bytes until the following synchronise (caller contract). |
| lib.rs:21, 34 deref of `cfg` / `job` | Non-null checked; validity is the caller's contract. |
| engine.rs tests (246, 264, 275, 289) | Test-owned buffers/atomics that outlive the job. |

No `unwrap()` on CUDA/IO results. Poisoned mutexes are recovered with `into_inner` (no code path panics while holding a queue lock).

## Differences from the original / not reproduced exactly

- Error text of CUDA failures: original `file:line: cudaGetErrorString`; Rust `engine.rs:line: <cudarc DriverError Debug name>` (e.g. `CUDA_ERROR_INVALID_VALUE`). Both go to stderr and exit(1). `pread` messages are identical in form.
- `cudaSetDevice` is replaced by `bind_to_thread` of the device's primary context (retained once per device, shared by all workers). Same context as the C++ runtime uses; not compared with `cudaSetDevice` in any other way.
- Added guards (the original would crash or index out of bounds): `dev` outside 0..7 for kind != 0, `layer` outside 0..92 with prefill, `expert_raw == 0`, NULL `cfg`/`job`, `cq_enqueue` before `cq_init` -> message + exit(1). NULL `p_state`, `wb_state`, `arr_state` are skipped instead of dereferenced (the original dereferences them; `rel_state` NULL is skipped in both). NULL `act_hold` means no gate.
- A second `cq_init` is ignored (first configuration kept); the original has plain globals.
- Kind values other than 0 and 1 take the kind-2 path (as the original's `else`).
- cudarc event tracking: `disable_event_tracking` is not called, because the safe cudarc types (`CudaSlice`, `CudaStream`, events) are not used at all; only raw `result::`/`sys::` calls. Nothing is tracked.
- `off_t` is mirrored as `i64`: correct on LP64 Linux (checked by the layout test), not portable to 32-bit. Linux only (`pread` is declared directly).
- `unsafe extern "C"` functions: a Rust panic would abort the process; no known path panics.

## What was and was not verified (2026-10-21, box `gpu`, cargo 1.99.1)

- Verified: `cargo build --release` (0 warnings); `cargo test --release`: 9 unit tests pass, including the struct-layout test, which compiles `include/copyq.h` with `cc` and compares size and every field offset of both structs with `offset_of!`.
- Verified: `smoke_rust` on card 7 (kind 0, 1 with victim, 2 with victim and NULL rel, act_hold gate) and an error-path run (`pread` on fd -1 -> message, exit status 1).
- Not verified: flag ordering under weak memory (stores are `Release`, the stub test only checks the logical order; x86 hides reordering); multi-card use, many concurrent jobs, prefill on the real backend, jobs from several threads at once, long runs; no negative control of the layout test or the smoke compare (the smoke compare cannot pass vacuously, since the card slot starts with a third pattern); no comparison against the C++ reference library's output (another agent's task); no performance measurement.
