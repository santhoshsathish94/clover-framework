# loader (Rust) - notes

Implements `include/loader.h` (fixed contract, not edited by this crate). Behaviour reference: `struct Loader`, `g_loader[]`, `pending_mu()`, `pending_cv()`, `g_pending`, `loader_submit`, `loader_wait` (backend-gpu.cu 538-572) and `open_layer` (574-610) **as committed at 4d8d891** (the blobs are LF; the Windows working tree is CRLF and was not read); `CHECK` kernels.cuh:17, `g_slab` decode.cuh:504, `g_prefill` decode.cuh:500. The header's line numbers are those of the blob.

Layout: `src/queue.rs` (the per-card queues, the pending counter, `Loader`), `src/job.rs` (the body of an `ld_open_layer` job: callback order, path, opens, slab reads; returns the fatal message instead of exiting), `src/sys.rs` (`open`, `pread`, `write`, `exit` declared by hand, the `Sys` trait, `fatal_msg`), `src/abi.rs` (`repr(C)` mirrors of the header), `src/lib.rs` (the four C entry points, guards, process-wide state), `src/tests.rs` (unit tests), `smoke_rust.c` / `smoke_rust.sh` (the shared smoke test of `ref/smoke_ref.c` against `libloader.so`). No dependencies; `panic = "abort"` in release; crate-type cdylib + rlib.

## Design

- `Loader` = 8 `CardQ { Mutex<{started, VecDeque<Job>}>, Condvar }` + `pending: Mutex<usize>` + `Condvar` + the thread-start hook. Created once (`Box::leak`) and kept in a `static OnceLock` together with the configuration; nothing is ever dropped (the original never destroys `g_loader[]` or its threads).
- `submit`: `pending += 1` under the pending lock, then the card lock; the first job of a card starts that card's thread **while the card lock is held**, then the job is pushed and the condvar notified (`notify_one`), the same sequence as the original. The original's unlocked `if (!g_loader[card]) g_loader[card] = new Loader` is not reproduced: the card state is created up front, so concurrent first submits are safe (header, difference 1).
- Worker thread: calls the start hook (`cb.set_device(card)`) first, then loops: wait for a non-empty queue, pop the front, run it with no lock held, `pending -= 1` under the pending lock, `notify_all()` after releasing it. Detached (the `JoinHandle` is dropped), 8 MB stack (`std::thread`'s default in the original is the system's, 8 MB with the usual ulimit).
- `wait`: lock `pending`, wait on the condvar while it is not 0 (loop, so spurious wakeups are handled).
- `ld_open_layer`: validates, writes the bookkeeping (`*h_L`, `*h_card`, `*h_k_in`, `handles[layer]` if `layer <= 92`) on the calling thread, copies the dataset string (`CString`), builds a `LayerJob` and submits a closure that runs `job::run` and, on `Err(message)`, calls `fatal_msg` (stderr write, then C `exit(1)`).
- `job::run` calls, in the original's order: `set_device`, `ensure_work`, `load_trunk`, (`layer >= 1`) `load_expert_tables`, `alloc_state`; then for `layer >= 1`: path (`snprintf` semantics: cut to 4095 bytes), `open(path, O_RDONLY | O_DIRECT)` with `*h_fd` written after each open, the plain `open` on any failure of the first, the `cannot open` message when both fail; with prefill: `host_alloc(&slab[layer], 4 * expert_raw)` then 4 records, each a `pread` loop continuing at `got` until `expert_raw` bytes (`rd <= 0` -> message). `4 * expert_raw`, `r * expert_raw + got` and the `off_t` are computed with wrapping arithmetic like the C (`size_t` / `off_t`); with the engine's values nothing wraps.
- `open`, `pread`, `write`, `exit` are declared in `sys.rs` (`extern "C"`), so the dynamic linker resolves them like the C++ reference's calls (the differential driver relies on this to interpose `open` and `pread` for both libraries). `exit` is the C library's `exit`, not Rust's `process::exit`.
- Process-wide state is a `OnceLock<State>`; a second `ld_init` is ignored, also an invalid one (the reference checks `cfg == NULL` first, then returns if already initialised, then validates).

## Behaviour map (original -> Rust)

| Original (backend-gpu.cu at 4d8d891) | Behaviour | Rust |
|---|---|---|
| 538-540 `struct Loader`, 541 `g_loader[MAX_CARDS]` | one queue per card 0..7, never destroyed | `queue.rs` `Loader`, `CardQ`, `Box::leak` |
| 542-544 `pending_mu()`, `pending_cv()`, `g_pending` | process-wide counter, mutex, condvar (leaked) | `Loader.pending`, `Loader.pending_cv` |
| 547 `if (!g_loader[card]) ... new Loader` | queue created lazily, unlocked | created up front; first-submit thread start under the card lock |
| 549 `g_pending++` under `pending_mu` | before the card lock and before the push | `submit`, first statement |
| 551-563 `if (!l.started)` thread start, under `l.mu` | the first job of a card starts its thread | `submit`, under the card lock |
| 554 `CHECK(cudaSetDevice(card))` as the thread's first action | before waiting for a job | `worker`: `(self.on_start)(card)` -> `cb.set_device` |
| 555-558 pop front under `l.mu`, `j()` outside it | FIFO per card, one at a time | `worker` loop |
| 559-560 `g_pending--` under the lock, `notify_all()` after | after the job returned | `worker`: `*lock(&self.pending) -= 1; notify_all()` |
| 563 `l.th.detach()` | never joined | `JoinHandle` dropped |
| 565-566 `push_back`, `notify_one()` | after the thread start | `submit` |
| 568-572 `loader_wait` | wait while `g_pending != 0` | `Loader::wait` |
| 576 `configure()` | engine configuration | not part of the loader (caller) |
| 578-579 `h->L = (int)L; h->card = card_of(L); h->k_in = (int)k;` | on the calling thread | `ld_open_layer`: `put(h_L)`, `put(h_card)`, `put(h_k_in)` (card given by the caller) |
| 579 `if (L <= (unsigned)N_LAYERS) g_handles[L] = h;` | table store | `handles[layer] = handle` when `layer <= 92` |
| 580 `const std::string ds(dataset)` | copy | `CStr::to_owned()` |
| 581 `loader_submit(h->card, lambda)` | job on the card's thread | `st.loader.submit(card, closure)` |
| 582 `CHECK(cudaSetDevice(h->card))` | job start | `cb.set_device(card)` |
| 583 `ensure_work(h->card)` | | `cb.ensure_work(card)` |
| 584-585 `layer_load_trunk`, `if (L >= 1) layer_load_expert_tables` | | `cb.load_trunk`, `cb.load_expert_tables` |
| 586-590 the state group | | `cb.alloc_state(handle, positions)` |
| 592-595 path, two opens, message, `exit(1)` | `snprintf` into char[4096]; first open O_DIRECT | `job::run`; `expert_path`, `sys.open` x2, `Err(cannot open ...)` |
| 597 `CHECK(cudaHostAlloc(&g_slab[L], 4 * EXPERT_RAW))` | | `cb.host_alloc(&slab[layer], 4 * raw)` |
| 598-606 the record loop, message at 602, `exit(1)` | `rd <= 0` fatal, `got += rd` | `job::run` loop, `Err(prefill: ...)` |
| `exit(1)` | C exit | `fatal_msg` -> `exit(1)` |

## `unsafe` blocks (each has a one-line SAFETY comment in the source)

| Where | Argument |
|---|---|
| `abi.rs` | none (types only) |
| `queue.rs` | none: only std synchronisation; `Loader::new` leaks a box (`&'static`) |
| `sys.rs` `write`, `exit`, `open`, `pread` | libc calls on fd 2 with a live buffer; `open` with a NUL-terminated path and no mode argument; `pread` into a buffer the header says is writable for `len` bytes |
| `job.rs` `unsafe impl Send for Cb, LayerJob` | the header makes the callbacks, `ctx`, tables and `*h_fd` valid for the process and callable from loader threads |
| `job.rs` callback calls | function pointers validated non-NULL by `ld_init`; `handle` and `dataset` as the header says |
| `job.rs` `*p = v` (fd), `slab.add(layer)`, `*slab.add(layer)` | `h_fd` is the caller's int; `layer <= 92` checked at the call; `host_alloc` stored the block |
| `lib.rs` `unsafe extern "C"` entry points, `&*cfg`, `&*req`, `CStr::from_ptr`, the `handles` store, `put()` | non-NULL checked first; validity is the header's contract; `layer <= 92` for `handles` |
| `lib.rs` `unsafe impl Send for Arg` | the caller's job argument, used only by the card thread |

No `unwrap()` on I/O results. Poisoned locks are recovered with `into_inner` (no code path panics while a lock is held: jobs and the hook run without one).

## Choices where the header or the original is silent

- **Check order** of the guards (the reference has the same): `ld_init`: `cfg` NULL, already initialised (return), then `set_device`, `ensure_work`, `load_trunk`, `load_expert_tables`, `alloc_state`, `handles`, with prefill `host_alloc` then `slab`. `ld_submit`: init, `fn`, card. `ld_open_layer`: init, `req`, `dataset`, card, slab slot. The slab-slot check happens at the call, before any bookkeeping (the original would write out of bounds later, in the job).
- `ld_wait` before `ld_init` returns at once (nothing can be pending).
- A failing `pread` is `rd <= 0` for both end of file and error, as the original; `EINTR` is not retried.
- `*h_fd` receives the result of the first open, then, if that was negative, the result of the second (the original assigns `h->fd` twice); when both fail it holds -1 when the process exits.
- If `host_alloc` returns without storing a pointer, `slab[layer]` stays what it was; the reads then target that pointer (NULL -> `pread` fails with EFAULT -> the `prefill: cannot read` message). The reference behaves the same (the text is the same); not tested.
- `expert_raw == 0` is accepted (zero-byte block, no reads); not rejected as in `copyq`'s `cq_init`.
- A path containing an interior NUL cannot occur (the dataset is a C string); `job::run` still returns an error instead of unwrapping.
- Thread creation failure: message `loader: cannot start the thread of card <n>: <reason>` and `exit(1)` (the original's `std::thread` constructor would throw -> `terminate`). Not testable here.
- Thread-start `set_device` and the job's `set_device` are two calls of the same callback with the same card (the original's two `cudaSetDevice` calls); the thread-start call is made from the loader thread before any job, even if the first job is a plain `ld_submit` job.
- `ld_submit` with a job that calls `ld_wait` deadlocks (same as the original); not checked.

## Tests (`cargo test`, 32 tests, expectations derived by hand from the original and the header; recording callbacks and a fake libc)

Job: layer 0 order and no open / no fd write; layer 3 without prefill (callback order, first open `O_RDONLY | O_DIRECT`, fd stored, no read, no slab); direct open refused -> plain open; both opens fail -> exact message and `*h_fd == -1`; fd 0 counts as success; path text, `%u` of 4294967295, cut at 4095 bytes (ds of 4070, 4071, 4080, 5000 bytes) and the cut path in the message; prefill: 4 records, one `pread` each (offsets and lengths), short reads of 3 bytes (3+3+3+1 per record, offsets `r*10+g`), end of file in record 2 (the exact call list and message), empty file, read error without retry, layer text `92`, `expert_raw == 0`, the engine's `expert_raw` 17547264 (offsets 0, 17547264, 35094528, 52641792, block 70189056). Guards: texts of the card and slab-slot checks; `validate_config` rejects each missing item in the header's order, accepts missing `host_alloc`/`slab` without prefill. Queue: no thread before the first submit and one start hook per card, hook runs before the first job on the job's thread, FIFO on one thread (200 jobs), 8 cards -> 8 distinct threads, 4 cards on a spin barrier (concurrency), no overlap of one card's jobs (50 jobs), wait blocks until the job finished and returns with the count 0, a job submitted from another thread during the wait is waited for, a job that submits jobs (same card child runs after the parent returned), 4 waiters all return, 50 rounds of 4 threads making the first submits to one card (one hook, per-submitter order kept). C ABI: `struct_layout_and_o_direct_match_the_c_header` compiles `include/loader.h` with `cc` and compares size and every field offset of the three structs, `O_DIRECT` and `sizeof(off_t)`; `exported_functions_end_to_end` (ld_wait before init, ld_init twice, ld_submit, ld_open_layer on a real file: bookkeeping, handles table, callbacks, fd readable at an aligned offset); `fatal_message_goes_to_stderr_and_exits_with_status_1` re-runs the test binary as a child and checks status 1 and the exact stderr bytes.

## Smoke (`smoke_rust.c`)

`smoke_rust.c` includes `ref/smoke_ref.c` (one source for both implementations) and is linked against `libloader.so` by `smoke_rust.sh`; six modes (see ref/NOTES.md); PASS in all six.

## Differential driver (`verify/`, details and raw lines in `verify/RESULTS.md`)

2820 units (scenario x seed) against the reference and against this crate, 2820 / 2820 PASS on both, canonical output identical in all 2820; a 14100-unit soak the same (14100 / 14100). The reference's unlocked first submit shows violations in 339 of 1600 (matrix) and 1686 of 8000 (soak) concurrent-first-submit trials; this crate 0 of 1600 and 0 of 8000. One instrument flake (thread count read too early) was found by the matrix and fixed in the driver. Final artefacts on the box (`~/rustmig/loader`): `target/release/libloader.so` md5 `2f1d3eacf8ebc2e0078b61f8ab629f21` (built by `cargo build --release` in `verify/build.sh` from the sources whose md5 are listed in `verify/final.log`), `ref/libloader_ref.so` md5 `bb41075c183f95a9d9d800a0a377477b`, `include/loader.h` md5 `af286f75ad09c319d2e748d317cda7be`. The `.rs`, `.h`, `.toml` and `.inc` files are identical on Windows and on the box (md5 compared: header, `abi.rs`, `job.rs`, `lib.rs`, `queue.rs`, `sys.rs`, `tests.rs`, `loader_a.inc`, `loader_b.inc`); `verify/sync.ps1` strips CRs from the `.sh`, `.c`, `.cpp` and `.py` files on the box, so their md5 there differ from the Windows files (`verify/final.log` lists the box md5).

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/loader`)

    cargo build --release        # target/release/libloader.so  (cargo test does NOT rebuild it)
    cargo test                   # or --release
    ./smoke_rust.sh              # gcc ... -L target/release -lloader, then ref/run_smoke.sh ./smoke_rust
    cd verify && ./build.sh && ./run.sh 200 out-final      # differential driver vs ref/ (see verify/RESULTS.md)

(from Windows: `pwsh verify\sync.ps1` copies the crate to the box; `python ref\extract.py` regenerates the `.inc` files from `git show 4d8d891:`.)

## Not covered

See the end of `verify/RESULTS.md` for the list of what the driver checks and cannot check; in short: thread interleavings are not deterministic (properties checked, not schedules), no memory-order checking tool (Miri, TSan, loom) was run, no CUDA and no engine run (the GPUs were in use by another agent: the engine with `K3GPU_LOADER=rust` was not and could not be tried), only x86_64 Linux was built, thread-creation failure and `write` failure on stderr are not exercised, `O_DIRECT` with an `expert_raw` that is not a multiple of the device's block size is not exercised, no performance measurement.
