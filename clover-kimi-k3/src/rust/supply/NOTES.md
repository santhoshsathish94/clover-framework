# supply (Rust) - notes

Implements `include/supply.h` (fixed contract; md5 in `verify/RESULTS.md`). Behaviour reference, read 2026-10-25 from the engine **as committed at 4d8d891** (`git show 4d8d891:clover-kimi-k3/src/code/gpu/decode.cuh`; the working tree was not read): `pool_ensure` (decode.cuh:536-620), `run_parallel` (473-484), `wall_now` (487), `g_wb_skipped` (495), `ExpertStats` (437), `MAX_READERS` (439), `struct SlotCache` / `SlotCacheSel` (208-401, only what `pool_ensure` calls), `Pending` / `pend_take` / `Arrival` / `arr_wait` (506-534, behind the ops table), the `CHECK` macro (kernels.cuh:17), `EXPERT_RAW` / `EXPERTS` (moe.cuh:5). The callers (moe_block_sparse, decode.cuh:721, and backend-gpu.cu:495-516 for `g_chunks`, `g_vq`, `g_prefill`) were read to know what the configuration values mean; nothing in the engine was changed.

Layout: `src/lib.rs` (the C ABI, process-wide state), `src/ensure.rs` (the algorithm), `src/abi.rs` (`#[repr(C)]` structs), `src/tests.rs` (unit tests), `smoke_rust.c` (includes `ref/smoke_ref.c`: the same C smoke test as the reference). No dependencies (the three libc functions `pread`, `strerror`, `__errno_location` are declared by hand); crate type cdylib + rlib; `panic = "abort"` in release.

## Design

- Everything outside the crate is reached through `sy_ops`, in the order the original makes the calls (`supply.h` ORDER OF EFFECTS). `ensure` is a method of `Supply` (a `Copy` struct holding the ops and the config) so the tests build their own.
- `sy_ensure` takes a busy flag: a concurrent or re-entrant call (for example an ops callback that calls back into the supply) is `exit(1)` with a message, not a data race. `sy_wb_skipped` does not take it (it may be read from an atexit handler after a reader thread called `exit()`).
- The reader jobs are a `Vec<Job>` (dst pointer, file offset, expert, offset inside the record, length) and an `AtomicUsize` that hands out the next index; `run_parallel` is `std::thread::scope` with threads `1..n` spawned first and the calling thread running `work(0)`, which is what the original does with `std::thread` and `join`. `n = min(nthreads, jobs)` as `int`, so 0 or negative `nthreads` still runs everything on the calling thread with tid 0. The chunk arithmetic is a pure function (`chunking`) tested against a hand-derived table.
- The shared read-only state of the jobs (`Ctx`) is a copy of the ops, the `rstream[]` and `bounce[]` arrays, the event, the slab table address, `layer`, `fd`, `dev`, `raw`. The slab table entry is read at each use (as the original reads `g_slab[layer]`), not captured.
- `prot` is built once per call as a sorted, de-duplicated `Vec<i32>`: the original's `unordered_set` (distinct keys) turned into an array in hash order; the slot cache uses it for membership only (slotcache.h), and the count is the same.
- The clock is `std::time::Instant` (`CLOCK_MONOTONIC` on Linux, as `wall_now`). `t_ensure` is added by the outer function around the whole body, so it also covers the early return when nothing is needed (the original's `Timed` destructor does the same). `t_read` / `t_copy` only when something was needed.
- Arithmetic as the C: `ekey` is wrapping `i32`; the file offset is `(e as i64) * raw + co` (wrapping); `chunk` is `size_t` arithmetic (wrapping); `(float)counts[e]` is `as f32`; the slab address is `slab[layer] + (e & 3) * raw (+ co)` with wrapping pointer arithmetic.
- Fatal messages are built as bytes and written with one `write_all` to `stderr` (unbuffered), then `process::exit(1)`; the `pread` error text is `strerror(errno)` read right after the failing `pread`, as bytes (not through `String`).
- `ops.check` is called after every CUDA call with the status (also 0) and the site text (NUL-terminated byte strings in `ensure::site`); a non-zero status ends the process with `exit(1)` if `check` returns.

## Behaviour map (original -> Rust)

| Original (decode.cuh) | Rust (`ensure.rs`) |
|---|---|
| `Timed timed(pool.st.t_ensure)` | `Supply::ensure`: `Instant` before, `t_ensure +=` after `ensure_inner` returns (also after the early return) |
| `if (g_vq) CHECK(cudaStreamSynchronize(0)); else CHECK(cudaDeviceSynchronize());` | `stream_sync(NULL)` + check "cudaStreamSynchronize(0)" / `device_sync()` + check |
| `prot` (unordered_set of `ekey(layer, e)`) | sorted deduped `Vec` |
| `pool.st.needed += experts.size()` | `needed += experts.len()` |
| loop: `pool.c.find(key)`; hit: `[arr_wait]`, `touch(key, (float)counts[e])`, `h_eptr[e] = p`, `hit_card++` | same; `find` is not called on a NULL cache (the original's `rs ? ... : nullptr`), `touch` neither |
| miss: `pool.c.acquire(key, prot, (float)counts[e], &vk, &vh)`; `vk >= 0`: `[arr_wait(vk)]`, `victims.push_back`; `h_eptr[e] = p`; `need.push_back` | `sc_acquire` (`vk` starts at -1 instead of uninitialised); same order |
| `if (need.empty()) return;` | same |
| step 1: `hc.c.find(key)` then `pend_take`, `hc.c.remove`, `held.push_back`, `hit_host++` | same (`find` and `remove` guard a NULL host cache) |
| `CHECK(cudaSetDevice(pool.dev))` | `set_device(pool.dev)` + check |
| step 2: `!hc.cap_slots \|\| g_prefill` -> `break`; `!hc.c.worth` -> `g_wb_skipped++; continue`; `hc.c.acquire(v.key, g_pend_keys, v.heat, ...)`; D2H on `pool.copy`; `written_back++` | same; `host_acquire` is the planner's (it passes its pending keys); `worth` is true for a NULL host cache |
| `CHECK(cudaEventRecord(pool.written_back, pool.copy))` | `event_record` + check, always |
| step 3: for needs with a host slot, H2D on `pool.copy` from `slab[layer] + (e & 3) * raw` (prefill) or the host slot | same |
| step 4: `n_drive`, `pool_readers = min(readers, pool.readers)`, `nchunk`, `chunk`, jobs per expert and chunk (`from_drive++` per expert, `break` at `co >= raw`) | `chunking()`, the job loop |
| job lambda: `b = pool.bounce[tid]`; prefill `b = slab + ...` else `pread` loop (`r <= 0` fatal); `set_device(dev)`, `wait_event(rstream[tid], written_back, 0)`, H2D on `rstream[tid]`, `stream_sync(rstream[tid])`, each checked | `run_job` |
| `run_parallel(jobs, pool_readers)` | `run_parallel` (scoped threads) |
| `CHECK(cudaStreamSynchronize(pool.copy))`, `hc.c.release(h)` for each held, `t_read += t1 - t0`, `t_copy += now - t1` | same order |

## Choices where the header or the original is silent

- `vk` is initialised to -1 and `vh` to 0 before `sc_acquire`, `dk` / `dh` likewise (the original leaves them uninitialised; the slot cache writes `*vkey` on every path, slotcache.h).
- `readers <= 0` (or `pool.readers <= 0`): `pool_readers <= 0`, `n_drive >= pool_readers` gives one chunk, `run_parallel` runs every job on the calling thread with tid 0 (the original does the same by the same arithmetic). `bounce[0]` / `rstream[0]` must then exist. Tested (`zero_or_negative_readers_still_run_every_job_on_the_calling_thread`).
- A tid or stream index >= `SY_MAX_READERS` (only possible if `pool.readers` > 32, outside the contract) is an array index panic = abort; the original reads past the arrays.
- A NULL `pool->cache` is outside the contract: `find` returns NULL without a call, `sc_acquire` is called with NULL (the original would create a cache); same in the reference.
- `fd` is not looked at in prefill mode (it is passed on only to the jobs, which do not read).
- `sy_ensure` before `sy_init`: message `supply: sy_ensure before sy_init`, `exit(1)` (the original has no such state; the same in the reference harness).
- `experts == NULL` with `n > 0` is not handled (undefined, as the original's vector copy would be); `n == 0` is fine with a NULL pointer.
- `sy_init` twice replaces ops and config and resets `g_wb_skipped` (header); a call in flight is not possible (main thread only, busy flag).
- `ekey` wraps; the reference is built with `-fwrapv` so both wrap (the C original is undefined on overflow; `layer <= 92`, `experts <= 896` never overflow).
- `sy_stats` is a plain C struct with the layout of `ExpertStats` (`static_assert` in the reference harness, compared with a real C compile in a Rust test).

## Tests (`cargo test`, 25 tests; expectations derived by hand from the original text; fake ops table with one ordered log shared by the tests, so they run one at a time)

Card hits with and without the card queue (the arrival wait comes before the touch); empty call; the real constants (layer 92, expert 895, key 83327); all from RAM with the held slots released in order; all from the drive with one reader (jobs in expert order on the calling thread, bytes on the card); the mix with one victim written back and one not worth a RAM slot (counters); every victim not worth (the event is still recorded); RAM capacity 0; NULL RAM cache; a duplicate expert; the card queue waiting for the victim before it is copied down; prefill (no `pread`, no write-back, slab sources; chunk sources continue inside the slab record); the chunking table (real record 17547264 = 4284 blocks and 10000 / 4096 bytes, forced chunk counts, zero and negative readers); forced chunks stop at the end of the record; `readers` 0 and -3; `pool.readers` capping the argument; real threads (20 rounds: every chunk once, each reader copy comes from the bounce slot of the thread whose stream it uses; 8 experts on 4 threads); fatal paths in a child process (EOF in a partly present expert, bad fd, CUDA error through `check`, `check` returning, call before `sy_init`: message, exit status, the calls before it); the exported C functions end to end (init, ensure, counter, `n = 0` with NULL, re-init); struct sizes and offsets against a real `cc` compile of the header.

Negative controls (the mutation controls of this crate are done by another agent; none were run by me).

## Smoke (`smoke_rust.c`, C program against `libsupply.so`)

The same 73 checks as `ref/smoke_ref.c` against the reference. Output: `PASS (73 checks, 0 failed)`.

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/supply`)

    cargo build --release          # target/release/libsupply.so   (cargo test does NOT rebuild it)
    cargo test
    gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I include -I ../slotcache/include -o smoke_rust smoke_rust.c -L target/release -lsupply -Wl,-rpath,$PWD/target/release && ./smoke_rust

`verify/cycle.sh` runs all of this plus the reference and the driver build; `verify/sync.ps1` (Windows) copies the crate to the box and runs a command there.

## Not covered

- Equality with the C++ original beyond what the differential driver checks (`verify/RESULTS.md`): ops-call sequence, slot-cache calls, bytes in the pool / RAM arenas, counters, stderr text and exit status, the multiset of reader copies when several threads run.
- The engine, CUDA, a GPU, a real copy queue or planner (`pend_take`, `arr_wait`, `host_acquire` are modelled in the driver); real expert sizes (the driver uses records of 4 to 20 KB, with 17547264 only in the arithmetic tests); performance of any kind.
- Thread interleavings: no ThreadSanitizer, Miri or loom run; the Rust side shares only read-only data between reader threads and an atomic counter, which is argued, not checked by a tool.
- Order in which reader jobs are TAKEN when several threads run (only the set of copies is compared there).
- Out-of-contract calls (see above), 32-bit or non-Linux targets.
