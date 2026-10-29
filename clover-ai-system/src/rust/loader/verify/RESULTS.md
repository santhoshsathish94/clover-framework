# loader - differential driver results (`verify/`)

First driver for `include/loader.h`: the same seeded scenarios run against `ref/libloader_ref.so` (the engine's loader text as committed at 4d8d891, behind the harness) and against `target/release/libloader.so` (Rust). **Not** the mutation controls and **not** the independent review (other agents).

Box `gpu` (Ubuntu 24.04, kernel 6.8.0-147-generic, x86_64, cargo 1.99.1, g++ 13.3.0), 2026-10-25 ~02:02 UTC, no GPU and no CUDA call. `/tmp` is ext4 (the root volume, 99 % full), `/dev/shm` is tmpfs. All processes under `nice -n 10`, at most 20 units at a time. Each unit = one scenario and one seed, run once against each library in its own process; the lines starting `C ` of the two outputs must be byte-identical and both must print `RESULT ... PASS`.

## The driver (`ld_driver.cpp`, one binary, `ld_driver <ref|rust> <lib.so> <scenario> <seed>`)

- Every CUDA effect is a recording callback (events with card, thread id and text, under a mutex; the order of events of ONE card is deterministic because one thread runs a card).
- `open()` and `pread()` are defined by the driver (`-rdynamic`, `-D_FORTIFY_SOURCE=0`), so both libraries resolve to them: they log every call on a path ending in `experts.direct` and can refuse `O_DIRECT` (EINVAL), refuse every open (ENOENT), cap the length of every read (short reads) and fail the n-th read (EINTR / EIO / EAGAIN). The driver's own file access uses raw syscalls.
- Exit paths run in a forked child with stdout and stderr captured: exit status, whether `exit()` flushed stdio (the child prints `BUFFERED` without a newline first) and whether atexit handlers ran (`ATEXIT ...` is written by the handler, with the events, opens and reads seen by then).
- The expectations are computed by the driver itself from the original's text (a model of the pread loop: request length, offset, returned length for a file size, a cap and a failing call; the callback order; the snprintf cut). A passing unit means the library agrees with that model and with the other library; it does not mean the model is right.
- A normal process end (`exit()`, not `_exit`) with the idle detached threads alive closes every passing unit: it must neither hang nor crash.
- Slab and handles tables have 16 canary entries after the 93; they are checked at the end of every scenario.

## Scenarios, and the properties each one checks

| scenario | what runs | checked (each a counted check) |
|---|---|---|
| `fifo` | 0-8 cards (random subset), 0-39 jobs per card (some cards 0 jobs), jobs submitted in random card order from one thread, some jobs sleeping 0-150 us | every job ran once; per card the execution order equals the submission order; one thread per card, different threads for different cards; no two jobs of a card overlap; `set_device` called once per used card, as the first event of that card, on the thread that runs its jobs; no event and no thread for a card without a job; the number of new threads (counted in `/proc/self/task`, settled) equals the number of used cards: lazy start, nothing else started, the threads are still alive and idle; the ctx pointer is passed; a second `ld_wait` returns at once |
| `barrier` | k = 2..8 cards, the first job of every card blocks on a 10 s timed barrier of k parties, 0-3 later jobs per card | the k first jobs ran at the same time (cross-card concurrency actually observed); order and no overlap per card |
| `wait` | (a) `ld_wait` before `ld_init` and with nothing pending: returns within 500 ms; (b) quick jobs; (c) 1-6 jobs blocked on a gate: the waiter has not returned after 100 ms, returns after the gate opens, all jobs done; (d) job A blocked, waiter waiting, ANOTHER thread submits B (same or other card) blocked: after A finished the waiter still waits (100 ms), returns after B finished; (e) 1-3 gated root jobs that submit trees of child jobs (depth up to 3, 4 children): the wait returns with every descendant executed; (f) 1-4 simultaneous waiters all return | the listed ones; "still waiting" is observed for 100 ms (a premature return is detected, a wait that would end very late is not) |
| `stress` | every card warmed from the main thread, then 2-6 submitter threads (ticket assigned under a per-card mutex that also covers the `ld_submit` call), jobs that submit child jobs, a thread calling `ld_wait` in a loop | submitted == executed; per card the execution order equals the ticket order; one thread per card; no overlap |
| `race` | 2-5 threads make the FIRST submit to the same fresh card at the same moment, for each of the 8 cards | one thread and one start hook per card. **Informational for the reference** (its first submit is unlocked: header, difference 1); a check for Rust |
| `init2` | an all-NULL second `ld_init`, then a valid but different second `ld_init` (other callbacks, tables, prefill) | both ignored: the first configuration's callbacks, tables and prefill setting are the ones used |
| `open` | 1-11 layers drawn from 0..92 (and 93, 100000, 4294967295 without prefill), random cards (several layers per card), `k` and `positions` up to 4294967295, files of 4-6 records of random bytes in a dataset dir under `/tmp`; random: prefill on/off, `expert_raw` 4096 / 8192 / 65536, `O_DIRECT` answered naturally or refused, read cap, a barrier in the first `load_trunk` of every card, and "hold" (jobs held in `ensure_work` while the caller's dataset buffer is overwritten after all `ld_open_layer` calls returned) | bookkeeping (`L`, `card`, `k_in` as `int`, `handles[layer]` for layer <= 92) written when the call returns; per card the exact event list (start `set_device`, then per layer in submission order: `set_device, ensure_work, trunk, [tables], state, [host_alloc 4*raw]`); one thread per card; `*h_fd` untouched for layer 0, a valid fd for the others that is the last successful open; open sequence (first open has `O_DIRECT` and `O_RDONLY`; with refusal a second plain open); the fd still open and readable at an aligned offset; the exact pread sequence (length, offset, returned) against the model; slab equals the first 4 records of the file (prefill) or stays untouched; dataset string copied |
| `shm` | like `open`, dataset in `/dev/shm` (tmpfs), O_DIRECT not forced | same, and what the kernel does with `O_DIRECT` there |
| `natural` | `experts.direct` is a symlink to `/proc/version`: the kernel itself refuses `O_DIRECT` (EINVAL) | first open `O_DIRECT` fails with errno 22, second plain open succeeds, the fd reads "Linux versio" |
| `real` | the engine's `expert_raw` 17547264, a sparse 5-record file, prefill, `O_DIRECT` natural or refused | the 4 reads (17547264 bytes at 0, 17547264, 35094528, 52641792) and nothing past byte 70189056; the 70 MB slab equals the file's first 4 records (compared in 1 MiB chunks) |
| `fatal` | 34 child-process cases per seed: missing file with each open mode (natural, `O_DIRECT` refused, everything refused); path totals of 4094, 4095, 4096, 4097 bytes (cut to 4095); end of file inside the 4 records (file size random, caps, both open modes); read error at the n-th read (EINTR, EIO, EAGAIN, cap 0/1000/4000); `ld_submit` before init, card 8, card -1, NULL job; `ld_open_layer` before init, card 9, NULL req, NULL dataset, layer 93 and 4000000000 with prefill; `ld_init(NULL)` and each missing required callback / table, prefill without `host_alloc` / `slab`; an exit while another card's job runs and the main thread waits | exit status 1; the stderr text byte for byte followed by the atexit text (events and open / read counts as the model predicts, so also that nothing ran after the failure and that no retry happened); stdout flushed by `exit()`; no hang |

## Raw summary lines (excerpt of `final.log`, which is kept in this directory in full; the lines for `cargo test` and the two smokes are abridged here; per-unit raw files were deleted from the box after reading, `final.sh` recreates them in ~25 s)

```
== date 2026-10-25T02:02:15+00:00  host ubuntu2404  kernel 6.8.0-147-generic  cargo cargo 1.99.1 (5f94df478 2026-08-27)
== cargo test: running 32 tests / test result: ok. 32 passed; 0 failed; 0 ignored
== final md5
2f1d3eacf8ebc2e0078b61f8ab629f21  ../target/release/libloader.so
bb41075c183f95a9d9d800a0a377477b  ../ref/libloader_ref.so
f104b44a3bd82b6657e6e68f4bdb5fa6  ld_driver
af286f75ad09c319d2e748d317cda7be  ../include/loader.h
== smoke_ref   PASS basic (5) noprefill (9) prefill (10) missing (2) eof (2) badcard (2), 0 failed
== smoke_rust  PASS basic (5) noprefill (9) prefill (10) missing (2) eof (2) badcard (2), 0 failed
== matrix N=200
units: 2820 of 2820, wall 25 s
SUMMARY units=2820 ref_pass=2820 rust_pass=2820 canon_same=2820 ref_rc0=2820 rust_rc0=2820
SCENARIO fifo units=1000 ref_pass=1000 rust_pass=1000 canon_same=1000
SCENARIO barrier units=200 ref_pass=200 rust_pass=200 canon_same=200
SCENARIO wait units=200 ref_pass=200 rust_pass=200 canon_same=200
SCENARIO stress units=200 ref_pass=200 rust_pass=200 canon_same=200
SCENARIO race units=200 ref_pass=200 rust_pass=200 canon_same=200
SCENARIO init2 units=200 ref_pass=200 rust_pass=200 canon_same=200
SCENARIO open units=600 ref_pass=600 rust_pass=600 canon_same=600
SCENARIO shm units=100 ref_pass=100 rust_pass=100 canon_same=100
SCENARIO natural units=40 ref_pass=40 rust_pass=40 canon_same=40
SCENARIO real units=40 ref_pass=40 rust_pass=40 canon_same=40
SCENARIO fatal units=40 ref_pass=40 rust_pass=40 canon_same=40
CHECKS ref=177367 rust=177567
O_DIRECT natural answers (layer lines, rust runs): /tmp accepted=1659 refused=0; /dev/shm accepted=610 refused=0
race (informational) ref violations: 339 of 1600 first-submit trials  rust: 0 of 1600
BEGIN_LINES=2820 UNIT_LINES=2820
NOT-OK UNITS:            (none)
F-lines: ref=0 rust=0
stderr files: 0
outdir bytes: 6757267
== leftovers of this run in /tmp and /dev/shm: (none)
```

Coverage of the seeds (`stats.sh`): open+shm 700 units, 4345 layer lines (layer 0: 238, layer 92: 198, layers 93 / 100000 / 4294967295: 14 / 13 / 15, `k_in = -1`: 536, 2122 prefill layers with slab equal, 1022 card lines with two or more layers); prefill 351 / 349, `O_DIRECT` natural 395 / refused 305, hold 350 / 350, barrier 221, reads capped at 0 / 1 / 7 / 512 / 4096 / 5000 / 8192 bytes in 147 / 61 / 70 / 150 / 105 / 62 / 105 units, up to 262144 `pread` calls for one layer; fatal 1360 cases, all exit 1, all flushed; EINTR / EIO / EAGAIN 41 / 41 / 38; `wait` waiters 1 / 2 / 3 / 4 in 48 / 55 / 46 / 51 units, 3268 nested nodes (largest tree 61); `fifo` cards used 0..8 = 44 / 181 / 161 / 152 / 180 / 141 / 87 / 48 / 6, 65249 jobs, 1196 zero-job cards; `barrier` parties 2..8 = 36 / 20 / 39 / 19 / 28 / 29 / 29; `stress` 803 submitter threads, 50527 jobs, 396 concurrent waits; `real` 40 of 40 slab equal (23 with natural `O_DIRECT`, 17 plain); `natural` 75 layers, 75 refused with errno 22; `init2` 200.

## Soak (`./run.sh 1000 out-soak`, same binaries, run right after the matrix above)

```
units: 14100 of 14100, wall 106 s
SUMMARY units=14100 ref_pass=14100 rust_pass=14100 canon_same=14100 ref_rc0=14100 rust_rc0=14100
SCENARIO fifo units=5000 ... barrier 1000 wait 1000 stress 1000 race 1000 init2 1000 open 3000 shm 500 natural 200 real 200 fatal 200   (every one: ref_pass = rust_pass = canon_same = units)
CHECKS ref=883395 rust=884395
O_DIRECT natural answers (layer lines, rust runs): /tmp accepted=8237 refused=0; /dev/shm accepted=2906 refused=0
race (informational) ref violations: 1686 of 8000 first-submit trials  rust: 0 of 8000
BEGIN_LINES=14100 UNIT_LINES=14100       NOT-OK UNITS: (none)      F-lines: ref=0 rust=0      stderr files: 0      outdir bytes: 33653758
```

## Observations

- **O_DIRECT on this box:** `/tmp` (ext4) accepted `O_DIRECT` in all 1659 opens of mode "natural"; `/dev/shm` (tmpfs, kernel 6.8) accepted it too (610 of 610). A refusal by the kernel itself was only produced with a procfs file (`natural`: errno 22, 75 of 75). The fallback to a plain open was therefore exercised with (a) the driver's forced refusal (EINVAL, all `mode=1` units) and (b) procfs; not with a file system that refuses `O_DIRECT` for a regular data file.
- **The original's unlocked first submit is real:** 339 of 1600 trials (21 %) with 2-5 threads making the first submit to a card at once gave the reference more than one thread or start hook for that card (an earlier run of the same scenario: 325 and 386 of 1600). The Rust crate: 0 of 1600. This is the data race named in the header (difference 1); the engine submits from one thread, so it has not been observed in the engine.
- **An instrument flake, found and fixed in the driver (not in a library):** the second full run with the thread-count check in `fifo` ended 2019 / 2020 for Rust: seed 3 reported 6 threads where 5 were expected, and the next read, a moment later, 5. The extra entry was most likely the joined helper thread of `timed_wait` still listed in `/proc/self/task` for a moment (the reference passed the same seed in the same run). The check now waits for 5 consecutive equal reads (3 s limit); a thread left over by a library keeps the count wrong for the whole 3 s and fails. The matrix was re-run after the fix with `fifo` at 1000 seeds: 1000 / 1000 for both libraries. This is an inference from the second read, not an isolated proof: I did not reproduce the transient on demand.
- **Order of work during this session:** the driver passed 2020 / 2020 on its first complete run, 2019 / 2020 on the second (`fifo 3`, above), then 2820 / 2820 after the fix. A first-run pass means the driver and both libraries agree, not that the driver can fail; its sensitivity is shown by the `race` results on the reference (it does see a violation when one exists) and by the `extract.py` negative controls; no mutants were made (not part of this task).

## Not covered

- **Thread interleavings are not deterministic.** What is checked are properties (per-card FIFO by ticket, one thread per card, no overlap, the count, the wait outcomes with gates, concurrency with a barrier). Not checked: a specific schedule, fairness, wake-up latency of `ld_wait`, a missed wake-up that only occurs in a narrow window. No Miri, ThreadSanitizer or loom run; memory ordering is argued (mutexes), not tested (x86 hides weak-memory bugs).
- **Visibility of a job's writes** (`*h_fd`, slab bytes) to the thread that returns from `ld_wait` is only exercised on x86-64; it relies on the pending mutex.
- **No CUDA, no engine:** the GPUs were in use by another agent; the callbacks are fakes. There is no `K3GPU_LOADER` wiring in the engine yet (engine files not touched); the engine with the Rust loader was neither built nor run.
- Not exercised: failure to create a thread; a failing `write` to stderr; `ld_wait` called from inside a job (deadlocks by contract); `host_alloc` returning without storing a pointer; `O_DIRECT` with an `expert_raw` that is not a multiple of the block size; `ld_init(NULL)` after a successful init; `ld_open_layer` with NULL `h_*` pointers (the header says "do not write"; the code skips them, no test passes NULL); `ld_open_layer` from several threads; a caller that changes `*h_card` between submit and run; callbacks that call back into the loader; aarch64 (the `O_DIRECT` constant for it is only reasoned and unit-tested against the header on x86-64); a file system refusing `O_DIRECT` for a regular file.
- The reference is the original's text with the listed adaptations (see `ref/NOTES.md`): agreement with it is agreement with that text behind a harness, not with a running engine. Where the original has no behaviour (guards), both implementations have my text.
- File descriptor numbers, timings and thread ids are not compared (they differ run to run).

## Mutations a driver (this one) could easily miss, for the agents who write the mutation controls

1. `notify_all` -> `notify_one` after the decrement: only the `wait (f)` units with 2+ waiters can see it (152 of 200 units), and only as a hang.
2. `pending++` moved after the card lock, after the thread start or after the push; or `pending--` / `notify_all` reordered (notify before the decrement: a lost wake-up needs a waiter asleep at that exact moment). No deterministic test.
3. `ld_open_layer` writing `*h_L`, `*h_card`, `*h_k_in` or `handles[layer]` AFTER queueing the job instead of before: the driver reads them right after the call returns, so it cannot tell (the job does not read them; the driver's callbacks use their own record).
4. The job reading `*h_card` at run time instead of using the value captured at submit: only differs if the caller changes the int in between (not exercised).
5. `Ordering::Relaxed` instead of the mutex's ordering anywhere, or a non-atomic counter: invisible on x86.
6. Thread stack size, thread names, `Box::leak` vs a dropped loader at process end (the driver ends passing units with `exit()`, but nothing is freed either way).
7. Closing the fd after the slab read: caught only by the fd read-back in `open` / `natural` / `real` (present); a `close` of a different descriptor would not be.
8. `ld_init(NULL)` handled before the "already initialised" check or after it (the driver only calls `ld_init(NULL)` as the first call).
9. The second guard order (e.g. the slab-slot check before the card check): only visible for a request that is wrong in both ways (card out of range AND layer > 92 with prefill) - not generated.
10. `expert_raw == 0` and `expert_raw` not a multiple of 512 under a real `O_DIRECT`: not generated by `open` (it uses 4096 / 8192 / 65536 / 17547264); unit tests cover 0 only with the fake libc.
11. The `l.th = std::thread(...)` assignment on a fresh `Loader` versus a thread started by `Builder::spawn` with a different stack size: not observable.
12. A retry of `pread` on EINTR that gives up after one retry: the `readerr` cases fail the n-th call only once, so a single retry would succeed and be seen as extra calls (caught); a retry only on EINTR with `EIO` unchanged is caught only in the 41 EINTR units.

## Re-run

```
pwsh verify\sync.ps1                                   # Windows: tar of the crate -> box
ssh gpu 'cd ~/rustmig/loader/verify && ./final.sh 200'  # build, cargo test, md5, both smokes, matrix (N=200 -> 2820 units, ~25 s), stats
ssh gpu 'cd ~/rustmig/loader/verify && ./run.sh 1000 out-soak'   # a longer soak
```
`./unit.sh <scenario> <seed> <outdir>` runs one unit; `./build.sh` rebuilds both libraries and the driver; remove `out-*` and `/tmp/ldloader-verify.*` when done.

## Driver v2 (2026-10-25): the blind spots the mutation controls found

The mutation controls (`mutants/RESULTS.md`, an independent agent) ran 101 broken copies of the crate against the driver above: 86 failed it, 15 did not (5 equivalent, 2 only for `expert_raw >= 2^30`, 1 fd 0, 1 contract-silent, 6 real gaps) and a few were caught by 40 units or fewer. This section records what was changed in the driver to close them, what was run to show it, and what is still open. Changed: `ld_driver.cpp`, `run.sh`, `summarize.sh`, `stats.sh`, `final.sh`, `sync.ps1`; new: `canon_cmp.sh`, and in `mutants/` the scripts `run_mut2.sh`, `try_mut.sh`, `mk_all.sh`, `v2_campaign.sh`, `v2_gap_tries.sh`, `analyze2.py`, `mk_results2.py`, the record `v2-hotwait-tuning.txt` and the audit trail `v2out/` (gzipped `units.log` of every run). Not changed: the crate's `src/`, `include/loader.h`, `ref/`, `NOTES.md`, the independent agent's scripts, diffs and results (a section is appended to `mutants/RESULTS.md`).

### What changed in the driver, per blind spot

All new scenarios come after the old ones in `run.sh` / `run_mut2.sh`, and the random draws of the old scenarios are untouched (new random choices come from a second stream or from new scenarios), so every old seed is the same test as before. `canon_cmp.sh 40` ran the old driver (kept as `verify/v2work/ld_driver_v1` on the box, md5 `f104b44a3bd82b6657e6e68f4bdb5fa6`) and the new one on seeds 1-40 of every old scenario against both libraries (564 units, 1128 library runs): **no canonical `C ` line of the old driver was removed or changed; lines were only added (in `open` and `shm`, 280 of the 1128 runs)**.

| gap (mutants) | change in `ld_driver.cpp` | what it can now see |
|---|---|---|
| handle bookkeeping (M82, M83, M84) | `cb_trunk`, the first callback that gets the handle, checks that `*h_L`, `*h_card`, `*h_k_in` and `handles[layer]` are already written (header: "at once, on the calling thread, before the job is queued"). New scenario `book`: every handle lives on its own page and the `handles[]` table on a page of its own, all read-only; the first store of the loader to a page raises SIGSEGV on the calling thread, in the middle of `ld_open_layer`; the handler waits until the job of that handle reaches `load_trunk` (at most 15 ms), records whether it did, makes the page writable and returns. A loader that writes first and queues afterwards never has a job to wait for; one that queues first has the job run inside the handler with the fields unwritten. The result does not depend on how wide the window naturally is. In `open` units with `hold` a gate job is queued first on every used card, so no layer job can start before the caller has overwritten the dataset buffer and every `*h_card` (with card + 1 mod 8) after all `ld_open_layer` calls returned; the job must keep the card it was given. `*h_fd` is checked from inside the job: at `host_alloc` it must equal the last successful open, at every `pread` it must equal the descriptor that is read. | M82 (`book`), M83 (`book`; `open` without hold when the 2 ms window holds the callbacks), M84 (`open` units with hold) |
| hot waiters (M05, M11, M14, M15, M05w, M11w) | New scenario `hotwait`, the `lw_probe.c` of the controls inside the driver and hardened: 1-5 waiter threads call `ld_wait` in a loop while the main thread submits single jobs and bursts of 2-40 jobs to one card (a third of the jobs run 5-30 us) and calls `ld_wait` itself (every job must be done when it returns); every 25 rounds the stream stops and every waiter must still be returning from `ld_wait` (a stuck waiter has slept through its wake-up); after the last round every waiter must return; a watchdog fails a unit that makes no progress for 10 s. New check: a waiter that returns from `ld_wait` while a job that STARTED BEFORE its call is still running has seen the counter at 0 with a job queued (the jobs note their start time and a running count; a newer job only makes the check skip, so a correct loader cannot trip it). In 15 of 16 seeds a chaos thread stalls threads with signals (the handler sleeps 0-300 us; three of four signals go to the submitting thread). It only delays: every schedule it produces is one a correct loader must survive, and it widens the nanosecond windows (a stall lands inside a window with probability window / loop length). | M05, M11, M14, M15, M05w, M11w |
| opens are not events (M33) | The `open()` interposer logs `open L<n> direct` / `open L<n> plain` as an event on the card of the layer, flagged so that the callback-only views and the `ATEXIT` text of `fatal` do not change; `open` and `shm` compare the full list (callbacks, opens, `host_alloc`) per card and emit it as a new canonical line `open card N seq ...`. | M33 in every `open` / `shm` unit with a layer >= 1 |
| `O_DIRECT` refused only with EINVAL (M46) | In `open` units of mode 1 the refused first open fails with EINVAL, EACCES, EOPNOTSUPP, ENOENT, EIO, EPERM or EISDIR (drawn from a second random stream); the unit checks the errno it forced; new canonical line `open refuse errno=N`. | M46 in the units that refuse with another errno than EINVAL |
| guard order (M87, M92) | New scenario `guard` (children, exact stderr and `ATEXIT` text): card 9 AND layer 93 with prefill (the card is reported), card -1 and layer 93, ld_open_layer before init with both wrong, NULL dataset with card 9, NULL job with card 9, a NULL job before init, `ld_init(NULL)` after a valid init (fatal "cfg is NULL"), and pairs of missing configuration items in `ld_init` (the first of the header's list is reported). | M87, M92 |
| stack depth (M27) | New scenario `stack` (children): a job that touches 3 MB and one that touches 6 MB of stack on a loader thread must run (skipped, the same way for both libraries, if RLIMIT_STACK is below 8 MB or unlimited). | M27 |
| large `expert_raw` (M58, M59) | New scenario `big`: `expert_raw` from 2^30 to 2^33 + 4096 (`4 * raw` up to 32 GiB), without allocation and without a file: `host_alloc` hands out a made-up address, the `open` interposer gives `/dev/null`, the `pread` interposer answers from a virtual file size and never touches the buffer. The driver checks `4 * raw`, every offset `r * raw + got` and length, the number of reads and (in every `pread` of every scenario now) that the buffer is `slab + offset`. **The driver can cover this, so no unit test in `tests.rs` is proposed for it.** | M59 (every unit), M58 (raw >= 1.5e9; raw = 2^30 cannot show a 32-bit product, 3 * 2^30 < 2^32) |
| fd 0 (M51) | New scenario `fd0` (children): the child closes its stdin, so the first successful `open` returns descriptor 0 (prefill or not, `O_DIRECT` accepted or refused); the unit checks `*h_fd == 0`, that the descriptor reads the layer's file, and the slab. | M51 |
| weak catches (M25, M11, M14) | M25: new scenario `race2`, 40 fresh processes per unit, 8 cards x 3-8 threads making the first submit at the same moment, half of the processes under chaos. M11 and M14: `hotwait`. | M25, M11, M14 |

Decisions the owner of the header should see (the driver pins what both implementations do; the header is silent or ambiguous):

- **Stack size.** The header states none. The original's threads (`std::thread`) get the process default, RLIMIT_STACK (8 MB with the usual `ulimit -s`; glibc on x86-64 takes 2 MB if it is unlimited); `NOTES.md` says the crate uses 8 MB for that reason. So the contract promises nothing; `stack` pins the original's behaviour under an 8 MB limit (3 and 6 MB of stack in a job).
- **`*h_card` after the call.** The header is silent. The reference's job reads its own copy (`IntRef` keeps the value, the caller's int is only written through), the crate captures the card at the call, and the ORIGINAL text reads `h->card` (the Handle's live field) when the job runs. The `open` hold units overwrite the caller's `*h_card` while the jobs are held back and require the captured card, i.e. they pin the behaviour of the reference and the crate, which differs from the original's text for a caller that changes the field. Whether the engine ever does, was not checked (engine not read). If the header should leave this open, drop the overwrite in `scn_open` (M84 then stays uncaught).
- **`ld_init(NULL)` after a successful `ld_init`.** The header says a NULL `cfg` is fatal and that a second `ld_init` is ignored "also an invalid one". Reference and crate: fatal (`loader: cfg is NULL`); `guard` pins it (M87 swaps the two checks).
- **Which of two missing configuration items is reported.** The header lists them in the order set_device, ensure_work, load_trunk, load_expert_tables, alloc_state, handles, (prefill:) host_alloc, slab; reference and crate report the first; `guard` pins it.

### What was run, and what it showed

Box `gpu` (Ubuntu 24.04, kernel 6.8.0-147, x86_64, 96 cores, load average 80-120 from other agents during the runs), 2026-10-25 06:57-[[END_TIME]] UTC; no GPU, no CUDA, no sudo. Raw numbers below are copied from the files in `mutants/v2out/` (`ctl/`, `units/`, `progress.txt`).

**(a) The unmodified crate passes the whole new driver.** `./run.sh 200` three times (3480 units each: the old 2820 plus 660 new; every unit runs the reference and the crate in their own processes) and `./run.sh 1000` once (17,400 units); the library is the crate's own `target/release/libloader.so` (md5 `cb3f42cb67c2d932ad032835b8262378`, see (d)):

| run | units | ref pass | rust pass | canonical lines identical | ref / rust exit 0 | not-ok units | wall |
|---|---|---|---|---|---|---|---|
| N=200, a | 3480 | 3480 | 3480 | 3480 | 3480 / 3480 | 0 | 53 s |
| N=200, b | 3480 | 3480 | 3480 | 3480 | 3480 / 3480 | 0 | 63 s |
| N=200, c | 3480 | 3480 | 3480 | 3480 | 3480 / 3480 | 0 | 72 s |
| N=1000 | 17400 | 17400 | 17400 | 17400 | 17400 / 17400 | 0 | 402 s |

N=1000 per scenario (every one: ref pass = rust pass = canonical identical = units): fifo 5000, barrier / wait / stress / race / init2 1000 each, open 3000, shm 500, natural / real / fatal 200 each, book 1000, hotwait 1000, race2 500, big / fd0 / stack / guard 200 each; 244,033,601 counted checks for the crate, F lines 0 for both. Coverage of the new scenarios in that run (`stats.sh`): `book` 3983 handles written on read-only pages; `hotwait` 1000 units, 2983 waiter threads, 4,458,748 rounds, 49,048,990 jobs, 177,877 quiesce checks with 0 stuck waiters, 948 units with chaos and 6,757,958 signals; `race2` 20,000 fresh processes and 160,000 first-submit trials: **the reference violated "one thread and one start hook per card" in 58,360 of them (36 %), the crate in 0** (the old `race` scenario saw 1958 of 8000, 24 %, on the reference in the same run: the new one finds the original's race 1.5 times as often per trial, which also shows that it can fail); `big` raw 2^30 ... 2^33 + 4096 (slab up to 34,359,754,752 bytes) 200 units; `fd0` 800 of 800 children got descriptor 0; `stack` 400 jobs of 3 / 6 MB; `guard` 3000 cases, all exit 1; `open` refusals with errno 1 / 2 / 5 / 13 / 21 / 95 in 215 / 196 / 192 / 174 / 186 / 190 of the 1528 mode-1 units (the other 375 use EINVAL; errno 22 is also printed by every unit that refuses nothing).

[[B]]
