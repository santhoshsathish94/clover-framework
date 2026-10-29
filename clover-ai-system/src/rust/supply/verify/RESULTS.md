# supply - differential verification results

Date 2026-10-25, box `gpu` (`~/rustmig/supply`), **no GPU used and no CUDA call made** (every CUDA effect is a recording fake). The GPUs, `~/k3gpu`, `~/ncm` and other agents' files and processes were not touched; my scratch was under `~/rustmig/supply/verify/out`, run with `nice -n 10`, 20 parallel jobs, 53 MB per run (the data files) and 212 MB in total before cleanup; the data files, the earlier scratch runs and the debug build were deleted afterwards (`verify/out` keeps `final.log`, `final-1..3` with summaries, `units.log` and the first two seeds' gzipped logs of each scenario).

## What is compared

`verify/sy_driver.cpp` runs one unit (a seeded world and 8 to 45 `sy_ensure` calls, config printed on the first line of every log) against ONE library per child process: the C++ reference (`ref/libsupply_ref.so`, the original `pool_ensure` text behind a harness) or the Rust crate (`target/release/libsupply.so`), and compares the two logs byte for byte; the exit status and the stderr text (the libraries' fatal messages go into the same file) are part of it.

Everything but the supply is real or modelled:
- the slot caches are the real `libslotcache.so` (md5 below), the same file in both arms, for the card pool and for the RAM cache; pool slots, RAM-cache slots, bounce slots and slab are real memory (records of 4 to 20 KB; the record size 17547264 appears only in the arithmetic tests);
- the drive reads are real `pread` calls on files whose bytes are a pure function of (file, offset) (`f<layer % 4>_<experts>_<raw>.bin`); failure units use a file cut at 40 % or a closed fd;
- the CUDA ops are fakes that record the call (with the hash of the bytes a copy reads) and do the copy with `memcpy` when the call is made; `check` is a fake too (a non-zero status prints `cuda error <n> at <site text>` and exits 1; statuses are injected by occurrence number: the 1st to 3rd call of one of ten op kinds);
- the planner side is modelled like the engine's: the RAM cache holds entries that are **pending** (POISON bytes until `pend_take` completes the read) and the pool holds entries that are **arriving** (POISON until `arr_wait`); `host_acquire` is `sc_acquire` on the RAM cache with the pending keys as `prot`; the world keeps the engine's invariant that an expert is never in the pool and in the RAM cache at once.

The log has one line per call of every `sy_ops` function with its arguments and results (slot numbers, victim key and heat bits, `prot` size / distinct count / set hash, bytes hashes of copies), one line per reader copy, a `HP` line per call (where each requested expert is on the card), and after every call a `Z` line: the six counters of the pool, `g_wb_skipped`, the full state of both caches (key, slot, heat, next-use of every entry in order, spares in order), a hash of each arena's bytes, the number of arriving and pending entries, the number of next-use function calls. Passing units that are not among the first two seeds of a scenario delete their logs.

Absolute checks (they hold for each arm on its own, a failure is `VIOLATION ...` and exit 3): every requested expert is on the card with the bytes of its record (from the file or, in prefill, the slab); every cache entry that is not pending or arriving holds its expert's bytes; the counters equal the counts derived from the calls (`needed`, `hit_card`, `hit_host`, `from_drive`, `written_back`, `wb_skipped`; `find hits + acquires = n`); the set of reader copies equals the set a separate chunk model derives (own arithmetic, with the hash of the source bytes of each copy), in list order when one thread runs and as a multiset when several do; a victim is complete when it is copied down (arr_wait before it), a host key is taken out only after `pend_take`, a card hit is touched only after `arr_wait`, the event is recorded before a reader waits for it and before the main copy, a reader copy comes from the bounce slot (or slab) of the thread whose stream it uses, the final `stream_sync(pool.copy)` happens after every job ended, a host slot is released only after it and only if it was held, `worth` / `host_acquire` are never called with capacity 0 or in prefill, `t_ensure >= t_read + t_copy` and both are exactly 0 in a call that needed nothing.

**Threads.** A reader job's calls are collected per job (`J d=.. dst=.. n=.. src=.. h=.. [k=..] ck=kd,kw,kc,ks`) and committed when its last check arrives. In a call where one thread runs (`min(readers, pool.readers) <= 1`, or fewer than two jobs) the records go into the log in commit order (so the job order is compared); where several run they are sorted (the multiset is compared, the order in which jobs were taken is not). T1 is the second set: 8 to 32 reader slots, 12 to 40 experts per call, chunk counts up to 8, fake ops that sleep up to 400 us at random, **two runs per arm** (4 children, all four logs must be equal), and a check that at least half of the calls with >= 8 jobs used more than one reader thread (otherwise a supply that ran everything on the calling thread would pass).

Scenarios (60 units each unless noted; seeds 1..N; all sizes, layers, flags are seeded): Y1 all hits on the card (with arrivals), Y2 all from RAM (with pending reads), Y3 all from the drive, Y4 mix, Y5 write-back / not worth / RAM capacity 0 / NULL RAM cache / RAM cache full of pending entries (80 units, small RAM caches on purpose), Y6 prefill with a slab (`fd` -1 half of the time), Y7 chunking (1 to 4 experts per call, 4 to 32 reader slots, forced chunk counts 1 to 64, raw 4097 / 5000 / 12345 ...), Y8 counts (0 to 9) / duplicate experts / layers 0 to 92 / E = 1, 17, 896, Y9 reader counts -2 to 16 against `pool.readers` 1 to 32, YF failures (80 units: short file, closed fd, injected CUDA status, RAM cache full; `pool.readers` 1 so the first failing job is determined), T1 real threads (60 units). Policies 0, 1, 2 (with and without the next-use function), card queue on and off, prefill on and off, chunk override on and off are drawn in every scenario.

## Commands

```
# on the box, ~/rustmig/supply (export PATH=$HOME/.cargo/bin:$PATH)
verify/final.sh                         # cycle.sh (reference build + smoke, cargo build/test, Rust smoke, driver build, md5) then three full runs final-1..3; log in verify/out/final.log
verify/go.sh NAME --keep-going          # driver build + one full run in the background (summary in verify/out/NAME/summary.txt); returns at once
verify/run.sh NAME [--scn Y2,T1 --n Y2=100,T1=20 --jobs N --keep-going]    # the run itself (REF=/RUST=/JOBS= in the environment point it at other builds; a mutated crate goes in RUST)
verify/dbg.sh Y2 7                      # one unit again with its logs kept (out/dbg), prints config and the tail of the reference log
# on Windows
powershell verify/sync.ps1 -Cmd 'verify/final.sh' -Out C:\x\final.txt     # copy the crate to the box and run a command there
```

## Files that were run (md5 on the box, final cycle 2026-10-25 02:18 UTC)

```
0970175f3c2badc6c4f2b0bba8996b38  include/supply.h
92924844e25b8c1aebc65ba2fbe02363  ref/supply_ref.cpp
c416610e5591e132ba7b90ff0aa2448d  ref/supply_types.inc
2254d65eca7e4b52a89a125abcb7dfab  ref/supply_extracted.inc
7532a0ef7291735fb1dd5a1d53984390  ref/libsupply_ref.so
0fd33fdaa14c1156c465031f154c4edc  src/abi.rs
916430e32957a452def64e94791d1118  src/ensure.rs
6b7bbc9694995dfe97bd4869a841dee9  src/lib.rs
890d028e5eda4823373f7c1fda57f9ec  src/tests.rs
4e760a9f36b649de42697dacc45816bf  Cargo.toml
f77f2b730e8c1b254f379cd9fc73f9fb  target/release/libsupply.so          (cargo build --release; cargo test does not rebuild it)
390b6cae80354ededbe315147a84696b  verify/sy_driver.cpp
612e76a647f3663b859dbed71eecad64  verify/sy_driver
536470b28584426b4c60d0e7390ab4fd  ../slotcache/target/release/libslotcache.so
```

## Results (final cycle: `verify/out/final.log`)

Reference smoke `PASS (73 checks, 0 failed)`; `cargo test` `25 passed; 0 failed`; Rust smoke `PASS (73 checks, 0 failed)`; driver built without warning.

Differential run `final-1` (700 units; `final-2` and `final-3` gave the identical lines, with the T1 interleaving counts 4360 and 4353 instead of 4363, which is real thread timing; all three exit status 0, no failed unit, no retry, no fail file):

```
Y1 units=60 pass=60 fail=0 logged_lines_ref=62475 logged_lines_rust=62475 (all lines 64413/64413, Z lines 1818/1818) units_ending_in_fatal_exit1=0 PASS
Y2 units=60 pass=60 fail=0 logged_lines_ref=102992 logged_lines_rust=102992 (all lines 104915/104915, Z lines 1803/1803) units_ending_in_fatal_exit1=0 PASS
Y3 units=60 pass=60 fail=0 logged_lines_ref=119253 logged_lines_rust=119253 (all lines 121142/121142, Z lines 1769/1769) units_ending_in_fatal_exit1=0 PASS
Y4 units=60 pass=60 fail=0 logged_lines_ref=102672 logged_lines_rust=102672 (all lines 104588/104588, Z lines 1796/1796) units_ending_in_fatal_exit1=0 PASS
Y5 units=80 pass=80 fail=0 logged_lines_ref=91605 logged_lines_rust=91605 (all lines 93979/93979, Z lines 2214/2214) units_ending_in_fatal_exit1=8 (slotcache 8, pread 0, cuda 0) PASS
Y6 units=60 pass=60 fail=0 logged_lines_ref=93140 logged_lines_rust=93140 (all lines 95004/95004, Z lines 1744/1744) units_ending_in_fatal_exit1=0 PASS
Y7 units=60 pass=60 fail=0 logged_lines_ref=51570 logged_lines_rust=51570 (all lines 53483/53483, Z lines 1793/1793) units_ending_in_fatal_exit1=0 PASS
Y8 units=60 pass=60 fail=0 logged_lines_ref=114830 logged_lines_rust=114830 (all lines 116829/116829, Z lines 1879/1879) units_ending_in_fatal_exit1=0 PASS
Y9 units=60 pass=60 fail=0 logged_lines_ref=127326 logged_lines_rust=127326 (all lines 129246/129246, Z lines 1800/1800) units_ending_in_fatal_exit1=0 PASS
YF units=80 pass=80 fail=0 logged_lines_ref=25569 logged_lines_rust=25569 (all lines 26006/26006, Z lines 277/277) units_ending_in_fatal_exit1=68 (slotcache 2, pread 47, cuda 19) PASS
T1 units=60 pass=60 fail=0 logged_lines_ref=236699 logged_lines_rust=236699 (all lines 238075/238075, Z lines 1256/1256) units_ending_in_fatal_exit1=0 threaded_ensures=4496 of_which_reader_copy_commit_order_differed_from_job_list_order=4363 PASS
```

`logged_lines` are the log lines minus the `Z`, `CFG` and `END` lines; the two arms have the same number in every scenario. 700 units x 3 runs = 2100 unit comparisons, 0 differences. "Fatal exit 1" units are expected to end that way and the two arms must end the same way with the same log and message; a fatal unit passes only if the log holds one of the three known messages (`every slot is in use by one call`, `reading expert .. failed:`, `cuda error ..`). YF: 47 units ended in the `pread` message (short file or closed fd), 19 in an injected CUDA status (the sites hit in the sample I read: `cudaSetDevice(dev)` in a reader job), 2 in the slot cache's fatal, 12 never reached the failure (the injected occurrence did not exist, or the RAM cache never filled).

Coverage (sums over the units of the reference arm, `final-1`; ensure calls / card hits / pool acquires = experts that were not on the card / RAM hits / write-backs / victims not worth a RAM slot / arrivals waited for / pending reads waited for / reader copies / calls with more than one chunk per expert / calls that needed nothing / calls with a duplicate expert):

```
Y1 1818 / 14195 / 708 / 470 / 704 / 1 / 2428 / 5 / 448 / 123 / 1506 / 48
Y2 1803 / 3069 / 6263 / 5971 / 4902 / 587 / 27 / 2095 / 429 / 48 / 259 / 30
Y3 1769 / 6713 / 8683 / 2781 / 7204 / 681 / 9 / 0 / 9562 / 647 / 228 / 54
Y4 1796 / 6582 / 6661 / 3108 / 5016 / 763 / 275 / 415 / 5202 / 528 / 283 / 18
Y5 2214 / 4739 / 7221 / 1342 / 2943 / 2050 / 329 / 89 / 8438 / 880 / 229 / 50
Y6 1744 / 8546 / 7789 / 2360 / 0 / 0 / 384 / 630 / 8713 / 702 / 316 / 35
Y7 1793 / 1314 / 3185 / 789 / 2171 / 277 / 31 / 65 / 5383 / 1021 / 246 / 57
Y8 1879 / 9193 / 6718 / 3229 / 6113 / 287 / 393 / 250 / 5129 / 400 / 575 / 633
Y9 1800 / 9039 / 8569 / 3665 / 6902 / 892 / 266 / 362 / 7795 / 486 / 281 / 33
YF 277 / 971 / 1527 / 431 / 1195 / 31 / 61 / 113 / 1214 / 17 / 41 / 6
T1 1256 / 12068 / 17974 / 9146 / 15218 / 435 / 242 / 696 / 16331 / 880 / 0 / 0
```
Y1's "calls that needed nothing" is 1506 of 1818; Y6 (prefill) has 0 write-backs as it must.

Raw logs I read myself: `final-1/Y2/u1` (first 64 lines: the world's placements, a call taking one pending expert from RAM with `PN` before `H R`, the `H2D` on `c`, the release after `SS(c)`, then a seven-expert call with a victim of the previous call), `final-1/T1/u1` (first 40 lines), the `cmp` of the Y2 u1 ref and rust logs (identical md5 of the uncompressed logs), and the tails of YF 4 (`cuda error 705 at cudaSetDevice(dev)` after the last main-thread copy) and YF 2 (`reading expert 14 of layer 0 failed: Bad file descriptor`, in chronological order after the last write-down) and of the earlier fatal units (below).

## History of the driver (what went wrong and was fixed, in order)

1. First run of every scenario: all units failed with equal line counts. The `Z` line hashed the slot ADDRESSES of the cache snapshots; the two children have different heap addresses. Fixed: slot numbers inside the arenas.
2. 14 to 34 % of the units of several scenarios ended in the slot cache's fatal exit `every slot is in use by one call`, cutting them short (the arms agreed on it, so they counted as passes). Causes in my world, not in a library: noise placed into a pool whose slots all belonged to the call; a RAM cache too small for the experts a call takes out plus the pending ones plus room to write down; and **the world put the same expert into the pool and the RAM cache**, which the engine never does (decode.cuh:536-540) and slotcache.h leaves undefined (acquiring a held key). Fixed (a RAM cache of at least `Np + 8` slots, pending entries capped at `Nh - Np - 2`, the invariant kept). Fatal exits are now only in Y5 (8 units, the RAM cache full of pending entries) and YF.
3. The thread statistic first counted "order differs from sorted"; replaced by "commit order differs from the job list's order" (4363 of 4496 threaded calls). A check that several reader threads were really used was added after I saw that a supply running every job on the calling thread would pass every other check.
4. The `HP` line (where each requested expert is) was added after the first full runs; an earlier full run (700 units, weaker world, before items 2 to 4) also passed.

## Not covered (stated, not claimed)

- **No engine run and no GPU run** (the cards are used by another agent). No real CUDA call, copy engine, stream ordering or event semantics: the fakes copy immediately, so "queued behind" orderings on a real stream are checked only as call order. The planner (`pend_take`, `arr_wait`, `host_acquire`) and the copy queue are modelled, not the real crates.
- The driver compares what is listed under "What is compared". Not compared: the values of `t_read` / `t_copy` / `t_ensure` (sanity only); the order in which reader jobs are TAKEN when several threads run; which thread ran which job; which of several failing jobs reports first (failure units use one reader thread; no failing job under real threads); the engine's `g_sc_calls` / other counters.
- Not exercised by the driver (the unit tests cover them): `sy_init` twice and the counter reset, `n = 0` with a NULL list, `ekey` overflow, negative `counts`, an expert number outside `0..experts-1`, a NULL pool cache, `pool.readers` > 32, a NULL slab entry in prefill. `readers` 0 and negative are exercised (Y9: -2..16).
- Real expert size (17547264 bytes), 8 cards, 93 layers at once, a real route: records are 4 to 20 KB, one card pool per unit, layers drawn from {0, 1, 5, 17, 50, 91, 92} (calls) and 1 to 3 other layers (warm entries).
- No ThreadSanitizer, Miri, loom or AddressSanitizer run (the planner crate had an ASan run of its reference; I did not do one here). The reference was not mutated and neither was the driver; mutation controls and the independent review are for the other agents.
- Sizes of the sample: 60 to 80 seeds per scenario; one set of seeds (the same in the three runs).

## Places where a mutation would be easy to miss (a note for the mutation and review agents)

1. **Which threads run, in what order jobs are taken**: a Rust supply that takes jobs in another order, or reverses the job list, passes whenever more than one reader thread runs (sorted multiset); with one thread the order is compared. The thread-usage check only demands that more than one thread took jobs in half of the big calls, not that `min(pool_readers, jobs)` threads ran, so spawning fewer threads than the original passes.
2. **`prot` order** passed to `sc_acquire` (membership only; the log has the count, the distinct count and a set hash, so a missing de-duplication or a wrong set is caught, a different order is not).
3. **`t_read` / `t_copy` / `t_ensure` values**: only `t_ensure >= t_read + t_copy`, zero when nothing was needed, and non-negative. A mutant that adds the right interval to the wrong field passes.
4. **`dk` / `dh`** (the RAM cache's own victim of a write-down) and the initial values of `vk` / `vh`: not observable through the ops table.
5. **Failure injection** is the 1st to 3rd occurrence of ten op kinds with one reader thread; a mutant that only breaks the error path of a call made after the third occurrence, or only under several threads, is not reached.
6. `sy_init` twice, `n = 0` with NULL, wrapping of `ekey`, the first-call check "call before `sy_init`": unit tests only (the driver calls `sy_init` once per unit).
7. A mutant that changes only the memory ordering of the job counter or the `Instant` / clock source is invisible.
8. Chunk arithmetic has an independent model in the driver, but written by the same author as the extractor and the Rust code: a misreading of the original that I share is not caught by any of them (the smoke test's values were derived by hand from the original text, which is the only other check).
9. `pend_take` and `arr_wait` are modelled so that a call completes the entry at once; a mutant that calls them but ignores that they may block (the real ones spin) is not distinguishable.

## driver v2 (2026-10-25): the blind spots found by the independent mutation check

Direction (from the owner of the work): close the blind spots of `verify/sy_driver.cpp` that `verify/mutants/RESULTS.md` found (the real gaps M52, M62, M76, the weak catches M63 M45 M91 M75 M64 M65, the exit status that no log shows, a `check` that returns instead of exiting), keep the old scenarios deterministic, and prove the new driver against the verifier's own mutants. Changed: only files under `verify/` (the driver, `build.sh`, `md5.sh`, the new directory `verify/v2/`). Not changed: `src/`, `include/`, `ref/`, and the verifier's mutants, diffs and results (md5 of all `mutants/*.diff` before and after rebuilding the 98 mutants with its own `mk_mutants.py`: identical, `v2/mutants-diff-md5-before.txt` / `-after.txt`; a pointer section was appended to `mutants/RESULTS.md`). **No GPU, no CUDA call, no sudo**; nothing under `~/k3gpu`, `~/ncm` or another agent's files or processes was touched or killed; every mutant, probe and final run had `ulimit -v 3 GB` per process.

### What the driver now does that it did not do before

| blind spot (report) | mutants | change in `sy_driver.cpp` |
|---|---|---|
| exit status invisible (W1) | M67 | the child's exit handler is registered with glibc `on_exit`: the status handed to `exit()` is written into the log (last line `EXIT <status>`) and is the status the child leaves with (before: every `exit()` became 1) |
| a successful short read cannot happen (W4) | M62 M63 | the driver executable defines `pread()` (`build.sh` links with `-rdynamic`; `nm -D` shows that the C++ reference and the Rust library both import `pread@GLIBC_2.2.5`, so both bind to it). File descriptors the driver registers are virtual files: their bytes are the same pure function as the data files, a read returns a seeded short count (a function of unit seed, offset and length only, so both arms and any thread order see the same counts), a negative offset is EINVAL as for a regular file; every other descriptor goes to the kernel (`pread64`). New scenario **Y11** (80 units): short counts on 30 / 60 / 90 % of the reads; in half of the units (one reader thread) also the file cut at 30-70 % (a short read, then end of file) or errno EIO / EINVAL / ENOMEM / EBADF for every 8th or 16th record |
| EOF loop, EOF text, catches that depended on the YF mix | M64 M65 M63 | the same Y11. A call cap (more than 400,000 `pread` calls in one child = VIOLATION) ends a read loop that does not end after a fraction of a second (M64 needed the 120 s alarm per unit, 26 children at once) |
| the number of `pread` calls and of short returns | all | a `PREAD calls=<n> short=<n>` line before `EXIT`: both arms must make the same calls |
| 32-bit file offset, chunk cap 8 (W2) | M52 M42 M43 | **Y10** (40 units): records of 4,198,401 B (4 MiB + 4097) or the real 17,547,264 B, E = 896 or 700, `pread` virtual (no file of 3.8 or 15.7 GB), experts drawn from the first one whose offset `e * raw` is >= 2^31 (e >= 512 and e >= 123), 1-2 experts per call and 2-32 readers, so `n_drive = 1` with `pool_readers` 9..32 gives 8 / 7 / 9 chunks of three different sizes (528,384 / 602,112 / 466,944 B); arenas from `calloc` (zero pages), no slab, a unit stays far below 1 GB |
| thread count (W3) | M76 | **T2** (60 units) = T1 with a barrier in the fake `set_device` of a reader job: the first `min(pool_readers, jobs)` jobs of a call (one per thread: a blocked thread cannot take another) wait until all of them have started; after 10 s the unit fails with "n of m reader threads started a job". A `BAR <n>` line records the expected count |
| `check` that returns (W5) | M74 | **YG** (80 units): the injected CUDA status goes to every op kind in turn (seed % 10, 1st to 3rd occurrence, one reader thread); in half of the units the fake `check` RETURNS after printing `cuda error ..` instead of exiting: both arms must `exit(1)` at once, with no further call |
| `check` status | M75 | the same YG (80 units over 10 op kinds, instead of the 19 YF units that reached an injected status) |
| forced chunk count 1 | M45 | **Y12** (60 units): `chunks = 1` in 70 % of them, 1-4 experts, 4-16 readers (the computed count would be above 1) |
| NULL RAM cache | M91 | **Y13** (60 units): a NULL RAM cache with capacity 0 or with prefill |
| second `sy_init` | M93 | **Y14** (40 units): `sy_init` again at half of the unit, the counter must read 0 afterwards (`dY reinit wbs <before> -> <after>`) |

The default run is now 18 scenarios, 1120 units (the 11 old ones, 700 units, and Y10 40, Y11 80, Y12 60, Y13 60, Y14 40, YG 80, T2 60). `make_cfg` draws of the 11 old scenarios and their random streams are unchanged: `v2/cmp_old_new.sh 5` ran the ORIGINAL driver (`v2/orig/sy_driver.cpp`, md5 390b6cae...) and the v2 driver on seeds 1-5 of Y1..YF and T1 against the same libraries: 120 kept logs (reference and Rust arm of every unit, four per T1 unit) are byte-identical once the `PREAD` and `EXIT` lines that v2 appends are removed, 0 different, 0 missing (`v2/logs/cmp-old-new.txt`). The old scenarios therefore produce the same canonical output as before, for both arms.

### Proof that the new driver can fail and does not fail on the unmodified crate

**(a) The unmodified crate passes the whole v2 driver.** `v2/final_v2.sh` (log `v2/logs/chain-final.out`): `cargo build --release` (nothing to do), `cargo test` **25 passed, 0 failed**, the Rust smoke test **PASS (73 checks, 0 failed)**, the driver built without a warning, then three full runs `v2final-1..3` under `ulimit -v 3 GB`, each **1120 of 1120 units PASS, exit status 0, no retry**:

| scenario | units | pass | end in a fatal exit 1 (as they must) | logged lines, reference = Rust |
|---|---|---|---|---|
| Y1 | 60 | 60 | 0 | 62,475 |
| Y2 | 60 | 60 | 0 | 102,992 |
| Y3 | 60 | 60 | 0 | 119,253 |
| Y4 | 60 | 60 | 0 | 102,672 |
| Y5 | 80 | 80 | 8 (slot cache) | 91,605 |
| Y6 | 60 | 60 | 0 | 93,140 |
| Y7 | 60 | 60 | 0 | 51,570 |
| Y8 | 60 | 60 | 0 | 114,830 |
| Y9 | 60 | 60 | 0 | 127,326 |
| YF | 80 | 80 | 68 (slot cache 2, pread 47, cuda 19) | 25,569 |
| T1 | 60 | 60 | 0 | 236,699 (4,496 threaded ensures; commit order differed from list order in 4,356 / 4,337 / 4,342 in the three runs) |
| Y10 | 40 | 40 | 0 | 13,041 |
| Y11 | 80 | 80 | 40 (pread: end of file or errno) | 81,818 |
| Y12 | 60 | 60 | 0 | 47,190 |
| Y13 | 60 | 60 | 0 | 67,571 |
| Y14 | 40 | 40 | 0 | 55,078 |
| YG | 80 | 80 | 79 (cuda; one unit never reached its injected occurrence) | 6,442 |
| T2 | 60 | 60 | 0 | 248,205 (4,860 threaded ensures; 4,786 / 4,774 / 4,787 with another commit order) |

The three runs gave identical lines except the T1 / T2 interleaving counts (real thread timing). The logged-line counts of the 11 old scenarios are exactly the ones printed for the original driver in the table above (Y1 62,475, Y2 102,992, ...). Runs of the unmodified crate on the final driver, all 1120 / 1120: the campaign's control M00, the three final runs, the repeats of M00 (2 x 1120 / 1120120120), and a load test, `v2/stress_v2.sh`: **three full runs at the same time** (`JOBS=32` each, load average 175 at the end, 21-22 s per run): 3 x 1120 PASS, no barrier timeout.

**The interposed `pread` fires in both arms.** Last lines of the logs of Y11 seeds 2-11 (units run again on their own with their logs kept; reference and Rust arm print the same `PREAD calls=<n> short=<n>`), `short` = reads that returned fewer bytes than the file had to give: u2 `calls=3 short=1 EXIT 1` (a short read, then end of file: the fatal path), u3 `430 / 361 EXIT 0`, u4 `718 / 589 EXIT 0`, u5 `318 / 185 EXIT 0`, u7 `7 / 2 EXIT 1` (errno EBADF after two short reads), u8 `1029 / 850 EXIT 0`, u10 `415 / 129 EXIT 0`, u11 `50 / 33 EXIT 1` (errno EIO); in each of the ten units both arms are equal, and in the units without a cut or an errno one unit makes up to 1029 `pread` calls (the call cap is 400,000). u6 and u9 (cut at 30 / 50 %) ended in the fatal "end of file" at the first read of a record beyond the cut. In the kept seed-1 logs of the other 17 scenarios `short` is 0 (they read real files) and the largest `calls` is 390; Y10 seed 1: `calls=92 short=0` in both arms (every read at an offset above 2^31 went through the virtual file without an error).

**(b) The mutants of the real gaps and of the weak catches now fail** (rebuilt with the author's `mk_mutants.py`, one full v2 run each, `v2/logs/table-v2.md` has all 97). "first failure" = the first differing line of the reference log / of the Rust log of the lowest seed of the first failing scenario (the `.diff` of that unit; `EXIT` and `PREAD` are the new lines):

| mutant | change | failing units / 1120 (before: of 700) | per scenario | first failure (ref / rust) |
|---|---|---|---|---|
| M52 | job file offset: `e * raw` as a 32-bit multiply | 40 (0) | Y10=40 | Y10 1: ref `CFG scn=Y10 seed=1 E=896 raw=4198401 ...` / rust `reading expert 787 of layer 1 failed: Invalid argument` (the offset wraps negative, as in the verifier's demo with a real sparse file) |
| M62 | pread loop: `got = r` instead of `got += r` | 61 (0) | Y11=61 | Y11 2: ref `reading expert 16 of layer 1 failed: end of file` / rust `VIOLATION pread called 400001 times: a read loop that does not end` |
| M76 | one reader thread fewer | 60 (0) | T2=60 | T2 1: ref `BAR 8` / rust `VIOLATION reader barrier timed out: 7 of 8 reader threads started a job ...` |
| M63 | pread loop: offset without `got` | 59 (2) | YF=2 Y11=57 | YF 26: ref `reading expert 0 of layer 17 failed: end of file` / rust `VIOLATION reader copy 0 differs from the model: got [J d=0 dst=P1 n=12..` |
| M45 | `cfg.chunks > 1` instead of `> 0` | 44 (6) | Y4=1 Y6=2 Y7=3 Y10=1 Y11=1 Y12=34 Y14=2 | Y4 46: ref `J d=1 dst=P3 n=5000 ... k=0 ...` / rust `VIOLATION 2 reader copies recorded, the model expects 1 (n_drive=1, ch..` |
| M91 | `host_find` without the NULL guard | 70 (10) | Y5=10 Y13=60 | Y5 30: rust log empty (the child dies of SIGSEGV; a crash, not a difference of content) |
| M75 | `ops.check` always gets status 0 | 98 (19) | YF=19 YG=79 | YF 4: ref `cuda error 705 at cudaSetDevice(dev)` / rust `PREAD calls=2 short=0` (no message: the status never reached `check`) |
| M64 | `r < 0` fatal, `r == 0` not (spins on EOF) | 55 (31) | YF=31 Y11=24 | YF 1: ref `reading expert 44 of layer 92 failed: end of file` / rust `VIOLATION pread called 400001 times: a read loop that does not end` |
| M65 | EOF text `EOF` instead of `end of file` | 55 (31) | YF=31 Y11=24 | YF 1: ref `... failed: end of file` / rust `reading expert 44 of layer 92 failed: EOF` |
| M67 | fatal exit status 2 instead of 1 | 87 (0) | YF=47 Y11=40 | Y11 1: ref `EXIT 1` / rust `EXIT 2` |
| M74 | no exit when `check` returns with a status | 38 (0) | YG=38 | YG 7: ref `cuda error 707 at cudaMemcpyAsync(dev_dst, b, ...)` then `EXIT 1` / rust goes on: `VIOLATION reader copy 0 differs from the model` |
| M42 | `nchunk` capped at 7 instead of 8 | 36 (0) | Y10=36 | Y10 1: rust `VIOLATION 7 reader copies recorded, the model expects 8 (n_drive=1, ...)` |
| M43 | `nchunk` capped at 9 instead of 8 | 36 (0) | Y10=36 | Y10 1: rust `VIOLATION 9 reader copies recorded, the model expects 8 (n_drive=1, ...)` |
| M93 | second `sy_init` does not reset the counter | 22 (0) | Y14=22 | Y14 1: ref `dY reinit wbs 8 -> 0` / rust `dY reinit wbs 8 -> 8` |
| M00 | control, unmodified copy | 0 (0) | - | - |

I read the raw `.diff` of the first failure of M52, M62, M76, M74, M67 (the lines above) and the first failures of M93; the other lines are copied from the generated table.

**(c) All 97 mutants of the verifier run again on the new driver** (`v2/run_set.sh` = the author's `mutants/run_all.sh` with `ulimit -v 3 GB`; 98 runs of 1120 units; the `.so` md5 of every mutant equals the one of the first pass, so the toolchain did not change). `v2/compare_v2.py` (reads `mutants/logs/table.md` of the first pass, read only):

- caught before: **81**; caught now: **89**.
- caught before and not caught now: **none**.
- not caught before, caught now (8): **M42 M43 M52 M62 M67 M74 M76 M93**.
- not caught before and not caught now (8): **M16 M26 M41 M57 M81 M90 M92 M97**, the ones the verifier called equivalent; 0 of 1120 units each, also in the two repeats.
- control M00: 0 failing of 1120. Every one of the 98 runs has 1120 UNIT lines, no duplicate.
- every one of the 81 mutants caught before has MORE failing units now (none fewer; `v2/counts_v2.py`); the lowest failing counts of any caught mutant are now M93 22, M42 36, M43 36, M74 38, M52 40, M45 44 (before: 2 units for M63, 6 for M45, 10 for M91). The per-mutant counts are in `v2/logs/table-v2.md` next to the first pass's `mutants/logs/table.md`.
- first complete pass (`v2/logs/pass-v2.0/`, on the driver before the `PREAD` line was added): the same verdicts (89 caught, 8 newly caught, none lost).

Extra controls of this pass (mine, not part of the 97; `v2/mk_extra.py`): X1 (32-bit unsigned product for the file offset) 13 of 1120 failing (Y10=13, the real-size records with e >= 245); X2 (a short read accepted as the whole read) 63 (Y11=61, YF=2; first failure `PREAD calls=3 short=1` vs `calls=2 short=1`); X3 (the calling thread numbered 1) 448.

**(d) The probes again** (`v2/probes_v2.sh`: the verifier's probe builds, a full v2 run each, counts of how often the changed path / the differing state is reached; `v2/logs/probes-v2/`). First pass (700 units) -> v2 (1120 units):

| probe | first pass | v2 |
|---|---|---|
| P52 (32-bit offset): jobs where the 32-bit product differs from the 64-bit one; largest `e * raw` | 0 of 85,409; 3,665,920 | 368 of 145,324; 14,599,323,648 (2^31 = 2,147,483,648) |
| P42 / P43 (chunk cap): calls with ratio >= 8 where 8 and 7 / 8 and 9 chunks give different sizes | 0 / 0 (941 calls) | 36 / 49 and 25 / 36 (1340 / 1353 calls) |
| P62 (short read): reads that returned fewer bytes; loop iterations with `got > 0` | 2; 0 | 3,376,688; 24,399,776 (the mutant loops until the call cap in 63 processes, so these two counts are inflated by the mutant itself; what matters is that the state is reached) |
| P67 / P74 / P93 | 47 fatal exits; `check` returned with a status: 0; counter non-zero at an init: 0 | 87 fatal exits; 38; 22 |
| P76 (threads): `run_parallel` calls with n >= 2 / n >= 3 | 9,598 / 6,562 of 12,098 | 13,300 / 8,980 of 17,091 |
| **equivalent ones** P16: `*vkey` untouched after `sc_acquire` | 0 of 93,819 | 0 of 152,801 |
| P26 (`continue` vs `break`; also a byte-identical `.so`) | reached 15,780 times | 26,289 |
| P41: `n_drive == pool_readers`, alternative nchunk != 1 | 0 of 1,124 | 0 of 1,529 |
| P57: `pool_readers < 1` (jobs > 0) | 248 (196) | 248 (196) |
| P90: `host_worth` with a NULL RAM cache | 0 of 74,420 | 0 of 120,518 |
| P92: `pool_find` with a NULL pool cache | 0 of 182,487 | 0 of 279,006 |
| P97: `sy_ensure` with `n = 0` / with NULL experts | 0 / 0 of 19,479 | 0 / 0 of 28,677 |

The eight equivalent mutants are still equivalent on the larger driver: the changed state is either never different (P16, P41, P57, P26 and the byte-identical M26, M81) or never reached (P90, P92, P97: out of contract). The probe event files were deleted after their summaries were written (the one of P62 was 1.5 GB, mostly endless-loop lines); `v2/logs/probes-v2/` keeps the summaries, the build output and the probe diffse summaries, the build output and the probe diffse summaries, the build output and the probe diffs.

**Repeats** (`v2/rerun_v2.sh`: two more full runs of each, one at a time, 3 GB cap; `v2/logs/reruns-v2.txt`, 52 runs): the control M00 0 of 1120 both times; M42 36, M43 36, M45 44, M52 40, M62 61, M63 59, M64 55, M65 55, M67 87, M74 38, M75 98, M76 60, M91 70, M93 22 failing units, **identical in every run** (campaign and both repeats); the eight equivalent mutants (M16 M26 M41 M57 M81 M90 M92 M97) 0 of 1120 both times; the extra controls X1 13 and X2 63 both times, X3 442 / 444 / 448 (its count depends on thread timing). Nothing was flaky among the mutants the pass is about. (`v2/rerun_v2.sh`: two more full runs of each, one at a time, 3 GB cap; `v2/logs/reruns-v2.txt`, 52 runs): the control M00 0 of 1120 both times; M42 36, M43 36, M45 44, M52 40, M62 61, M63 59, M64 55, M65 55, M67 87, M74 38, M75 98, M76 60, M91 70, M93 22 failing units, **identical in every run** (campaign and both repeats); the eight equivalent mutants (M16 M26 M41 M57 M81 M90 M92 M97) 0 of 1120 both times; the extra controls X1 13 and X2 63 both times, X3 442 / 444 / 448 (its count depends on thread timing). Nothing was flaky among the mutants the pass is about. (`v2/rerun_v2.sh`: two more full runs of each, one at a time, 3 GB cap; `v2/logs/reruns-v2.txt`, 52 runs): the control M00 0 of 1120 both times; M42 36, M43 36, M45 44, M52 40, M62 61, M63 59, M64 55, M65 55, M67 87, M74 38, M75 98, M76 60, M91 70, M93 22 failing units, **identical in every run** (campaign and both repeats); the eight equivalent mutants (M16 M26 M41 M57 M81 M90 M92 M97) 0 of 1120 both times; the extra controls X1 13 and X2 63 both times, X3 442 / 444 / 448 (its count depends on thread timing). Nothing was flaky among the mutants the pass is about.

**(e) Toolchain and the final md5** (`v2/final_v2.sh`, `md5.sh`; box `gpu`, 2026-10-25 07:04 UTC): `rustc 1.95.0 (59807616e 2026-04-14)`, `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`, `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`. After `cargo build --release` (the crate's `src/`, `Cargo.toml`, `include/supply.h` unchanged: md5 as in the table above): **`target/release/libsupply.so` md5 `b0288adb5dcdf7c74989f46688826ebc`**, the same as in the verifier's report; a copy built in a new directory by `mk_mutants.py` (M00, rebuilt twice in this pass: `v2/build-weak.out`, `v2/build-all.out`) has the same md5. The `f77f2b73...` printed in the table above for the first cycle is therefore not reproduced by this toolchain and the sources of that table (still unexplained; I did not investigate). Other md5: `ref/libsupply_ref.so` `7532a0ef7291735fb1dd5a1d53984390` (unchanged, not rebuilt), `../slotcache/target/release/libslotcache.so` `536470b28584426b4c60d0e7390ab4fd`, `verify/sy_driver.cpp` `5aac6c78ce2f2ff6435446f34ddf7169` (CR removed; the Windows copy has CRLF), `verify/sy_driver` `ca9410d050d3289fc59aefbd3096543f`.

### Not closed, and limits of the new checks

- **Eight mutants stay uncaught on purpose**: M16 M26 M41 M57 M81 M90 M92 M97 are equivalent (M26 and M81 have a `.so` byte-identical to the control: md5 `b0288adb...` in every build; the others were argued and probed by the verifier). They were run again on 1120 units (0 failing, in the campaign and in the repeat runs) and their probes again on v2 (above).
- **Caught by a crash, not by a difference of content**: M91 (Rust child killed by SIGSEGV, empty log; M36 M37 M80 as before). A crash is a failure of the unit, but the log does not say why.
- **M62 and M64 are caught by the `pread` call cap**, a rule of the driver that `supply.h` does not state (a correct library never calls `pread` more often than reader jobs plus short returns; the cap is 400,000 per unit; the largest count of the control is in the `PREAD` lines of the final runs, see above).
- **Y10 uses a virtual `pread`**, not a 3.8 GB or 15.7 GB file: it models what the kernel does (EINVAL for a negative offset, end of file, short counts). The verifier's `mutants/demo/m52_real_size.c` ran the same mutant against a real sparse file and the real kernel and got the same `Invalid argument`; the old scenarios still read real files. Y10 runs without prefill (a 4-record slab of 17.5 MB per layer would make the units heavy; prefill is covered by Y6 at small sizes).
- **T2's barrier is timing based** (10 s timeout). 0 false timeouts in all runs of the unmodified crate (the three final runs, the three-at-once load test, the repeats, the campaign); on a much slower or more loaded machine a correct crate could fail it. It pins the number of threads that start, not which job each runs or in which order jobs are taken.
- **Not covered, unchanged** (see "Not covered" above): a real GPU and real CUDA streams, 8 cards, the real planner, memory-safety tools (no ASan, TSan, Miri, loom), mutations of the reference or of the driver itself, two-site mutations, out-of-contract inputs (W6: NULL pool cache, `n = 0` with a non-NULL list), the order in which reader jobs are taken when several threads run.
- The three extra controls X1-X3 are mine, not the verifier's (X1: 32-bit UNSIGNED product for the file offset, which differs from the 64-bit one only for `e * raw >= 2^32`, i.e. real-size records with e >= 245; X2: a short read accepted as the whole read; X3: the calling thread numbered 1). They are listed separately and are not part of the 97.

### Files

Changed or new under `verify/`: `sy_driver.cpp`, `build.sh` (`-rdynamic`), `md5.sh` (prints rustc / cargo / g++ versions), `v2/` (`run_set.sh`, `rerun_v2.sh`, `final_v2.sh`, `stress_v2.sh`, `chain_v2.sh`, `probes_v2.sh`, `cmp_old_new.sh`, `compare_v2.py`, `mk_extra.py`, `orig/` = the driver source and `build.sh` as they were, `logs/`: `table-v2.md` (all 97 mutants), `compare-v2.txt`, `cmp-old-new.txt`, `reruns-v2.txt`, `probes-v2/`, `pass-v2.0/` = the first complete campaign on the driver before the `PREAD` line was added (same verdicts), `chain-*.out`, `X*.diff`). The author's scripts `mutants/mk_mutants.py` and `mutants/run_all.sh` were used unchanged; `v2/run_set.sh` only adds `ulimit -v 3 GB` in front of `run_all.sh`; the probe scripts were used through copies that write to `v2/logs/probes-v2/` (so `mutants/logs/` is untouched).
