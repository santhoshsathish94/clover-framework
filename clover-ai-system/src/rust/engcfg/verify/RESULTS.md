# engcfg verification results

Written 2026-10-25. Box `gpu` (Ubuntu 24.04, glibc 2.39, g++ 13.3.0, cargo 1.99.1), no GPU, no CUDA call, no engine run. Reference = the engine's own text at 4d8d891 behind the contract (`ref/`), candidate = `target/release/libengcfg.so`. Everything below comes from one run of `verify/all.sh` (raw log: `verify/box-out/all.log`, per-unit log: `verify/box-out/units.log.gz`); nothing here is claimed beyond what the driver compares (see "What this does not show").

## Hashes (the run depends on exactly these)

| file | md5 |
|---|---|
| include/engcfg.h | 2151618f63a3cfb708df877298a59ceb |
| Cargo.toml | ef7005519a50904f4be3a1df11a227e8 |
| src/lib.rs | a721ae5ad5b1ae7d67484f9141384192 |
| src/tests.rs | 1f06242c037c7a219099c939709c2903 |
| smoke_rust.c | 9389244bc1e26f327aa7fa35b06aa5e1 |
| ref/extract.py | 970d2911adcb75a801e81d93cfa50b44 |
| ref/engcfg_ref.cpp | eea9881cda223f26673f73a102201c6b |
| ref/ecref.h | 38fd3b86cead7ac08d62b436f3b8ac0e |
| ref/smoke_ref.c | f9764d22ca9d5f791e4af54224274fc3 |
| ref/x/MANIFEST.txt | 70a20f1aeae11a54d89c5f962928e9fe (34 slices re-checked on the box against its sha256 and byte counts: 0 mismatches) |
| ref/libengcfg_ref.so | 2f92613d0fa5409404ce1efb5aedf100 |
| **target/release/libengcfg.so** | **b156f9c819832b1482069bdd43bb8954** |
| verify/diff_driver.cpp | 1dbc7da71b0f2430a855a9194808a418 |

The source files above are byte-identical on Windows and on the box (md5 compared). The `.sh` files are copied with their CRs removed on the box.

## Steps and raw summary lines

- Reference build (`ref/build.sh`, g++ -O2, `-Wall -Wextra`): no warning. `smoke_ref`: `PASS (61 checks, 0 failed)`.
- Rust: `cargo build --release` no warning; `cargo test`: `test result: ok. 56 passed; 0 failed`; `smoke_rust`: `PASS (61 checks, 0 failed)`. The `.so` exports exactly the 13 `ec_*` functions of the header (`nm -D`).
- Differential run, 24 jobs at `nice -n 10`, wall time about 20 s (an identical earlier run measured with `time`: real 20.6 s, user 2m05, sys 5m45, the sys time being the forks):

```
E1: 60 units PASS, 0 FAIL/CRASH, 600000 comparisons, 0 single-library checks, 300000 cases
E2: 40 units PASS, 0 FAIL/CRASH, 2978884 comparisons, 160000 single-library checks, 160000 cases
E3: 25 units PASS, 0 FAIL/CRASH, 2430336 comparisons, 0 single-library checks, 2430336 cases
E4: 4000 units PASS, 0 FAIL/CRASH, 256000 comparisons, 4000 single-library checks, 240000 cases
E5: 100 units PASS, 0 FAIL/CRASH, 300000 comparisons, 300000 single-library checks, 300000 cases
E6: 60 units PASS, 0 FAIL/CRASH, 204146 comparisons, 0 single-library checks, 60000 cases
E7: 100 units PASS, 0 FAIL/CRASH, 800000 comparisons, 0 single-library checks, 100000 cases
E8: 1 units PASS, 0 FAIL/CRASH, 3 comparisons, 0 single-library checks, 3 cases
TOTAL: 4386 units (0 failed), 7569369 comparisons, 3590339 cases  -> all PASS
  E8 card_of layout 1, cards 0, layer 5: reference died with signal 8, candidate with signal 6
  E8 card_of layout 1, cards -1, layer INT_MIN+1: reference died with signal 8, candidate with signal 6
  E8 force_routes n = -1 reaching the vector: reference died with signal 6, candidate with signal 6
```

E1 = env_int 300000 cases, E2 = settings + pend_cap 160000, E3 = card_of (one exhaustive unit of 46153 + 24 x 100000 random), E4 = force_routes (4000 scripts x 60 calls, each script in two forked children), E5 = print_summary 300000, E6 = selectors / trace / prefill / messages 60000, E7 = composite configure() 100000 cases x 3 arms (300000 forked children), E8 = three out-of-contract deaths. "Comparisons" counts every scalar, string or byte block compared; E4's unit of 64 comparisons = 60 selected-array blocks + exit status, signal, stderr, total size.

- `count.sh` (awk over `units.log`, independent of the driver's own sums): `BEGIN lines: 4386   UNIT lines: 4386`, per-scenario units / comparisons / cases equal to the driver's, `TOTAL units=4386 cmp=7569369 cases=3590339`.
- Sanitizer arm (`asan.sh`): the reference rebuilt with `-fsanitize=address,undefined`, the same full run against the Rust build: `TOTAL: 4386 units (0 failed), 7569369 comparisons, 3590339 cases -> all PASS`; AddressSanitizer errors: 0; UBSan: 72 reports = 3 source positions x 24 processes (UBSan reports each position once per process; the 24 are the random E3 units), all `signed integer overflow` at `ref/x/card_of.inc` columns 71, 124 and 132 (`L - 1`, `g_cards - 1`, `L * g_cards`) on the extreme inputs of E3 (INT_MAX, INT_MIN, 2^30, ...): the original's undefined behaviour, which x86-64 gcc wraps and the Rust code wraps explicitly. No report from any other code, none in the children of E4-E7 (their stderr is compared, so a report there would have been a mismatch). Wall time about 3 minutes.
- Plumbing checks (`--flip N`: the N-th comparison of each unit sees a perturbed candidate value): E1 N=7, E2 N=100, E3 N=300, E4 N=5, E5 N=50, E6 N=20, E7 N=5: every scenario ended `FAIL` with a located report (first difference named) and exit status 1. (E4 with N=100 passed because a unit of E4 makes only 64 comparisons; N=5 is the one in the log.) This shows that comparison, reporting and the exit status fire; it is not a mutation control.

## What the random cases reached (branch statistics, summed over the 4386 units)

```
  E1 env_int.default_taken                                               155669
  E1 env_int.parsed                                                      144331
  E1 env_int.parsed_negative / parsed_zero / parsed_positive             29824 / 39854 / 74653
  E2 settings.cases                                                      160000
  E2 settings.routecache_on / vqueue_on / prefill_on                     98884 / 11461 / 73649
  E2 settings.policy2 / rc_policy2_vqueue_off                            34473 / 10056
  E2 settings.cards_le0 / reserve_huge / vq_cap_initial                  60969 / 45570 / 61116
  E4 force.calls                                                         240000
  E4 force.calls_that_dealt                                              66848
  E4 force.calls_that_wrote_past_n_top                                   7324
  E4 force.calls_that_left_sel_alone                                     173152
  E4 force.scripts_with_warning                                          4000
  E5 summary.cases / lines                                               300000 / 2620283
  E5 summary.experts_line / card_lines                                   233510 / 785699
  E5 summary.route_cache_line / queue_line / call_class_lines            124588 / 116554 / 459932
  E5 summary.output_has_nan / has_inf / has_negative_zero                90585 / 57225 / 77370
  E5 summary.output_over_1000_bytes                                      209113
  E6 messages.cases / null_routecache / queue_on                         17927 / 1826 / 8595
  E6 prefill.fatal / on_line                                             1342 / 4654
  E6 selector.ok / fatal / mode_rust                                     12036 / 12120 / 2422
  E6 selector fatal by message: "must be cpp or rust" COPYQ 1951, SLOTCACHE 1967, ROUTETABLE 1969;
     "this build has no <header>" COPYQ 1758+106, SLOTCACHE 1800+123, ROUTETABLE 1800+129 (the second number: values with a newline, classified by their last line); other 517
  E6 trace.fatal / opened / null / file_created / preexisting_content    5215 / 5038 / 1668 / 4173 / 1669
  E7 composite.cases                                                     100000
  E7 composite.exit0 / exit1                                             52881 / 47119
  E7 composite fatal by message: PREFILL needs route cache 17094; cannot write (trace) 9864;
     must be cpp or rust COPYQ 3166, SLOTCACHE 5444, ROUTETABLE 5093; this build has no <header> COPYQ 1166+38, SLOTCACHE 2062+65, ROUTETABLE 1873+68; other 1186
  E7 composite.route_cache_line / prefill_on_line / rust_announce_lines  49526 / 14952 / 24241
  E7 composite.cpp_selector_lines / start_line / exit_summary_present    206050 / 52881 / 52881
  E7 composite.trace_opened                                              10617
```
(The complete list, including the per-kind selector keys, is in `box-out/diff-stdout.txt`.)

Reading: a branch that these counters show as never reached would be untested; the table is meant to be checked against the list of branches in `include/engcfg.h`. Reached with counts in the thousands or more: unset/empty/parsed values of env_int with all three signs; route cache on/off, policy 2 with the queue on and off, prefill on, `cards <= 0`; the three exits of `force_routes` and the dealing (66848 calls dealt, 7324 of them wrote past `n * top` through a compensated negative count); every summary line kind, NaN / inf / -0 in the output, outputs longer than 1000 bytes (huge doubles); selector values of every kind in both build variants (12120 fatal, 12036 ok, 2422 `rust`); trace paths that open, create, fail, or truncate an existing file with content (1669); the composite `configure()` ended in exit status 0 in 52881 cases and in a fatal exit in 47119 (PREFILL without table 17094, trace 9864, selectors the rest), PREFILL ON line in 14952, route cache line in 49526, `rust` announce lines in 24241, trace stream opened in 10617.

## What this does not show

- Equivalence outside the generators: values the generators never produce (listed below), libcs other than glibc 2.39, locales other than C, 32-bit or big-endian, several threads, stderr that is closed, full or a pipe (the driver captures into files).
- The real engine: no CUDA, no `dlopen`, no route-cache file, no real `g_host.init`; the call order is validated only against the original `configure()` text with stand-ins (E7 arm A), which the driver's arm B (the same contract calls on the reference library) also agrees with.
- The reference is the original's text behind a wrapper, not an independent implementation; where the wrapper adapts the text, `ref/NOTES.md` lists it (ADAPTED 1-13). A wrong adaptation would be shared by both arms of a comparison only if it is also in the Rust code, which was written from the header, not from the wrapper.
- The two out-of-contract classes (E8) are only checked to die in both libraries; `ec_print_summary` with `cards > 8` and a `component` outside 0..2 are not driven at all.

## History of the driver (what changed after the first runs, and why)

- First small run (`--scale 0.02`): all PASS, but the branch counters showed that E7 ended in a fatal selector error in 33 of 40 cases and E6 selectors were fatal in 11 of 13; selector values were re-weighted (E7 mostly unset / `cpp` / `rust`) so the later steps of `configure()` are reached. Branch counters (STAT lines) and the pre-existing trace file case and the `need`-wraps-to-0 summary case were added for the same reason: an earlier full run passed over inputs that never reached those branches.
- The atexit marker in every child (to tell `exit` from `_exit`) was added after several full runs; they and the final one give identical totals.
- An earlier `all.sh` run only failed its `smoke_rust` step, because of a wrong rpath in the script (fixed); the smoke test passed by hand and in the final run.

## Mutations a driver could easily miss (not written; for the mutation agent)

1. **Rounding-neutral rewrites of the floating-point expressions.** `1000 * t / n` as `t / n * 1000.0` or `1000.0 * (t / n)`, `100.0 * hc / need` as `hc / need * 100.0`: results differ by at most an ulp and are printed with 1 or 3 decimals, so the output changes only when the value sits within an ulp of a decimal rounding boundary; the generators put near-ties in the inputs (`t_calls`, `t_read`, ...) but not in these quotients. Treat them as probably missed.
2. **Integer width of `force_routes` internals.** `total` accumulated in `i32` (needs counts summing past 2^31), `s` (the dealing counter) in `i32` (needs more than 2^31 deals), `n * top` in `i32` (n above 1.3e8), `fill` as `u8` / `u16` (needs 256 / 65536 deals to one token): the largest n in the driver is 200 and the compensated-negative cases add at most about 300 deals, so none of these is reachable. *(Driver v2: now driven by E9 and the appended E4 calls, except the one that needs 2^31 deals to one token; see "driver v2".)*
3. **How a line reaches stderr.** One write per line vs several writes, buffered vs unbuffered: the driver compares the final bytes of a file only. Behaviour when fd 2 is closed, full or a pipe is not driven. *(Driver v2: the candidate's writes are counted; see "driver v2".)*
4. **`exit` vs `_exit`** on the fatal paths of `configure()` is only caught by the driver's own atexit marker in every child (added to the driver for this reason; a mutant that skips only the engine's other exit-time work would still be invisible, because the driver has no such work).
5. **Locale and libc dependence**: `atoi` whitespace in a non-C locale, a non-glibc `printf`: not driven.
6. **Component values outside 0..2** for `ec_selector_*` and `cards > 8` for `ec_print_summary` (both abort in Rust, undefined or out-of-bounds in the reference): not driven.
7. **Very long environment values** (more than a few KB) and values with `=`: the generators stop at about 5000 bytes (trace path) and do not put `=` into values.
8. **`K3GPU_TRACE` naming a FIFO, a symlink or a file of another owner**; `fopen` flags beyond "w" vs "a" / "r+" (the pre-existing-content case catches those two).
9. **The unused-when-off route cache variables** (`K3GPU_LOOKAHEAD` etc. read even without a route cache): not observable from outside, equivalent in behaviour.
10. **`n` between 201 and 2^31** in `force_routes` (allocation-size behaviour of `fill`), `top` outside {1, 2, 3, 4, 8, 16, 17}.

## driver v2 (2026-10-25): the blind spots found by the independent mutation check

**Direction (the owner of the work):** close the driver's blind spots. The independent check (`verify/mutants/RESULTS.md`) ran 140 mutants of `src/lib.rs` against the first driver: 123 caught, 17 not (8 argued equivalent, 2 caught by a unit test only, 7 real gaps). Allowed to change: the driver and its scripts under `verify/`; not `src/`, `include/`, `ref/`. **Not changed:** `src/lib.rs` (md5 a721ae5a...), `src/tests.rs`, `include/engcfg.h` (2151618f...), `ref/` (`libengcfg_ref.so` 2f92613d...), `Cargo.toml`; the header sentence "one write per original line" was not edited (see the last paragraph). The box was shared with other agents' mutation and driver runs (load average 25-125); every run below ran under `ulimit -v 3145728` (3 GB) except the sanitizer arm (ASan reserves terabytes of address space; its E9 children are unbounded, and no library in that arm allocates more than about 40 MB in E9).

**What changed (files).** `verify/diff_driver.cpp` (1020 -> 1270 lines; md5 f2ab7069..., binary ec868022...), `verify/asan.sh` (`--e9-vm-mb 0`), `verify/all.sh` (E9 in the recount and in the plumbing loop), `verify/mutants/run_all.sh` (`VLIMIT_KB`, default unchanged), `verify/NOTES.md` (scenario descriptions), this file, `verify/mutants/RESULTS.md` (a section appended), evidence files under `verify/box-out/` and `verify/mutants/box-out/` (`*v2*`). New in the driver, from the existing scenarios' point of view only additions (E1-E3 compare exactly as before, same counts; E4's first 60 calls per script are generated exactly as before, with six appended):

| blind spot (mutant) | what the driver now does | how it shows |
|---|---|---|
| counts longer than the experts (M95) | E9 `morecounts`, `morecounts-sum-ok`, `counts-without-experts` (fresh process: the first warning is visible, text compared); E4 appended kinds 0-2 | M95 fails E4 4000 + E9 3 units |
| a count above n with the sum still n*top, the `fits` check (M92) | E9 `count-above-n`, `count-above-n-negative`; E4 appended kind 3 | M92 fails E4 4000 + E9 2 (before: 54 E4 units, all through n = 0) |
| `total` in 32 bits (M98) | E9 `total32` (5 counts of 2^30: the true sum 5*2^30 reads 2^30 = n*top; a candidate that accepts needs a 4 GB `fill` vector and dies under RLIMIT_AS = 2 GB, the reference and the real library warn), `total32neg` (counts INT_MIN twice, sum -2^32 reads 0 = n*top for n = 0); E4 appended kind 4 (two extra experts of count INT_MIN) | M98 fails E4 4000 + E9 2 |
| `n * top` in 32 bits (M99) | E9 `ntop32` (n 4096, top 2^20), `ntop32neg` (n = top = 46341, one count -2147479015) | M99 fails E9 2 |
| `fill` of 8 / 16 bits (M100, M101) | E9 `u8` (300 deals to one token), `u16` (70,000), `s32-heavy` (513 deals per token); E4 appended kind 5 (257-400 deals) | M100 fails E4 4000 + E9 3, M101 fails E9 1 |
| the dealing counter in 32 bits (M102) | E9 `s32-heavy`: n = 2^22, top 1, 513 experts of count 2^22 and one of INT_MIN (sum n), so 2,151,677,952 deals; two buffers of 16 MB, about 15 s per library on the box (both run at once); the reference's whole buffer is first compared with `sel[j] = 1000 + min(512, j)` derived by hand, then the candidate's with the reference's. No `mmap` / `MAP_NORESERVE` buffer was needed: counts are bounded by n, so many experts of a moderate n reach 2^31 deals, where one huge n would need a 12.6 GB buffer | M102 fails E9 1 (the mutant dies with an out-of-bounds index) |
| explicit abort checks (M84, M105) | E8 now compares the stderr text of the three deaths: the candidate must leave nothing (an explicit `abort()`; a Rust panic of the `%` or of the vector size prints its message first), the reference must leave what it is designed to leave (nothing for the SIGFPE, UBSan's `runtime error: division` line in the sanitizer build, libstdc++'s text naming `std::length_error`) | M84, M105 fail E8 1 each |
| write granularity (M138; header sentence) | the candidate's fd 2 is a `SOCK_SEQPACKET` socketpair in E4 scripts, E5, E6, E7 (message-preserving: one write is one message; `SOCK_DGRAM` has no end-of-file, which the reader needs). Every write must end a line; E5 (no user bytes in any line) requires exactly one write per line of the reference's text; E6's route-cache + start lines exactly 2 writes, the announce line exactly 1 | M138 fails E4 4000, E5 100, E6 60, E7 100 units |

The reference is **not** held to the write rule: glibc's unbuffered `fprintf` writes a line of more than 128 bytes in pieces. The driver counts it (`writes.*` statistics, sampled every 16th E5 case): 14,828 of the 18,800 sampled cases had a reference line written in several pieces (253,961 writes for 165,011 lines). Statements about the Rust library's writes are therefore statements against the header's sentence, not against the reference's behaviour.

**Runs with the final driver (unmodified crate, `libengcfg.so` md5 cb7a3cf1...):** three separate full runs under `ulimit -v 3 GB` and a fourth inside `all.sh`: each `TOTAL: 4399 units (0 failed), 8075367 comparisons, 3614352 cases -> all PASS` (E1 60 / E2 40 / E3 25 / E4 4000 / E5 100 / E6 60 / E7 100 / E8 1 / E9 13 units; comparisons 600000 / 2978884 / 2430336 / 284000 / 600000 / 282073 / 900000 / 9 / 65), wall 30-32 s per run (`box-out/v2-clean-runs.log`, `box-out/v2-all.log`). The E9 reference checks (13 of 13 `reference_as_designed`) and the E4 appended calls (kinds 0-4 left `sel` alone and kind 5 dealt in all 4000 scripts) confirm that the new cases do what they are for in the reference. The branch statistics of E1-E7 that existed before are unchanged except E4 (264000 calls, 70848 dealt, 11324 wrote past `n * top`, 4000 each of the six appended kinds). Sanitizer arm (`asan.sh`, reference rebuilt with ASan+UBSan): the same totals, all PASS, 0 AddressSanitizer errors, 72 UBSan reports (3 positions in `ref/x/card_of.inc` x 24 processes, signed overflow on the extreme E3 inputs, as before). Plumbing (`--flip N`): E1 N=7, E2 N=100, E3 N=300, E4 N=5, E5 N=50 (now a perturbed write count), E6 N=20, E7 N=5, E9 N=4: every scenario ended FAIL with a located report. `cargo test`: 56 passed (unchanged crate); `smoke_ref` and `smoke_rust`: PASS (61 checks).

**Mutants (the author's `mk_mutants.py` / `run_all.sh`, unchanged except `VLIMIT_KB`):** @@MUTANT_SUMMARY@@

**Toolchain and the `.so` md5.** @@TOOLCHAIN@@

**Not closed, and what v2 still does not show.**

- **M104** (`fill` zero-extended into the index) differs only after 2^31 deals to one token. Counts are bounded by n, so that needs n = 1 and 2^31 experts (two arrays of 8.6 GB), outside the 3 GB bound; argued equivalent below that, not driven.
- **The rejecting direction of the 32-bit mutants.** A *valid* call with `n * top` or a sum of counts above 2^31 - 1 (accepted by the reference, refused by a mutant whose product wrapped negative) needs a `sel` buffer of more than 8 GB (n * top words). v2 drives the accepting direction (the mutants that accept a call the reference rejects); the other is not reachable inside the memory bound.
- **The silent-abort pin in E8 is the crate's design, not the header.** `include/engcfg.h` says only "the Rust library aborts"; the driver now also requires the candidate to leave nothing on stderr (NOTES.md of the crate: "abort without the libstdc++ message"; the explicit checks in `lib.rs`). A change of that design must change the driver.
- **M98 `total32` is detected by an allocation failure under RLIMIT_AS (a 4 GB vector), not by a wrong result.** Other mutants of the same family are caught by `total32neg` and E4 kind 4 through their output; a mutant that accepts `total32` and has no 4 GB allocation would be caught only by the warning text.
- **Header sentence.** "one write per original line" is true of the Rust library (now checked) but not of the reference for lines over 128 bytes (14,828 of 18,800 sampled E5 cases). I did not edit `include/engcfg.h`; if the owner wants the contract to say what the reference does, the sentence under "Conventions" (line 15 of the header) could read "one write per original line, at most (the reference's glibc `fprintf` splits lines of more than 128 bytes)" -- a wording fix for the owner to request, not applied.
- Everything in "What this does not show" above still holds (other libcs and locales, closed/full/pipe stderr, threads, the real engine).
