# Mutation check of the routetable differential driver (agent C2, 2026-10-22)

Question: can `verify/diff_driver` FAIL? Method: 20 mutant copies of the Rust crate (one edit each, `mk_mutants.py`) plus an unmodified control (M0), each built with
`cargo build --release --offline` in its own copy under `~/rustmig/routetable/verify/work/mut/<name>/` on the box, and run through the unmodified `verify/run.sh`
(full settings: 20430 units = R3 9, R1 400, R2 20000, R4 21; jobs 48). Each mutant was run twice: as shipped (the driver stops scheduling at the first failure) and with `--keep-going`
(every unit runs; counts of failing units per scenario). Raw logs on the box: `~/rustmig/routetable/verify/out/mut/<name>/` and `<name>-kg/`; copies of `stdout.txt`, the first-failure
report and the build log are in `box-out/`. Diffs: `<name>.diff` (unified, against the real `src/lib.rs`, generated on the box from the LF file; every mutant diff has exactly one hunk).

## Result table

`first difference` is copied from the raw log (library 0 = reference, library 1 = candidate). `kg` = units that failed in the `--keep-going` run (of R1 400 / R2 20000 / R3 9 / R4 21).

| mutant | change | .so md5 | default run | first difference (scenario, seed, what) | kg failing units |
|---|---|---|---|---|---|
| M0 | none (control) | 5281e02b3966dc6d4e4102bb4600bf11 (= real) | PASS | - | none (20430 PASS) |
| M1 | skip `<= cur` -> `< cur` | 464f1985... | CAUGHT | R1 seed 21 session 1 op#17 (cur=0): `rt_next_use(31489)` ref 1000000000, cand 0 | R1 400, R2 17321, R3 9 |
| M2 | sentinel 1000000000 -> 999999999 | 4cada219... | CAUGHT | R1 seed 3, fresh-table queries op#5: `rt_next_use(0)` ref 1000000000, cand 999999999 | R1 400, R2 20000, R3 9 |
| M3 | pointer reset to 0 on every `rt_next_use` | b4e200bd... | CAUGHT | R1 seed 3 op#1396 (cur=-355323): `rt_next_use(451)` ref 248, cand 30 | R1 400, R2 18326, R3 9 |
| M4 | `rt_next_use` inserts (empty list, 0) entry for an unknown key | 27aa92c1... | NOT CAUGHT | - | none |
| M5 | call listing an expert twice adds its number once | af36c500... | NOT CAUGHT | - | none |
| M6 | key = `e * experts + layer` | 7165805c... | CAUGHT | R1 seed 3, fresh queries op#1: `rt_next_use(244)` ref 94, cand 1000000000 | R1 397, R2 17332, R3 9 |
| M7 | expert loop also skips tabs | 077994db... | CAUGHT | R2 seed 0 session 3 table compare: `experts list length of call 54` ref 0, cand 1 | R2 11623 |
| M8 | expert may start at `+`/`-` followed by a digit (safe variant) | 03e9a8f2... | CAUGHT | R2 seed 1 session 3: `experts list length of call 1` ref 9, cand 12 | R2 11452 |
| M8L | literal M8: a bare `+`/`-` starts an expert (no progress -> endless push); run under `ulimit -v 4 GB` | 6414f0f2... | CAUGHT | R2 seed 3 session 1: `experts list length of call 7` ref 2, cand 3; also CRASH R2 seeds 0, 1: `memory allocation of 4294967296 bytes failed`, signal 6 | kg run killed by me after 1108 units: FAIL R2 335, CRASH R2 1398 |
| M9 | ntok not skipped | 02675c65... | CAUGHT | R1 seed 3 table compare: `experts list length of call 0` ref 22, cand 23 | R1 400, R2 19250, R3 9, R4 1 |
| M10 | empty lines dropped | d6799003... | CAUGHT | R2 seed 0 session 1 (file is one `\n`): `rt_calls` ref 1, cand 0 | R2 14464, R4 1 |
| M11 | last line without newline dropped | e690beed... | CAUGHT | R2 seed 0 session 3: `rt_calls` ref 168, cand 167 | R2 13013 |
| M12 | colon without digits records no count | 36531595... | CAUGHT | R2 seed 1 session 3: `counts list length of call 9` ref 3, cand 2 | R2 5824 |
| M13 | count kept only while counts and experts are aligned | 7a483283... | CAUGHT | R1 seed 3 table compare: `counts list length of call 0` ref 11, cand 0 | R1 308, R2 11236 |
| M14 | strtol: on u64 overflow the accumulator wraps (not i32 wrap, see below) | 27e87910... | CAUGHT | R2 seed 1 session 3: `experts[4] of call 4 (layer 86)` ref -1, cand 831473745 | R2 13243 |
| M15 | `rt_set_current` stores `cur as i32 as i64` | 03abd411... | CAUGHT | R1 seed 3 op#1399 (cur=9223372036854775807): `rt_next_use(1053)` ref 1000000000, cand 382 | R1 398, R2 12083 (R3: none) |
| M16 | missing-file message without trailing newline | 663dd5c7... | CAUGHT | R4 seed 1 `random-path-1-0`: candidate stderr `cannot read route cache /nonexistent-rt-verify/...` has no final `\n` (contract check) | R4 21 |
| M17 | missing-file exit status 2 | d68163f3... | CAUGHT | R4 seed 6 `random-path-6-0`: `exit 2 ... (expected exit 1 ...)` | R4 21 |
| M18 | `rt_layer` returns layer of call i+1 | a93fc0ff... | CAUGHT | R1 seed 3 table compare: `rt_layer(call 0)` ref 71, cand 41 | R1 400, R2 19588, R3 9, R4 1 |
| **M19** | strtol whitespace set loses 0x0b (VT) and 0x0c (FF) | 55961655... | **NOT CAUGHT** | - (20430 of 20430 units PASS, both runs) | none |

Every mutant .so md5 differs from the real one (5281e02b...) except M0, whose .so is byte-identical to the real build.

## Substitutions and interpretations (named)

- M8: the literal change (`+`/`-` alone start an expert) makes `strtol` return without consuming, so the expert loop never advances and pushes forever. Built as M8L and run under `ulimit -v 4194304`
  (allocation failure aborts the worker; the driver reports a mismatch or a CRASH). M8 is the closest terminating variant (sign must be followed by a digit).
- M13: "counts only kept if their number equals the number of experts so far" implemented as `counts.len() + 1 == experts.len()` at the moment a count is read.
- M14: implemented as u64 wrap-around instead of saturation (the crate parses as i64/u64, not i32); the value later cast to i32 as before.
- M4: expressible in this representation (a `HashMap<i32,(Vec<i32>,usize)>`), see equivalence below.

## NOT CAUGHT, and why

### M19 - NOT equivalent: a real behavioural difference the driver cannot see
Probe (own tool `probe.cpp`, file `probe-in/ff.trace` = `\f5 6 7:1\n\v3 4 9\n2\f3 4\n7\v8 9 1\n`, experts=100), outputs in `probe-out/ff.*.txt`:

    reference / real crate:  call 0 layer=5 experts[1]=7 counts[1]=1 | call 1 layer=3 experts=9 | call 2 layer=2 experts=4 | call 3 layer=7 experts=9 1
    M19:                     call 0 layer=0 experts[0] | call 1 layer=0 experts[0] | call 2 layer=2 experts[0] | call 3 layer=7 experts[0]

The reference and real crate agree (`ff REAL == REF`); M19 differs. Why the driver misses it (read from `diff_driver.cpp`, not instrumented): no generator emits byte 0x0b or 0x0c.
`fuzz_byte` draws digits, space, `\n`, `:`, `-`, `+`, `\t`, `\r`, letters, NUL, 0x80-0xff and ``.,;_=x/\%``; `fuzz_token`/`NUMS`/`SPECIAL` contain tabs, CR, NUL, 0x80/0xff, signs,
but no VT/FF; R1 emits only digits, spaces, `:`, `\n`; R3 reads `p1d.trace`, which contains 0 lines with these bytes (`grep -c -P '[\x0b\x0c]'` = 0, read only). The driver's feature counters
(`FEAT_NAMES`) also have no entry for them.
Scenario that would catch it (not run, driver not edited): add `\v` and `\f` to `fuzz_byte` (about 1% each) and `fuzz_token`, and to `SPECIAL` the strings `"\f5 6 7:1\n"`, `"1\v2 3\n"`, `"1 2\f3:\v4\n"`.
Any file whose first number is preceded by FF/VT, or with FF/VT between layer, ntok and the first expert, already shows `rt_layer` / `experts list length` differences (layer 5 vs 0 above).
Scope: this is a difference for inputs containing VT/FF; the production trace has none. That is a statement about `p1d.trace` only.

### M4 - equivalent (value-wise)
Argument: the only readers of `uses` are `from_calls` (before any query) and `next_use`. An entry `(Vec::new(), 0)` takes the `Some` arm, the `while` is not entered (`*p < l.len()` is false), and the
result is `EXHAUSTED`, the same as the `None` arm; the pointer is never used. Probe: `dupunk` (unknown key 999 queried at several `cur`, then known keys) gives output identical to the reference, real crate and M4.
Not equivalent in resources: M4 grows the map by one entry per distinct unknown key queried (unbounded under a stream of unknown keys). The driver does not measure memory.
A scenario that would see it (described, not run): query 10^7 distinct unknown keys and compare the candidate's peak RSS against the reference's.

### M5 - equivalent
Argument: each per-key list is ascending, so equal numbers are adjacent. The pointer only ever stops at the first element greater than `cur`, and every element before it is `<= cur`; hence it can never stop on the
second copy of a duplicated number, and it moves past a whole run at once. The list without adjacent duplicates (M5) therefore yields the same first-greater-than-`cur` value at every step, for any sequence of
`rt_set_current` / `rt_next_use` including backward `cur`. Probes `dup` and `dup2` (call 0 lists 7 twice, call 2 lists 7 three times, queries going up and down) are identical to the reference, real crate and M5.
Driver evidence: R1 gives about two thirds of its files a duplicate probability of 0.1 or 0.3 per expert (`gen_r1`); M5 passed all 20430 units.

## Unknowns and limits

- Equivalence of M4 and M5 is an argument plus four probe sequences and 20430 passing driver units, not a machine proof.
- Whether a generator emits 0x0b / 0x0c was read from the source, not measured by instrumenting a copy of the driver (the driver was not edited).
- One trace only (`p1d.trace`). R3 did not fail for M15 (only R1 and R2 caught it).
- The `--keep-going` run of M8L was stopped by me after 102 s (memory-heavy, extra); its counts above are partial.
- The first failure of the default run is whichever worker finished first; other scenarios may also have failed in that run (the kg counts show them).
- Local mirror files `src/lib.rs` and `include/routetable.h` have CRLF line endings (md5 724f3338... and 091b071a...); the diffs and md5s here are of the box files (LF).

## Hashes (box), before and after the campaign: identical (`box-out/md5-before.txt`, `md5-after.txt`)

    70dd008600231177469d9e5523e7d6d0  src/lib.rs
    5281e02b3966dc6d4e4102bb4600bf11  target/release/libroutetable.so (real)
    09c4d2b7bd2013b24aa8d3b808fc352e  ref/libroutetable_ref.so
    929d801fffeada4a2874dfc4addb7268  include/routetable.h
    d36ae6337123aaefea336b108ccaef0c  verify/diff_driver.cpp
    765cdb23fd78b1100972e9ef539c404c  verify/diff_driver (binary)
    4c12cba1657a6101a3335e2f4d580fee  ref/routetable_ref.cpp
    ec0d604f5b393491522a168e1a1c2588  ref/routetable_extracted.inc
    7af7cfd02465aacd0b1e8f2e8b899d87  src/tests.rs
    71ccfeb5ae2b693a88500db8fba071bd  Cargo.toml
    79a02e0bcc94accc35e062491eba46ed  Cargo.lock
    c4a193101570d3207c90f85aca0d927b  verify/build.sh
    e95d5e69a08cd42ec2e2f19bad77e14b  verify/run.sh
    0a3729acb5f678ef3ba4c4aaf538afad  ~/k3gpu/results/p1d.trace

## Files here

`mk_mutants.py` (creates the mutants), `run_all.sh` (build + both runs), `collect.sh` / `collect.txt` (evidence extraction), `probe.cpp`, `probes.sh`, `probes-summary.txt`, `probe-in/`, `probe-out/`,
`md5set.sh`, `pack.sh`, `<name>.diff` / `<name>.txt`, `box-out/` (summary.txt, per-mutant stdout.txt, first-failure report, build log).
