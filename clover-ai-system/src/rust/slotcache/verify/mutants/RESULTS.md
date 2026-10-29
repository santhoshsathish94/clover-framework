# SlotCache driver mutation campaign (agent C2, 2026-10-22)

Question: can `verify/diff_driver` FAIL? Method: 16 single-edit mutants of `src/lib.rs` (each a separate crate copy under `~/rustmig/slotcache/verify/work/mut/Mn/`, diff in `Mn.diff`), each run with `--opsmul 0.1 --jobs 40`, default seeds, all scenarios, against the C++ reference `.so`. M0 is an unmodified copy (control): its `.so` has the same md5 as the real one (cb19f0b3...), so a different md5 means a different build. Control: the real crate in the same quick mode passes everything (`REAL-quick.stdout.txt`).

Every mutant: build_rc 0, exactly 1 hunk in lib.rs, Cargo.toml / Cargo.lock / tests.rs identical to the real ones (`build-all.txt`).

"first diff" below = the failing unit with the lowest op index in that run (units run in parallel; op index counts the unit's own operations including its setup). Raw text for every mutant: `summaries-quick.txt`; raw files on the box `~/rustmig/slotcache/verify/out/mut/<Mn>-quick/fail-*.txt` and `cases/`.

| M | change | .so md5 | quick result | first diff (scenario seed op, operation, what differed) |
|---|---|---|---|---|
| M1 | heat factor 0.8 -> 0.9 | 1621cce8 | CAUGHT | S6s seed 16 op 7, sc_touch(key 3, w=88): lru entry 0 differs (ref heat 107.199997/42d66666) |
| M2 | heat window 48 -> 47 | b5b3d002 | CAUGHT | S4 seed 44 op 323, sc_acquire(key 87): vkey 14 vs 3, vheat 3f000000 vs 3f666667 |
| M3 | heat tie strict `<` -> `<=` | 8db9ab59 | CAUGHT | S4 seed 33 op 25, sc_acquire(key 3): vkey 5 vs 11 |
| M4 | belady victim scans smallest pair | c54d9c2a | CAUGHT | S3 seed 23 op 12, sc_acquire(key 2): vkey 3 vs 1, vheat 42a09249 vs ff800000 |
| M5 | worth `<` -> `<=` | 2fbc9684 | CAUGHT | S5 seed 64 op 13, sc_worth(key 19): ival 0 vs 1 |
| M6 | try_acquire `>` -> `>=` | 5cceee37 | CAUGHT | S3 seed 17 op 11, sc_try_acquire(key 2, cand_nu=1000000000): ref NULL, mutant slot #1 vkey 3 |
| M7 | belady touch does not re-insert pair | 394ebf90 | CAUGHT | S3 seed 23 op 8, sc_touch(key 1): ordered-set count 1 vs 0 |
| M8 | remove does not erase ordered-set pair | 9a02dec6 | CAUGHT | S4 seed 62 op 9, sc_remove(key 54): ordered-set count 1 vs 2 |
| M9 | belady victim ignores prot (acquire only) | 7dc53f5f | CAUGHT | S3 seed 51 op 9, sc_acquire(key 1, prot {2}): vkey 3 vs 2; also S9-f11 (ref exit 1, mutant exit 0) |
| M10 | find does not move to front | 8adaca60 | CAUGHT | S2 seed 34 op 8, sc_find(key 6): lru entry 0 differs |
| M11 | spares FIFO (`remove(0)`) | f7890353 | CAUGHT | S6s seed 18 op 5, sc_acquire(key 15): slot #2 vs #1 |
| M12 | heat update with `f32::mul_add` | cb8bd42c | CAUGHT | S2 seed 13 op 8, sc_touch(key 2, w=13.333333): heat 30.9333344/41f77778 vs 30.9333324/41f77777 (1 ulp) |
| M13 | acquire: no next-use call, nu=0, no pair (this also changes try_acquire, which calls acquire) | a453bd8e | CAUGHT | S4 seed 26 op 5, sc_try_acquire(key 2): next-use call logs differ (ref 1 call, mutant 0) |
| M14 | protected keys in the scan still count toward the window | a94a3389 | CAUGHT (see note 1) | S6L seed 19 op 4064, sc_acquire(key 2221, 32 prot keys): vkey 2408 vs 762 |
| M15 | fatal message `slots)` -> `slotz)` | 013bc87b | CAUGHT only by S9/Q5 (note 2) | S9-f1: stderr differs at byte 61; 12 of 14 S9 cases + Q5a-1 FAIL |
| M16 | belady() uses `policy >= 2` instead of `== 2` | 41985c4d | NOT CAUGHT, quick AND full | none |

Notes
1. M14: besides 5 mismatch units (S6L x2, S8b-p1-plan, S8c x2) all 40 workers died with the contract's fatal message (the mutant exits where the reference does not), so S1-S5 report no verdict at all and S7 shows "units=0 ... a unit did not finish: worker died". Confirmed with an S7-only run (8/8 workers died; real crate passes S7 64/64 with the same flags). The driver does flag it (FAIL), but as a crash, not as a located mismatch.
2. M15: S1-S8 all PASS (0 failing units); the random scenarios cannot see stderr or exit status because the fatal path would end the worker; only the child-process cases S9/Q5 compare them.
3. Timing: S8 is not scaled by `--opsmul`, so a quick run costs about 21-23 min unless the mutant already fails in an S8 belady unit (about 4.5 min), or fails almost at once (0-15 s for M10, M11).

## M16 (my own mutant) and why the driver is blind to it
Contract (slotcache.h): policy != 0 uses the heat rule; belady needs policy == 2 AND a next-use function. M16 makes every policy >= 2 behave like 2. The driver only ever calls sc_set_policy with 0, 1 or 2: diff_driver.cpp:386 (`pol_fixed`, 0..2) and :401 (`R.u(3)`), dd_engine.inc:208 (`cf.policy`, 0..2), and every S9/Q5 case in dd_cases.inc (`t.policy(0|1|2)`). So no operation ever runs with a policy value for which M16 differs.
Full default run (`out/mut/M16-full`, `./run.sh`, 1326 s): S8 11/11, S1-S7 all units PASS, S9 14/14, Q5 13/13, exit 0, stderr empty (`M16-full.stdout.txt`). The real crate and the reference agree with each other on policies -1, 3, 7, INT_MAX, INT_MIN in my own probe (`policy_probe.cpp`, not part of the driver): with 3 entries (heats 5, 1, 9), a function set, policy 3: reference and real victim key 2 (heat 1); M16 victim key 3 (heat 9), and worth(77) 1 vs 0. Policies 7 and INT_MAX behave the same way; -1 and INT_MIN do not separate M16 (both stay on the heat rule).
Proposed scenario (driver NOT edited): S10 = the S5 mix (policy switching, callback flipping, small and normal caches) where the policy draw is taken from {0, 1, 2, 3, 7, -1, INT_MAX, INT_MIN} instead of R.u(3), with the driver's own bookkeeping (`policy`, `acq_ok`, `try_ok`) treating "belady" as `policy == 2 && cb`; plus one Q5-style transcript case equal to my probe (policy 3, callback set, three entries with heats 5/1/9, one acquire: the victim must be the lowest-heat entry). Cost is one more Cfg entry and one case.

## Not covered
- Single-edit mutants only; no combination mutants. No mutants in the FFI shims (`report`, null-pointer tolerance, snapshot functions). The driver never passes NULL out-pointers or a NULL cache, so mutants there would be invisible to it by construction (not tested).
- Which scenario failed "first" in wall-clock terms is not known; I report the lowest op index.
- `summary.txt` written by the driver contains repeated copies of its blocks (also in the earlier `out/final-1`); I used `.stdout` as the source.
