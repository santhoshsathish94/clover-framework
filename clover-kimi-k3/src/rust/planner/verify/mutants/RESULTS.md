# Differential driver against mutants of the planner crate (agent C2, 2026-10-23)

Question: can the differential driver (`verify/pl_driver`, C++ reference `ref/libplanner_ref.so` against the Rust library, logs compared byte for byte) fail? It was run against deliberately broken copies of the crate.

## Method

- Mutant = copy of the crate under `verify/work/mut/<name>/` (Cargo.toml, Cargo.lock, src/, include/ and the three sibling include dirs), one change in `src/planner.rs` (M15: `src/abi.rs`), built with `cargo build --release`. Diff of each: `<name>.diff` here (first line is a description; the rest is a unified diff against the real file).
- M01-M21 are the 21 mutations of `~/rustmig/planner-mut.py` (read from the script by `mk_mutants.py`, all 21 applied). MA, MB, MC, MD are the four new ones; MD1 and MD2 are the two halves of MD (MD changes both places that compute `off`). M00 is an unmodified copy (control).
- Each mutant: one full run, `RUST=<mutant .so> JOBS=40 ./run.sh mut/<name> --keep-going` (`run_all.sh`), default counts P1 500, P2 500, P3 36, P4 500, P5 600, P6 100 = 2236 units, logs in `verify/out/mut/<name>/`. One run at a time, no GPU, no sudo.
- CAUGHT = at least one `UNIT <scenario> <seed> FAIL` line in `units.log`. The counts are those lines, counted by `analyze.py` and again independently with `grep -c` (identical). Every run has exactly 2236 UNIT lines, no duplicates.
- "First failure" = the lowest failing seed of the first scenario (order P1..P6) that has a failure, not the earliest in time. The two lines are copied from that unit's `.diff` (ref log, then Rust log, first differing line); where the Rust child stopped on one of the driver's model checks, its `VIOLATION` line is added (all of this is in `logs/analysis.txt`).
- Disk was at 94-98 % with other users, so the per-unit logs of these runs (raw and gzipped, 54 GB) were deleted afterwards by `cleanup_logs.py`; kept on the box (`out/mut/<name>/`): `units.log`, `summary.txt`, every `.diff` and `.res`, and the raw ref/Rust logs of the first-failing unit of each of the 25 caught mutants. The runs are deterministic apart from P6 and can be repeated with `run_all.sh`.

## Result table

| mutant | change | .so md5 | verdict | failing units / 2236 | failing per scenario | first failure |
|---|---|---|---|---|---|---|
| M00 | control: unmodified copy | d719148fa7f6299409041b0e1f2f3f07 | NOT CAUGHT | 0 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M01 | sweep: no re-look after swap (i += 1 after rel swap_remove) | d7e7dff8688a62e4daac612e9b75c44a | CAUGHT | 222 | P1=0/500 P2=148/500 P3=8/36 P4=42/500 P5=9/600 P6=15/100 | P2 4 |
| M02 | pend_cap return keeps planned_to = i + 1 | 20581db7e472d3bc7ea32dca1c87d715 | CAUGHT | 1110 | P1=339/500 P2=203/500 P3=21/36 P4=268/500 P5=279/600 P6=0/100 | P1 3 |
| M03 | skipped not counted | a5ed973b63fb28c970cb6a12e2825fce | CAUGHT | 1056 | P1=419/500 P2=217/500 P3=21/36 P4=234/500 P5=165/600 P6=0/100 | P1 2 |
| M04 | write-back threshold 9/10 | dc34a74b36a3f6448da6466df2e42d82 | CAUGHT | 7 | P1=0/500 P2=4/500 P3=0/36 P4=3/500 P5=0/600 P6=0/100 | P2 78 |
| M05 | full[card] never set | f7ce76b07f886a5732fc9966c85645dc | CAUGHT | 453 | P1=0/500 P2=265/500 P3=17/36 P4=117/500 P5=52/600 P6=2/100 | P2 3 |
| M06 | pend_take after remove in vq_issue | 81baa6c8e6076f892a63a2734dbe234d | NOT CAUGHT | 0 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M07 | vq prot without layer (raw expert numbers) | d50a2cd4f6c3336a8dac466c964faaba | CAUGHT | 610 | P1=0/500 P2=357/500 P3=18/36 P4=142/500 P5=92/600 P6=1/100 | P2 1 |
| M08 | replaced pending freed at once (no graveyard) | 8b55cc53c5c094f6f3590b4d2d3b6c00 | NOT CAUGHT | 0 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M09 | out counts every arrival | 93a01e2613f950bef478b1d761b68103 | CAUGHT | 123 | P1=0/500 P2=52/500 P3=16/36 P4=33/500 P5=22/600 P6=0/100 | P2 12 |
| M10 | cand always rt_next_use | 547eb3117183e5de8ca37489f545da68 | CAUGHT | 597 | P1=0/500 P2=350/500 P3=18/36 P4=138/500 P5=91/600 P6=0/100 | P2 1 |
| M11 | pos incremented before the plan | b2623bdf65a148ccbb384c1c1816fa46 | CAUGHT | 2209 | P1=500/500 P2=500/500 P3=36/36 P4=500/500 P5=573/600 P6=100/100 | P1 1 |
| M12 | read-ahead heat 2.0 | 74e32b698afcf4607d1b4bfb73e49a12 | CAUGHT | 1807 | P1=500/500 P2=461/500 P3=36/36 P4=476/500 P5=334/600 P6=0/100 | P1 1 |
| M13 | last = pos + lookahead (no +1) | 57296c1cfab9cdc4e4c8d3ceebbb7c02 | CAUGHT | 1105 | P1=229/500 P2=304/500 P3=29/36 P4=242/500 P5=301/600 P6=0/100 | P1 2 |
| M14 | pend_take counts used only when it waited | f90b8057fe1d7cc1827c2543cde22821 | CAUGHT | 1637 | P1=487/500 P2=444/500 P3=36/36 P4=456/500 P5=214/600 P6=0/100 | P1 1 |
| M15 | sp_config field order swapped (cards/lookahead) | d2d49367d3277439cd967a459f3ff258 | CAUGHT | 1662 | P1=391/500 P2=459/500 P3=31/36 P4=387/500 P5=392/600 P6=2/100 | P1 1 |
| M16 | sweep wb: do not erase pending entry when done | c9aebead929bd29c2a667220ce9f1f28 | CAUGHT | 210 | P1=0/500 P2=147/500 P3=6/36 P4=23/500 P5=12/600 P6=22/100 | P2 4 |
| M17 | host worth checked before pend_cap | cee6d0f7c464a79e6180eb7a9b0caf10 | CAUGHT | 1110 | P1=339/500 P2=203/500 P3=21/36 P4=268/500 P5=279/600 P6=0/100 | P1 3 |
| M18 | arr_wait does not remove the entry first (get, then remove after the spin) | 59f39d9a7928d76598f9724b482d2b79 | CAUGHT | 612 | P1=0/500 P2=326/500 P3=16/36 P4=123/500 P5=88/600 P6=59/100 | P2 1 |
| M19 | mismatch does not unset next-use | 95aaffb967736dc8998cf7b1aee31492 | CAUGHT | 794 | P1=0/500 P2=0/500 P3=0/36 P4=499/500 P5=295/600 P6=0/100 | P4 1 |
| M20 | vq victim wb inserted into wb_keys twice-order: push before acquire | d06fa25267088850897feb613951834f | CAUGHT | 188 | P1=0/500 P2=147/500 P3=6/36 P4=23/500 P5=12/600 P6=0/100 | P2 4 |
| M21 | t_plan also counts mismatch path (timer around everything) | fe563cd20fc471548d9eaac789644b88 | CAUGHT | 4 | P1=0/500 P2=0/500 P3=0/36 P4=4/500 P5=0/600 P6=0/100 | P4 70 |
| MA | vq_issue: try_acquire gets (cand_nu, key) instead of (key, cand_nu) | 0ab9b312ebc5e93b9e4f248fe68a670b | CAUGHT | 670 | P1=0/500 P2=358/500 P3=18/36 P4=143/500 P5=92/600 P6=59/100 | P2 1 |
| MB | pend_take counts pf_used also when there is no pending entry | c7d46745c14f070f08cfe5f49263921b | CAUGHT | 915 | P1=262/500 P2=216/500 P3=36/36 P4=290/500 P5=111/600 P6=0/100 | P1 2 |
| MC | card-pool prot list drops its last element | e1ae35af9d17fb72ee22bc8ef281c275 | CAUGHT | 603 | P1=0/500 P2=353/500 P3=18/36 P4=140/500 P5=92/600 P6=0/100 | P2 1 |
| MD | job off = e * expert_raw as a 32-bit multiply (both job builders) | 7f161ff4b123c9597b1d206bdedb5b3e | CAUGHT | 1868 | P1=500/500 P2=500/500 P3=36/36 P4=496/500 P5=236/600 P6=100/100 | P1 1 |
| MD1 | job off as a 32-bit multiply, read-ahead job (kind 0) only | 86426133305b35086b50a6738dc1bd68 | CAUGHT | 1780 | P1=500/500 P2=460/500 P3=36/36 P4=476/500 P5=214/600 P6=94/100 | P1 1 |
| MD2 | job off as a 32-bit multiply, card-queue job (kind 1/2) only | b14cea7c5a8776042b5563d60558fafe | CAUGHT | 573 | P1=0/500 P2=328/500 P3=10/36 P4=123/500 P5=53/600 P6=59/100 | P2 1 |

The control M00 has the same md5 as the real library (d719148f...) and passes 2236/2236, so a copy built in a new directory is byte-identical and the driver does not fail by itself. Every mutant .so md5 differs from d719148f.

Summary: 27 mutants (M01-M21, MA, MB, MC, MD, MD1, MD2): **25 CAUGHT, 2 NOT CAUGHT (M06, M08)**.

## NOT CAUGHT (0 failing units in all of P1-P6, P3 included)

**M06** (`pend_take` after `sc_remove` instead of before, in `vq_issue`). Judged equivalent under the contract (single caller thread, an ops table whose callbacks do not call back into the planner):
- `pend_take` makes no ops call. The only call between the two orders is `sc_remove`, whose wrapper (`w_remove`) logs `pR` and updates the driver's model, nothing the planner can see. So the sequence of ops calls, the job fields and the stats at every hook are the same in both orders; the Z (stats) lines are logged and checked against the pend/arr/rel/wb model after every hook and every pool_ensure group.
- `pend_take` could differ only by spinning, i.e. if the pending entry had state 0 at that point. `vq_issue` returns false before reaching it when the key is in RAM and its pending entry has state 0, so at the call the entry is absent or finished.
- Probe (`M06p`: the mutant plus one line appended to a file at the call, tag by state): P1, P2, P4, P5, P6 (2200 units, `probe-M06p.events.txt`): `absent` 15794, `present_done` 64044, `present_state0` 0 (in 120 and 261 distinct child processes). P3 seeds 0-11 only (`probe3-M06p.events.txt`): absent 8311, present_done 25621, present_state0 0. So the reordered path runs about 114,000 times in the probe runs and the state-0 case never occurs, as argued. (P3 seeds 12-35 were not probed.) The probe run itself passed except one unit, see Caveats.

**M08** (a replaced pending entry is freed at once instead of going to the graveyard). A real behaviour change, not equivalent: the queue's worker may still store 1 into the freed entry's state, and the crate's unit test `replaced_pending_entry_is_kept_until_its_state_reads_one` catches it (`planner-mut.out`). The driver did not see it; the reasons, the first observed, the other two reasoned:
- Never reached: probe `M08p` (mutant plus a file append when the dropped entry's state is not 1): 0 events in P1, P2, P4, P5, P6 (2200 units) and 0 in P3 seeds 0-11. The path needs `insert_pend(vk, ...)` to find an existing pending entry with state 0 under the victim key `vk`.
- Reading the driver (`pool_ensure`, `drv_pend_take`, `w_enqueue`): a pending entry for a key exists only while the host cache holds the key (pending slots are in `prot`, so not evicted), and every path that moves the key into a card pool takes the entry first (`pend_take` in `vq_issue` kind 2, or `drv_pend_take` in `pool_ensure`) or `vq_issue` refuses while its state is 0. So a driver-made call sequence does not put a pool key and an unfinished pending entry together; the unit test builds that state by hand with its fake ops. I did not prove this unreachable in general, only that this driver's generators did not produce it in 2212 units.
- Even if reached, the effect is a write into freed memory (the queue's store into a freed 16-byte `Pending`): the driver runs the Rust library without ASan/Valgrind (stable Rust only on the box), so it would see it only if the allocator happened to crash. Not run.

## CAUGHT only weakly

- **M04** (write-back threshold 9/10 instead of 8/10): 7 units, P2 4 (seeds 78, 234, 478, 483) and P4 3 (361, 451, 228); none in P1, P3, P5, P6. First difference in all 7: the reference issues the job (`pE ...`) where the mutant first calls `pW` (host worth check), i.e. the pending count was in [0.8, 0.9) of `cap_slots`.
- **M21** (`t_plan` also counts the mismatch path): 4 units, all P4 (70, 206, 404, 417), all with the first mismatch at hook 0. The driver logs only whether `t_plan == 0` (`tp0=`), never its value, so the mutant shows only when no planning happened before the mismatch.

## The four new mutants

| | change | caught | first failure |
|---|---|---|---|
| (a) MA | `vq_issue` calls `sc_try_acquire(pool, cand_nu, key, ...)` | yes, 670 units (P2 358, P3 18, P4 143, P5 92, P6 59) | P2 1: ref `pT 3 31841 1 ...`, Rust `pT 3 1 31841 ...`; Rust child then stops: `VIOLATION job 1: kind 1/2 off does not match the key of the last try_acquire` |
| (b) MB | `sp_pend_take` counts `pf_used` when there is no entry (the real code does not count; a mutant that "forgets" to count when absent is the real code) | yes, 915 units (P1 262, P2 216, P3 36, P4 290, P5 111) | P1 2: `VIOLATION stats pf_used = 461, model 454 (after 94 hooks)` |
| (c) MC | card-pool `prot` list drops its last element | yes, 603 units (P2 353, P3 18, P4 140, P5 92); none in P1/P6 | P2 1: ref `pT ... 2:77cd02506570dc83`, Rust `pT ... 1:a9246e52e0eab1ab` (protected set size 2 vs 1) |
| (d) MD / MD1 / MD2 | `off = e.wrapping_mul(expert_raw as i32) as i64` | yes: MD 1868, MD1 (read-ahead job only) 1780, MD2 (card-queue job only) 573 units | P1 1 (MD, MD1): ref `pE J1 k0 ... off9966845952`, Rust: `VIOLATION job 1: kind 0 off is not e * expert_raw`; P2 1 (MD2): same check for kind 1/2 |

## Caveats

- One driver flake seen: in the probe run `probe-M06p`, P4 seed 228 failed because the REFERENCE child stopped on its own model check (`VIOLATION stats vq_waited = 67, model 68 (after 46 hooks)`, `ref=exit3 rust=exit1`), a timing race in the driver's async-wait model is the likely cause (a helper thread stores 3-5 ms after the planner is called; not diagnosed, the unit was not re-run). It is the only unit with a non-0/1 reference status among 28 full runs plus 4 probe runs, and none of the 2236-unit runs of a not-caught mutant or of the control contains one. It did not affect any verdict above (M06 and M08 passed all units in their full runs); it is a hint that a single failing unit with a reference-side VIOLATION is not by itself evidence of a planner difference. All 7 + 4 units of M04 and M21 were checked: each has a real first difference (see above), none is of this kind.
- P6 verdict: a unit fails if any of its four children exits non-zero unless all four ended in the slot cache's "every slot is in use" fatal. P6 seeds 3 and 41 are such all-four-fatal units for the real crate; in M05 and M15 the only P6 failures (2 each) name a `ref1` child (exit1, and sig6 once in M05): they are those two units, where the mutated Rust children did not end the same way as the reference children (M11, M18, MA, MD1, MD2 show the same ref1 entries next to many Rust-side failures). P6 is invariant-only (no byte comparison), so P6 counts are weaker evidence than P1-P5; no mutant was caught only by P6.
- M08p / M06p probes ran P1,P2,P4,P5,P6 in full and P3 seeds 0-11 only (36 real-trace units would need up to tens of GB of raw logs on a disk with 52 GB free).
- `.diff` files here carry a one-line `# name: description` header before the unified diff.

## Files

`mk_mutants.py` (builds M00-MD2), `mk_probes.py` (M06p, M08p), `run_all.sh`, `run_probes.sh`, `run_probes_p3.sh`, `cleanup_logs.py`, `analyze.py`; `<name>.diff` per mutant (and `M06p.diff`, `M08p.diff`); `logs/`: `analysis.txt` (per mutant counts, first failure, first-difference lines), `table.md`, `progress.txt`, `build.out`, `md5-before.txt`, `md5-after.txt`, `summaries.txt` (every run's `summary.txt`), `failing-M04-M21.txt`, probe event counts (`probe*-M0*p.events.txt`), `probe_progress.txt`.

## Protected files (md5 before and after, `logs/md5-before.txt`, `logs/md5-after.txt`, `diff` empty)

libplanner.so d719148fa7f6299409041b0e1f2f3f07, libplanner_ref.so f9b823f277b7233ca32e37cded8ffdd2, src/lib.rs 4e68bc3510719fdddd44af22ec01cc4e, src/planner.rs 35231db30d37639de3a17a138f4c140a, include/planner.h 3660d416307eeffaa36d83a5f74890ce, verify/pl_driver.cpp 2eb49f58fba59957837119b58e6efe06; also unchanged: src/abi.rs 27ffa88b, src/tests.rs 1bee0878, src/dense.rs ee564730, verify/pl_driver 25aed02b, verify/run.sh 7ad097c4. No file of the planner tree outside `verify/work`, `verify/out`, `verify/mutants` is newer than the "before" file; nothing newer in `~/rustmig/{copyq,slotcache,routetable}`.

## First differences (verbatim from `logs/analysis.txt`)

See `logs/analysis.txt`.
