# slotcache (Rust) — notes

Implements `include/slotcache.h` (the header was edited by the integrator after the first build, to match the original more exactly; no signature changed; current md5 is in `verify/out/md5-before-bitmap.txt` on the box). Behaviour reference: `struct SlotCache`, `g_policy`, `g_nu_fn` in `src/code/gpu/decode.cuh` (read 2026-10-21, lines 196-286).

Layout: `src/lib.rs` (the whole implementation and the C ABI), `src/tests.rs` (unit tests, `#[cfg(test)]`), `smoke_rust.c` (C program against `libslotcache.so`).

## Design

- LRU list: intrusive doubly linked list over a slab (`Vec<Node>` with `prev`/`next` indices and a free list). `HashMap<i32, usize>` maps key to node. `find` unlinks and re-links at the front (the original's `splice`), O(1).
- `by_next`: `BTreeSet<(i32, i32)>`, same ordering as `std::set<pair<int,int>>`. `spare`: `Vec<*mut u8>`, `push` / `pop` (last in, first out).
- Policy and next-use function are process-wide atomics (`AtomicI32`, and the function address in an `AtomicUsize`, 0 = unset). `belady()` is re-read at each place the original evaluates it (branch choice in `acquire`, then again when storing the new entry's next use).
- `prot`: pointer + length. Up to 32 keys: linear scan of the slice. Above 32 keys, all in [0, 2^20): a reusable bitmap (128 KB per cache, allocated on first use, set at the start of the call and cleared at its end), because the engine's planners pass up to ~5,500 protected keys and the scan asks about 48 + |prot| keys per acquire (a reviewer measured 16.6 us per acquire at 5,491 keys against 1.8 us for the original's hash set). Any key outside the range falls back to the scan. Same answers either way (the original builds an `unordered_set`); duplicates allowed, order irrelevant. NULL with length 0 is an empty set.
- Slots are `*mut u8` that are stored and returned, never dereferenced.
- Fatal path: writes the exact message with `write_all` on `stderr()` (errors ignored, no `eprintln!`), then `std::process::exit(1)`. Release profile has `panic = "abort"`.
- Floats: `heat * 0.8f32 + w` in `f32`, two separate operations; Rust does not contract to FMA and nothing enables it.
- The exported functions that dereference pointers are `unsafe extern "C"` (same C ABI as a plain `extern "C"`); `sc_set_policy`, `sc_set_next_use`, `sc_new` are safe.

## Behaviour map (original -> Rust)

| Original | Rust |
|---|---|
| `find`: hit moves to front | `sc_cache::find` unlink + push_front |
| `touch`: belady: erase `{e.nu,key}`, `e.nu = g_nu_fn(key)`, insert; else `heat = heat*0.8f + w`; absent: nothing | `touch` |
| `worth`: true if not belady, or spare exists, or set empty; else `nu(key) < by_next.rbegin()->first` (function called only in the last case) | `worth` |
| `acquire` spare branch: `spare.back()` then pop; `*vkey = -1`, `*vheat` not written | `acquire` first branch, `vheat = None` |
| `acquire` belady branch: scan `by_next` from the largest pair downward, first key not in prot; `victim < 0` (sentinel -1) -> fatal; erase `{v.nu, victim}`; remove from list and map | second branch, `by_next.iter().rev()` |
| `acquire` heat/LRU branch: scan lru from the least recent end, prot keys skipped without counting, window 48 (policy != 0) or 1 (policy 0), strict `<` on heat so ties keep the earlier one in the scan; none found -> fatal; erase `{nu, victim}` from `by_next` | third branch |
| new entry: front of lru, nu = `g_nu_fn(key)` and inserted in `by_next` only if belady(), else nu = 0 and not inserted | end of `acquire` |
| `try_acquire`: if no spare, look at `by_next` (whatever `belady()` says) from the top for the first key not in prot, accept only if its next use `> cand_nu`, scanning stops at that first non-prot key; then call `acquire`, which decides again | `try_acquire` then `acquire` |
| `remove`: erase `{e.nu,key}`, unlink, erase from map, return slot, spare untouched | `remove` / `detach` |
| `release` / spare push | `sc_release` / `sc_add_spare`, same push |

## Differences / choices where the original is silent or outside the contract

- Acquiring a key that is already held (outside the contract): the old entry is dropped first so the list/map stay consistent. The original would leave a stray list node. Not tested.
- A negative key is never chosen as a belady victim: the original uses -1 as its "no victim" sentinel (`victim < 0`) and so does this code. Real keys are `layer * EXPERTS + e >= 0`.
- `by_next` containing a key that is not held cannot arise within the contract; if it did, the Rust `expect` would abort (the original would insert a default entry through `map[victim]`).
- If the next-use function unsets itself in the middle of one operation (between `belady()` and the call), `nu_call` returns 0 instead of calling a null pointer.
- `sc_new` allocates with `Box`; `sc_free(NULL)` is a no-op. NULL `vkey` / `vheat` pointers are skipped instead of written.
- `sc_snapshot_*` accept NULL output pointers when `cap` is 0 (used by the tests to size the buffers).

## Tests (`cargo test`, 20 tests, hand-derived; they hold a mutex because policy and function are process-wide)

- `large_prot_bitmap_equals_linear_scan`, `large_prot_bitmap_equals_linear_scan_belady`: 300 random rounds each; the bitmap path and the scan give the same victim, slot and heat bits, keys outside the bitmap range take the fallback, and the bitmap is all zero after every call.

- `policy0_evicts_least_recently_used`: spares last-in first-out, find moves to front, LRU victim, victim heat reported, size/contains.
- `policy0_ignores_heat_and_prot_does_not_count_toward_the_window_of_one`: hottest entry still goes if it is least recent; prot (with duplicates) skipped, next entry taken.
- `policy1_lowest_heat_ties_keep_earlier_in_scan_and_prot_is_skipped`: tie 11 vs 12 keeps 11; prot changes the victim.
- `policy1_touch_is_single_precision_heat_decay_and_changes_the_victim`: bits of `h*0.8f + w` in `f32`, touch of an absent key, touch does not reorder lru, touch flips which entry is the victim.
- `policy1_window_is_48_entries_not_in_prot`: 52 entries; the window is 48 from the tail, and with 3 prot keys it still holds 48 non-prot keys (a colder key at position 48 is reachable only then).
- `belady_scripted_victims_and_ordered_set`, `belady_equal_next_use_takes_the_larger_key`: scripted next-use function, ordered-set contents after each step, prot skipping, touch re-inserting, tie broken by the larger key.
- `worth_in_all_four_cases`: empty set, spare exists (function not called), `<` comparison incl. equality, not belady (policy 1; policy 2 with no function).
- `try_acquire_accepts_and_refuses`, `try_acquire_with_nothing_to_replace_refuses_and_precheck_ignores_belady`: refuse changes nothing and writes `*vkey = -1`; strict `>`; prot stops the scan at the first non-prot key; spare always accepted; no spare and empty set refuses; with policy 1 the pre-check passes on `by_next` and `acquire` then uses the heat rule (different victim than Belady would pick).
- `remove_returns_slot_without_making_it_spare`, `find_and_touch_of_absent_key_change_nothing`, `slab_reuse_after_many_evictions_keeps_the_list_consistent` (200 operations, checks list length equals map size).
- `stale_by_next_after_next_use_function_is_unset`, `entry_acquired_while_unset_is_invisible_to_belady_until_touched`: function unset mid-run, then touch / worth / acquire / remove; the function is never called while unset; erased pairs use the stored value; entries stored with nu 0 are not in the set until touched under belady.
- `struct_layout_matches_the_c_header`: compiles a C file with `cc` against `include/slotcache.h`, compares `sizeof(sc_entry)` and the offsets of key/slot/heat/nu with `offset_of!`. If `cc` is missing it prints `SKIPPED ...` (stdout and stderr; visible with `-- --nocapture`) and returns. It ran (not skipped) on the box.
- `fatal_path_exits_1_with_the_exact_message` (+ `fatal_child`): re-runs the test binary for five situations (policy 0, policy 1, belady with the only entry in prot, empty cache without spares in the heat branch and in the belady branch) and requires exit status 1 and stderr equal to the contract message, byte for byte.

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/slotcache`)

    cargo build --release                       # target/release/libslotcache.so
    cargo test --release                        # or cargo test
    gcc -O1 -I include -o smoke_rust smoke_rust.c -L target/release -lslotcache -Wl,-rpath,$PWD/target/release && ./smoke_rust

`smoke_rust` runs the scenario twice (3 slots, policy 0, acquire 1, 2, 3, find 1, acquire 4: victim key 2 with slot B). Spares are last-in first-out, so the two runs differ in the spare order (C,B,A and A,B,C); the victim is slot B in both.

## Not covered

- These tests are hand-derived. Equality with the C++ original is established by `verify/` (differential driver over `ref/`, scenarios S1-S11, mutants M1-M16 and B1-B3); see `verify/mutants/RESULTS.md` and the context file `docs/clover-kimi-k3-gpu-context.md`.
- Concurrency: the contract says single-threaded; only the globals are atomics, caches are not synchronised.
- A callback that re-enters the cache, a key acquired twice, releasing a slot twice, `NULL` cache pointers.
- NaN heat or w: the payload bits of a NaN result differ from the original (reviewer probe); the engine never produces NaN.
- Windows / 32-bit: layout checked only on x86_64 Linux (LP64).
