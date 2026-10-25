# planner (Rust) - notes

Implements `include/planner.h` (fixed contract, not edited; md5 3660d416). Behaviour reference (read 2026-10-22): `rc_plan` (backend-gpu.cu 267-288), `vq_sweep` (300-312), `vq_issue` (314-339), `rc_plan_vq` (341-370), `route_hook` (389-399), `pf_submit_rust` (202-225, how a PfJob becomes a cq_job), the globals/structs `Rel`, `PfJob`, `g_rel`, `g_wb_keys`, `g_vq_*`, `g_t_plan` (58-110, 293-298); decode.cuh `Pending`, `g_pend`, `g_pend_keys`, `Arrival`, `g_arr`, `pend_take`, `arr_wait`, `SlotCacheSel` (the methods the planner calls on the pool caches and the host cache).

Layout: `src/lib.rs` (C ABI, process-wide state), `src/planner.rs` (the planner), `src/abi.rs` (`#[repr(C)]` structs), `src/dense.rs` (map with a dense key vector), `src/tests.rs` (unit tests, a child module of `planner` so it sees private fields), `smoke_rust.c`. No dependencies; `panic = "abort"`; crate-type cdylib + rlib.

## Design

- `sp_ops`, `sp_config`, `sp_stats`, `cq_job` are `#[repr(C)]` in the header's field order; `off_t` is `i64` (LP64 Linux). The slot cache / route table are reached only through `sp_ops`; `rt_calls`, `rt_layer`, `rt_experts` are called at each use (the original reads `g_rc` at each use), `rt_next_use` exactly where the original calls `rc_nu_sel`.
- One `Planner` in a process-wide `UnsafeCell<Option<Planner>>`. The contract says one caller thread; an atomic `busy` flag turns a concurrent or re-entrant call (for example an ops callback calling back into the planner) into `exit(1)` with a message instead of a data race. `sp_layer_set` / `sp_layer_set_fd` do not take it.
- Layers: a static table of 93 slots, each `present`, `fd`, `card` as `AtomicI32`; `fd` is stored and read with Release / Acquire, so `sp_layer_set_fd` from a loader thread is safe. The planner reads the fd once per call of the plan and uses that value for the check and for every job of that call (the original reads `hd->fd` again when it builds each job; a second read could differ only if the loader stored between the two, and could only turn -1 into an fd).
- Pools (`sp_pool_set`) and the host cache (`sp_host_set`) live in `Wiring`. Calls made before the first `sp_init` are kept and used by it.
- `pend` (g_pend + g_pend_keys, which are always equal in membership in the original) is a `DenseMap<Box<Pending>>`: `HashMap<i32, usize>` index (cheap integer hasher), `Vec<i32>` keys, `Vec<Box<Pending>>`; removal is swap-remove. `pend.keys()` is passed as `prot` (pointer + length) to `sc_acquire` without building anything per call. `arr` (g_arr) is a `DenseMap<Box<Arrival>>`. `rel` (g_rel) and `wb_keys` are `Vec`s with swap-remove, like the original.
- Entries are `Box`es: the address of the `AtomicI32 state` inside (stable while the Box lives, even when the Vec moves it) is what goes into `p_state` / `wb_state` / `arr_state` / `rel_state`. An entry is freed only after its state was read as 1 with Acquire: `pend_take` and `arr_wait` spin first (`yield_now`), the sweep checks first. `Pending` and `Rel` are `repr(C)` / plain structs with the state first.
- Replaced entries (graveyard): `insert_pend` / `insert_arr` return the old value. If its state already reads 1 it is dropped at once; otherwise it goes to `dead_pend` / `dead_arr` (the original leaks the old Pending and keeps a shared_ptr in the job for the old Arrival). `vq_sweep` ends with `retain(state != 1)` on both lists, which frees an old entry when its state reads 1 during a later sweep. An entry whose state never becomes 1 stays for the life of the process (that is a leak, not a use after free). The sweep only runs in the card-queue plan.
- A second `sp_init` starts a new planning state (counters, pending, arrivals, releases, wb_keys, graveyards are not carried over); layers, pools and the host cache are kept. The old planner is `mem::forget`-ten, not freed (a worker may still write its states).
- Time: `std::time::Instant`. `t_plan` is the time of the planning branch of `sp_route_hook` (not of the mismatch branch, not of `rt_set_current`); `t_pwait`, `t_vwait` are the spin times.
- Messages: `write_all` on `stderr()`, errors ignored, no `eprintln!`. The mismatch text is `mismatch_message()`, tested byte for byte (also in a child process through the real stderr).
- Floats: `1.0f32` literal for the read-ahead heat and the `vq_issue` `try_acquire` call; the victim heat `vh` is passed on unchanged.
- Arithmetic as the C: `ekey = layer*experts + e` is wrapping i32; `off = (e as i64) * expert_raw` (wrapping); `last = min(calls, pos + (size_t)lookahead + 1)` with wrapping size_t arithmetic (a negative lookahead wraps like in C); `cap_slots * 8 / 10` in wrapping usize; `(long)pos >= vq_from`.

## Behaviour map (original -> Rust)

| Original | Rust (`planner.rs` unless noted) |
|---|---|
| `route_hook`: `rc_set_cur(pos)` first, then `if g_rc_ok` ... `g_rc_pos++` | `Planner::route_hook` |
| mismatch test `pos >= size || layer != L || experts != uniq`, message, `g_rc_ok=false`, `slotcache_set_next_use(nullptr)` | same condition (short-circuit order kept), `write_stderr(mismatch_message)`, `rc_ok = 0`, `ops.sc_set_next_use(None)` |
| `wall_now()` around the plan, `g_t_plan +=` | `Instant`, `st.t_plan` |
| `g_vq && (long)g_rc_pos >= g_vq_from` -> `rc_plan_vq` else `rc_plan` | `cfg.vq != 0 && pos as c_long >= cfg.vq_from` |
| `rc_plan`: loop `i = max(planned, pos) .. last`; no handle or fd < 0 -> `planned = i+1; continue` | `rc_plan` |
| key pending -> skip; `(pool.ready && pool.contains) \|\| host.contains` -> `skipped++` | same short-circuit order of `sc_contains` calls |
| `g_pend.size() >= g_pend_cap` -> `planned = i; return`; `!worth` -> `planned = i; return` | same, in this order |
| host acquire (prot = g_pend_keys, heat 1.0f), `new Pending`, `g_pend[key]`, `g_pend_keys.insert`, `g_pf_planned++`, job kind 0 `{fd, off = e*EXPERT_RAW, slot, p}` | `host_read_ahead` (acquire, insert, `planned += 1`, enqueue; dev 0, layer 0, `p_state` = the entry's state) |
| `vq_sweep` release list, swap-with-last, re-look at i | `vq_sweep` first loop (`Vec::swap_remove(i)` is exactly `v[i]=v.back(); pop_back()`) |
| `vq_sweep` wb_keys: no pending -> drop key; state 1 -> erase pending key + delete + drop key; else i++ | second loop |
| `rc_plan_vq`: sweep, prot = ekey(layer of call pos, e), `out` = arrivals with state 0, `full[]`, loop `pos .. last` | `rc_plan_vq` |
| `pool.ready && contains` -> next; `pool.ready && !full && out < vq_cap` -> `vq_issue(..., i == pos ? (int)pos : rc_nu_sel(key), prot)` (the argument is evaluated before the call) | same; `rt_next_use` is called before `vq_issue` starts |
| `vq_issue`: `in_ram = contains`; in RAM and pending state 0 -> false; `try_acquire`; NULL -> false | `vq_issue` |
| `in_ram`: `pend_take(key)`, `host.remove(key)`, `kind = 2`, new Rel pushed, `vq_ram++`; else `kind = 1`, `vq_drive++` | same order |
| `vk >= 0`: `arr_wait(vk)`; `!prefill && cap_slots && g_pend.size() < cap_slots*8/10 && worth(vk)` -> acquire(vk, prot = pending keys, heat vh), new Pending under vk (replacing), key into `g_pend_keys` and `g_wb_keys`, `vq_wb++`; else `vq_nowb++` | same; the replaced entry goes to the graveyard unless its state is 1 |
| `j.arr = make_shared<Arrival>(); g_arr[key] = j.arr; pf_submit(j)` | `insert_arr`, then `ops.enqueue` (one call per job) |
| `pf_submit_rust`: `p_state = j.p ? &state : null`, same for wb, arr, rel | each `*_state` is the entry's state address or NULL; kind 0 jobs only `p_state` |
| `pend_take`: no-op if absent; wait with timing, `g_pf_waited++`; `g_pf_used++`; erase, `delete` | `pend_take` (`pf_used` counted also when there was no wait) |
| `arr_wait`: no-op if absent; take out of the map first; spin if state 0, `g_vq_waited++` | `arr_wait` |
| `sp_host_acquire` | `host_acquire` (`sc_acquire` with prot = pending keys) |

## Choices where the header or the original is silent

- `sp_layer_set` with a layer outside 0..92 or a card outside 0..7, and `sp_pool_set` with a card outside 0..7, are ignored (the original indexes arrays out of bounds). A call whose layer is outside 0..92 counts as "no handle". (Until 2026-10-23 a card >= 8 was stored and aborted at the pool index; changed after the independent review, test `exported_functions_end_to_end`.)
- `sp_get_stats` does not abort when another planner call is in flight: it reads the counters without taking the busy flag (they may be mid-update), so the engine's atexit summary still works after a worker thread called `exit()` while the main thread spins in `sp_arr_wait`. Before this the process exited 1 from inside the atexit handler (review finding 1; same test).
- A second `sp_init` leaks the old pending entries and the in-flight host slots are no longer protected: outside the contract, argued not run (review finding 6); the engine calls it once.
- `uniq == NULL` with `n > 0` is treated as an empty list (the original segfaults).
- A NULL cache pointer for the host or a ready pool is passed through to `sp_ops`; the planner does not emulate `SlotCacheSel`'s `rs && ...` guards.
- `sp_*` functions other than `sp_layer_set*`, `sp_pool_set`, `sp_host_set`, `sp_get_stats`, `sp_init` before `sp_init`: message `planner: <name> before sp_init`, `exit(1)`. `sp_get_stats` before `sp_init` writes zeros.
- `job.layer` for a kind 1/2 job is the layer number of the call (the original's `hd->L`); the header says "the handle's layer", and `sp_layer_set(layer, ...)` makes the two the same.
- `prot` for the card pool is the experts of the current call as keys, built once per hook as a `Vec` (duplicates in `uniq` stay duplicates; membership only).
- The `out` count walks all arrivals at each card-queue hook (as the original); there are arrivals with state 1 that stay until `arr_wait` or a replacement.
- The header says `sp_pend_take` "adds the wait time to t_pwait and counts pf_waited" only when it waited, and counts `pf_used` always: implemented so.

## Tests (`cargo test`, 31 tests, expectations derived by hand from the original; fake ops table in `tests.rs` records every ops call in one ordered log)

read-ahead plan job list and `skipped` (`read_ahead_plan_job_list_and_skipped`); `pend_cap` early return and `planned` resumption; `worth()` false early return; calls without handle / fd < 0 and the loader setting the fd later; lookahead window and clamping; key / offset with 896 / 17547264 (64-bit offset); rc_ok = 0 from the start; vq_from selecting the plan; mismatch effects for five causes + exact message + message on the real stderr (child process); `t_plan`; card-queue plan with in_ram false and true (call order including `next_use`, job fields, release entry); issue refused while the RAM read is in flight, then accepted; the 8/10 write-back rule (7/8 pending of 10 slots, worth false, prefill, no slots, cap 13); victim arrival wait (done and spinning with a helper thread); `full[card]` per card; in-flight limit (`out >= vq_cap`, state-1 arrivals not counted); pool-held / RAM-held keys; host side of the vq plan; sweep swap-with-last order (release list and wb_keys); `pend_take` and `arr_wait` spinning with a helper thread, absent keys, finished entries; replaced pending entry and replaced arrival (graveyard until the state reads 1, freed at once when already 1); job state pointers equal the entries and the sweep releases the RAM slot; `DenseMap`; struct size / offsets of `cq_job`, `sp_ops`, `sp_config`, `sp_stats` against a real `cc` compile of the header (finds the sibling headers in `../*/include` or `*-include`); the exported C functions end to end (init, pre-init wiring, layers, hook, host_acquire, pend_take, re-init, mismatch).

Negative controls (scratch copy on the box, one mutation each, 21 mutations, `cargo test` after each): 20 caught (each by at least one named test; for example the sweep re-look, `planned = i + 1` on the cap, `skipped` not counted, 9/10, `full[]` never set, raw expert numbers in `prot`, no graveyard, `out` counting all arrivals, `cand` always from `rt_next_use`, pos incremented early, heat 2.0, no `+1` in `last`, `pf_used` only when waited, swapped config fields, wb entry not erased, worth before cap, arr_wait leaving the entry, next-use not unset, wb key pushed early, `t_plan` counting a mismatch). One NOT caught: M06 (`pend_take` after `sc_remove` instead of before in `vq_issue`): the two orders cannot be told apart through the ops log, and `pend_take` cannot spin there because an in-RAM key with a pending state 0 returned false earlier; treated as an equivalent mutation (reasoned, not proven by test).

## Smoke (`smoke_rust.c`, C program against `libplanner.so`)

Fake caches, a fake route table and a recording queue written in C: layer without a file, read-ahead jobs, `pend_take` spinning on a pthread, card-queue job, `arr_wait` spinning on a pthread, mismatch. Output: 10 PASS lines and `PASS`.

## Build and run (box `gpu`, `export PATH=$HOME/.cargo/bin:$PATH`, in `~/rustmig/planner`)

    cargo build --release                       # target/release/libplanner.so
    cargo test                                  # or --release
    gcc -O1 -Wall -Wextra -I include -I copyq-include -I slotcache-include -I routetable-include -o smoke_rust smoke_rust.c -L target/release -lplanner -lpthread -Wl,-rpath,$PWD/target/release && ./smoke_rust

(in the repository the three sibling headers are in `../copyq/include`, `../slotcache/include`, `../routetable/include`.)

## Not covered

- Equality with the C++ original (not claimed; another agent compares it with a differential driver against `ref/`).
- Concurrency: no test with a real copy queue and its worker threads, no Miri, ThreadSanitizer or loom run; the spin tests use one helper thread storing 1 with Release. The sequence "worker stores 1 -> planner reads 1 -> planner frees" is argued from the orderings, not checked by a tool.
- `sp_layer_set_fd` racing with a plan: single-threaded tests only (the atomics are the argument).
- A real route cache, the engine's `PfJob` objects, real slot caches (`libslotcache.so`) or the real queue: only the fakes. No performance measurement; the cost of `out` (walk of all arrivals per hook) and of hashing is not measured.
- Behaviour outside the contract (layers / cards out of range, NULL caches, calls from several threads, callbacks that re-enter the planner) is only handled as listed above.
- 32-bit or non-Linux targets (`off_t`, `unsigned long` sizes reasoned for LP64 Linux only).
