# planner C++ reference (ref/)

Reference implementation of `../include/planner.h`: the engine's planner functions, extracted unchanged from `src/code/gpu/backend-gpu.cu` and `decode.cuh`, run inside a thin harness that exposes the header's C ABI. It is the original's text behind a harness, with the adaptations listed below. It is not an independently verified implementation.

Files: `extract.py` (Windows, python), `planner_extracted.inc` (generated, do not edit), `planner_ref.cpp` (harness + wrappers), `build.sh` (box), `smoke_ref.c` (hand-derived scenarios, fake ops), `controls.sh` (mutants and sanitizer run, box).

## Extracted verbatim (bytes, CRLF kept; sources read-only)

`backend-gpu.cu` sha256 `c87d4a14ab38e87e59bbb547d6c7fb916affb6301c35a44f449897ce48802169`, `decode.cuh` sha256 `36ad36f19e8d86dc330838d85b9f73ffd0b01ff06ee246f66e15b00ea21f48d0` (same values printed at the start and at the end of this work).

| block | file lines | bytes | content | sha256 |
|---|---|---|---|---|
| D1 | decode.cuh 492-496 | 471 | `Pending`, `g_pend`, `g_pend_keys`, `g_wb_skipped`, `g_t_pwait`, `g_pf_used`, `g_pf_waited` | aad35a1d...41e9 |
| D2 | decode.cuh 506-517 | 532 | `pend_take` | 3361460d...00e3 |
| D3 | decode.cuh 521-534 | 614 | `Arrival`, `g_arr`, `g_t_vwait`, `g_vq_waited`, `arr_wait` | 5eaf7ca0...34ae |
| B1 | backend-gpu.cu 58 | 63 | `g_rc_pos`, `g_rc_planned`, `g_pend_cap` | 8048f2bd...07f2 |
| B2 | backend-gpu.cu 99 | 58 | `g_lookahead`, `g_rc_ok` | d908d0ac...f086 |
| B3 | backend-gpu.cu 102 | 58 | `g_pf_planned`, `g_pf_skipped` | 6c7e1632...6e70 |
| B4 | backend-gpu.cu 107-112 | 405 | `Rel`, `PfJob` | 38c88d68...1c21 |
| B5 | backend-gpu.cu 267-288 | 1294 | `rc_plan` | a7fb3e0b...2d08 |
| B6 | backend-gpu.cu 293-370 | 4195 | `g_rel`, `g_wb_keys`, `g_vq_cap`, `g_vq_from`, `g_t_plan`, `g_vq_ram/drive/wb/nowb`, `vq_sweep`, `vq_issue`, `rc_plan_vq` (with the comments between) | 1cec3cae...b837 |
| B7 | backend-gpu.cu 389-399 | 614 | `route_hook` | 5c4cad12...1465 |

`planner_extracted.inc` = D1 D2 D3 B1 B2 B3 B4 B5 B6 B7, nothing added: 8304 bytes, 151 lines, all CRLF, sha256 `d1aa0c84a584b2d5dcbc05a11a5be5505f3362149e2a0c95f9d47b2e787e423e` (md5 `0c0881bbddaa5aca0423429e18bad731`, identical on Windows and on the box). Independent check: a separate python slice of both sources by those line ranges equals the file byte for byte (same sha256). Not taken: `force_routes`, `rc_load`, `rc_nu`, `pool_ensure`, `pf_enqueue`, `print_summary`, `configure`, `SlotCacheSel`, CUDA code.

Markers: start lines found by `startswith` (or exact text for `static void vq_sweep()`), each required to match exactly one line; end = first following line that is exactly `}` (`};` for `PfJob`); B6 must contain `vq_sweep`, `vq_issue`, `rc_plan_vq` in order and exactly three top-level `}` lines. Exit 1, nothing written (a previous `--out` is deleted first), if a marker matches 0 or >1 lines, an end marker is missing, a block is implausibly long, braces (counted per character; comments and strings not parsed) do not balance or close early, expected text is missing, or blocks overlap.

Negative controls (Windows, 2026-10-22): temp copy of `backend-gpu.cu` with `static void vq_sweep()` renamed `vq_sweepX` (the replace was confirmed to change the text) -> `FAIL: expected exactly one line matching 'static void vq_sweep()', found 0`, **exit 1**, no output file. `--src C:\nonexistent` -> `FAIL: source file not found: C:\nonexistent\backend-gpu.cu`, **exit 1**, no output file. (A first attempt at the first control used a relative path and the copy had no `backend-gpu.cu`; it exited 1 for "file not found", which proves nothing about the rename, so it was redone with absolute paths.)

## Adaptations (each also marked `ADAPTED (n)` in planner_ref.cpp)

1. `ekey(layer, e)` = `layer * g_experts_run + e`, `g_experts_run` = `cfg.experts` (original: compile-time `EXPERTS`, decode.cuh:201).
2. `EXPERT_RAW` is `#define EXPERT_RAW g_expert_raw` (a `long` set from `cfg.expert_raw`); the original is an enum constant (moe.cuh:5). The text uses it only in `(off_t)e * EXPERT_RAW`.
3. `g_vq`, `g_prefill` (decode.cuh:211, :500) are plain globals set from `cfg.vq`, `cfg.prefill`.
4. `CallSet` (same text as backend-gpu.cu:56) and `g_rc`; `g_rc` is filled in `sp_init` from `rt_calls/rt_layer/rt_experts` as `rc_load_sel` does in rust mode (counts are not copied: the planner does not read them).
5. Minimal `Handle {int L, card, fd}`, `CardPool {int dev; bool ready; SlotCacheSel c;}`, `HostCache {SlotCacheSel c; size_t cap_slots;}`, `g_pool[8]`, `g_host`, `g_handles[93]`. `sp_layer_set*` store `fd` with a release store; the extracted text reads `hd->fd` with a plain load, as the original does (so the reference does NOT give the "acquire ordering" the header promises; see contract notes).
6. `SlotCacheSel` holds an `sc_cache *` and forwards `contains, worth, acquire, try_acquire, remove, release` to the ops table (prot sets become an int array, as decode.cuh does). Null-handle rules kept for the methods the original guards (`contains` false, `worth` true, `remove` NULL). `find`, `touch`, `add_spare` do not exist (not in the ops table, not called by the planner text).
7. `rc_nu_sel` calls `ops.rt_next_use`, `rc_set_cur` calls `ops.rt_set_current`, `slotcache_set_next_use` calls `ops.sc_set_next_use`; the cpp-mode `g_rc_cur`/`g_nu_fn` state of the originals is dropped. `wall_now` is the original's text.
8. `pf_submit(const PfJob&)` converts a PfJob to a `cq_job` with the text of `pf_submit_rust` (state pointers are the addresses of the atomic ints; kind 0 jobs carry dev 0 / layer 0 from the struct defaults), keeps the `shared_ptr`s of `arr`/`rel` alive in a `CqHold` list (the engine's `g_cq_hold`, with its sweep every 128 pushes) and calls `ops.enqueue`. The engine's `g_cq_jobs` counter is not kept (contract: accounting is the caller's). It is declared before and defined after the include because `PfJob` is defined inside it.
9. Harness-only: a second `sp_init` resets all planner state (the contract allows one planner per process; the smoke test runs several scenarios in one process). `sp_layer_set_fd` on a layer with no handle is ignored; the contract does not say what it should do.

10. (Added after the independent review, 2026-10-23.) The original's `R()` creates a slot cache when the handle is NULL; the reference passes NULL through. `sp_init` copies the route table once and resets the wiring, so the driver wires pools and host after `sp_init`; wiring before init and a double init are not compared against the reference. The counters `g_sc_calls`, `g_cq_jobs`, `g_rt_nu_calls` are dropped.

Wrappers: `sp_init` copies ops/cfg, sets `g_lookahead`, `g_pend_cap`, `g_vq`, `g_vq_cap`, `g_vq_from`, `g_prefill`, `g_rc_ok` (= `cfg.rc_ok`, not recomputed) and builds `g_rc`; `sp_route_hook` builds a `std::vector<int>` and calls `route_hook`; `sp_pend_take` / `sp_arr_wait` call `pend_take` / `arr_wait`; `sp_host_acquire` = `g_host.c.acquire(key, g_pend_keys, heat, vkey, vheat)`; `sp_get_stats` reads the globals and the sizes of `g_pend`, `g_arr`, `g_rel`, `g_wb_keys`.

## Build (box `gpu`, `~/rustmig/planner/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

```
python extract.py --src C:\personal\oss\clover-framework\clover-kimi-k3\src\code\gpu --out planner_extracted.inc    # Windows
scp include/planner.h gpu:rustmig/planner/include/ ; scp ref/{planner_ref.cpp,planner_extracted.inc,smoke_ref.c,build.sh,controls.sh} gpu:rustmig/planner/ref/
ssh gpu 'cd ~/rustmig/planner/ref && sed -i "s/\r$//" build.sh smoke_ref.c controls.sh && ./build.sh && ./smoke_ref && ./controls.sh'
```
`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -I../include -I../../copyq/include -I../../slotcache/include -I../../routetable/include planner_ref.cpp -o libplanner_ref.so` (no `-march`, no fast-math), then the smoke binary with gcc `-O1 -std=gnu11 -Wall -Wextra -pthread`. The sibling include dirs are the existing `~/rustmig/{copyq,slotcache,routetable}/include` (read only; md5 of each equals the Windows file with CRs stripped). The build printed no warning. `nm -D` exports exactly the 10 `sp_*` functions of the header.

Hashes: `planner_ref.cpp` md5 `0271beee4d762c7c2cd92af7300ec0b2` (Windows = box); `libplanner_ref.so` md5 `f9b823f277b7233ca32e37cded8ffdd2` (the same after every rebuild); `smoke_ref.c` md5 `5832da612a4ed87b85ee42dd4e620e4f` on Windows, `8d5b44e10a32eb5f5bcdc777daa3243d` on the box (CRs stripped by `sed`).

## Smoke test (`smoke_ref.c`)

No real slot cache, route table or copy queue: hand-written fakes (array caches with scripted spares/victims/try_acquire outcomes/a not-worth key, a 3-6 call route table, a next-use answer table, an enqueue that records each `cq_job`). Expected values were derived by hand from the contract and the original text (derivations in the comments), before the first run. Scenarios: A read-ahead plan (job fields kind/fd/off/slot/p_state, pending creation, `skipped` counting, null handle and fd -1 skipped, a not-ready pool ignored, `sp_host_acquire`, `sp_pend_take`); B `pend_cap` early return, the quirk that a call left half planned is never replanned once `pos` has moved on, `sp_layer_set_fd`; C `worth` early return; D five mismatch cases (different experts, other layer, shorter list, reordered, plus `rc_ok` 0 at start): exact stderr text, `rc_ok` cleared, `set_next_use(NULL)` once, nothing planned afterwards; E card-queue plan from call 0 (kind 2 and kind 1 jobs with flags, write-down with victim_host and wb pending, `full[card]` after a failed try_acquire, read-ahead fall-through, sweep of the release list and of `wb_keys`, `vq_issue` returning false on a pending state 0, `pend_take`/`arr_wait` spinning until a test thread sets the state, timings > 40 ms); F `vq_from` switching from read-ahead to the card-queue plan, `vq_nowb` when the victim is not worth a slot, `arr_wait(vk)` spinning inside `vq_issue`, a wb pending replaced by a second one (`wb_keys` [203,203]), swap-with-last in the release list and in `wb_keys`, the end-of-table mismatch (pos >= calls); G (four variants) the write-down condition: cap 6 (pending 4 not < 4), prefill, capacity 0, and the successful write-down; calls with no handle / fd -1 skipped in the card-queue plan, a not-ready pool; H the in-flight cap (`out` counted from arrivals with state 0, `out < vq_cap`).

Result (box, run 4 times in a row): `PASS (537 checks, 0 failed)`, exit 0. Under ASan+UBSan (`controls.sh`, leak detection off because the original leaks a replaced wb Pending): `PASS (537 checks, 0 failed)`.

History of the test: the first run had 1 failure that was my own test's bug (it read a `Pending`'s state after `pend_take` had deleted it; removed). Later I found by mutation that the write-down threshold was not exercised (M3 survived with 369 checks) and added scenarios G and H.

Controls (`controls.sh`, temp copies of the extracted text in `/tmp/planner-ref-controls`, shipped files untouched; every mutant changes one line, confirmed by diff):

| mutant | change | outcome |
|---|---|---|
| M1 | `rc_plan`: `>= g_pend_cap` -> `>` | FAIL, 12 checks |
| M2 | `vq_sweep`: release when state `== 1` -> `== 0` | FAIL, 11 |
| M3 | write-down threshold `* 8 / 10` -> `* 9 / 10` | FAIL, 8 (survived before G) |
| M4 | `route_hook`: `>= g_vq_from` -> `>` | crash (exit 139) |
| M5 | `vq_issue`: kind 2 -> 1 | FAIL, 5 |
| M6 | cand `pos` -> `pos + 1` | FAIL, 12 |
| M7 | no `set_next_use(nullptr)` on mismatch | FAIL, 9 |
| M8 | `pend_take`: `g_pf_used += 2` | FAIL, 6 |
| M9 | no `g_pf_skipped++` | FAIL, 13 |
| M10 | `!g_prefill &&` removed | FAIL, 8 |
| M11 | `g_host.cap_slots &&` removed | **PASS (not caught): equivalent**, `pending < 0 * 8 / 10` is always false for a size_t, so the clause is redundant |
| M12 | `&& worth(vk)` removed | FAIL, 19 |
| M13 | `!hd \|\| hd->fd < 0` skip removed in `rc_plan_vq` | crash (139) |
| M14 | `pool.ready &&` removed in `rc_plan_vq` | FAIL, 24 |
| M15 | `pool.ready &&` removed in `rc_plan` | FAIL, 37 |
| M16 | `arr_wait` does not erase the entry | FAIL, 7 |
| M17 | no `g_wb_keys.push_back` | FAIL, 9 |
| M18 | no `g_arr[key] = j.arr` | FAIL, 33 |
| M19 | `out` counts state 1 instead of 0 | FAIL, 7 |
| M20 | `out < cap` -> `<=` | FAIL, 17 |
| M21 | lookahead `+ 1` removed (both plans) | crash (139) |
| M22 | `max(planned, pos)` -> `pos` | abort (fake cache "no scripted victim left", exit 2) |
| M23 | `g_vq &&` removed in `route_hook` | exit 2 after FAIL lines |
| M0 | no change, same path | PASS |

M4, M13, M21, M22, M23 end in a crash/abort or an early exit rather than a clean FAIL count; they are counted as caught because the test cannot reach PASS. Mutants of the harness (`planner_ref.cpp`) were not run.

## Not covered

One scripted world per scenario; no differential test against the engine build and no real slot cache / copy queue; `sp_host_acquire` with a real victim path, `sp_pool_set` after ready, threads other than the helper thread that sets states; `ekey` with `e >= experts` aliasing (as in the route table); out-of-contract calls (card or layer out of range, `uniq` NULL with n > 0, `sp_layer_set_fd` before `sp_layer_set`). Read from the text, not run: a negative `lookahead` (`pos + (size_t)lookahead + 1` wraps, so -1 plans nothing and -k gives a `last` below `pos`); the original's data race on `Handle::fd`.

## Contract notes for the owner (header not edited)

1. **fd ordering**: the header says the planner reads the fd with acquire ordering. The original (and so this reference) reads `hd->fd` with a plain load while a loader thread writes it; only the Rust side can honour the sentence.
2. **Enqueue order and per-hook sequence** match the header in every path I exercised. What the header leaves implicit and the text does: the card-queue plan never updates `planned_to` and never counts `skipped` (the header's "the same host read-ahead as in the read-ahead plan" could be read to include both); on `cap`/`worth` failure it goes to the next expert, not `return`.
3. **Half-planned calls**: when the read-ahead plan returns with `planned = i` and a later hook has `pos > i`, the loop starts at `max(planned, pos)`, so the remaining experts of call i are never read ahead (scenario B). Consistent with the header's wording, but easy to port as "resume at planned".
4. **`full[card]`** is set whenever `vq_issue` returns false, including the "in RAM, its read has not finished" path where `try_acquire` was never called (scenario E hook 1, F hook 1), so one pending expert blocks the card's later experts in that hook. The header says "else full[card] = 1" without that remark.
5. **`vq_nowb`** is counted only when the victim `vk >= 0` and the write-down condition fails; a victimless issue counts neither `vq_wb` nor `vq_nowb`. The condition `cap_slots > 0` is redundant (M11).
6. **Replaced entries** (`g_pend[vk] = j.wb`, `g_arr[key] = j.arr`): the replaced `Pending` is leaked, not deleted, and `vk` is appended to `wb_keys` again (the list can hold a key twice; the sweep removes the second occurrence because its pending entry is gone). Exercised only with a scripted, inconsistent fake (F); I could not construct the case with a consistent slot cache (a key evicted from a pool whose wb pending is still open cannot come back to the pool before `pend_take`).
7. **Victim of `host.acquire` is dropped silently** in `vq_issue` (`dk`, `dh` are never read); the header only says so for the read-ahead plan. (`sp_host_acquire` returns its victim to the caller.)
8. **Who installs the next-use function**: the header's `sp_init` does not call `sc_set_next_use`, yet `sp_route_hook` calls `set_next_use(NULL)` on mismatch. The engine installs `rc_nu_sel` (a planner-side function) in `configure`; through this ABI the caller must install its own function that calls `rt_next_use` (the planner's `rc_nu_sel` is not exported). Not stated in the header.
9. **Lifetime of arrival/release flags**: "the job holds it" is, in this reference, the harness (`CqHold`, as the engine's `pf_submit_rust`); a recording enqueue that does not keep them alive would see freed memory after `sp_arr_wait` (the original's `arr_wait` removes the entry before waiting).
10. **sp_stats** has no `g_wb_skipped` (written by `pool_ensure`, printed by the engine's summary); counters `g_cq_jobs` are the caller's, as the header says.
11. `sp_layer_set_fd` for a layer that has no handle: undefined in the header; ignored here.
