# supply C++ reference (ref/)

Reference implementation of `../include/supply.h`: the engine's `pool_ensure` (with `run_parallel` and the little it needs), extracted unchanged from `src/code/gpu/decode.cuh` **as committed at 4d8d891**, run inside a thin harness that exposes the header's C ABI. It is the original's text behind a harness, with the adaptations listed below. It is not an independently verified implementation.

Files: `extract.py` (Windows, python, reads git), `supply_types.inc` and `supply_extracted.inc` (generated, do not edit), `supply_ref.cpp` (harness + ABI wrappers), `build.sh` (box), `smoke_ref.c` (hand-derived scenarios, fake ops; also run against the Rust library as `../smoke_rust.c`).

## Extracted verbatim (bytes, sources read-only)

Source: `git show 4d8d891:clover-kimi-k3/src/code/gpu/decode.cuh` (the working tree is being edited by someone else and is never read). **The committed blobs have LF line endings (0 CRLF)**; a Windows checkout with autocrlf shows CRLF, so a hash of the working-tree file differs from the blob hash by design. The task text said the engine files are CRLF: that is the checkout, not the committed text. Blob sizes, line counts and sha256 are hard-coded in `extract.py` and checked before anything is cut (a different blob exits 1):

| blob | bytes | lines | sha256 |
|---|---|---|---|
| decode.cuh | 45669 | 779 | `fc7682e9bc6a043af4ebe59b1a7cb27f750ff7e6cde4dcb8beeaed5294955dee` |
| kernels.cuh (checked, not taken) | 29623 | 533 | `0ce0a33edd8ffde2a0f2fc3dbff8b2028cae07ba210874660a123271dde6277a` |
| moe.cuh (checked, not taken) | 24905 | 444 | `4af5c825ebadaf33c15f8d13be7f1a45f4e800b92e476f5de1cd2bfdb271d28f` |

| block | decode.cuh lines | bytes | content | sha256 |
|---|---|---|---|---|
| T1 | 437 | 157 | `struct ExpertStats` | `75cef883...7299` |
| T2 | 439 | 27 | `enum { MAX_READERS = 32 };` | `d2af6212...1fca` |
| X1 | 473-484 | 572 | comment + `run_parallel` | `b8059949...404c` |
| X2 | 487 | 120 | `wall_now` | `b98b307e...f0f9` |
| X3 | 495 | 150 | `g_wb_skipped` (with its comment) | `e2960b35...d2cf` |
| X4 | 536-620 | 5748 | the comment block and `pool_ensure` | `6f77169b...467e` |

`supply_types.inc` = T1 T2 (184 bytes, sha256 `829d1eba9b1ed448a90d8df4608e0164c2e7b126ccb10b75e58ff56fa57eaf6a`); `supply_extracted.inc` = X1 X2 X3 X4 (6590 bytes, 99 lines, sha256 `727c71b394e538adbd82656e1822c258ebcf30822d371b18943873978c1df934`); nothing added, file order kept. Why two files: the harness's `CardPool` needs `ExpertStats` and `MAX_READERS`, and `pool_ensure` needs `CardPool`; in the engine they are interleaved with other code. Independent check: `extract.py` slices the same blob a second time by the line ranges above (`EXPECTED_RANGES`, found when it was written) and compares byte for byte. The markers are found with `startswith` / exact text, each must match exactly one line; ends are the first following line that is exactly `}`; braces are counted per character (comments and strings are not parsed); expected text fragments (13 for `pool_ensure`) must be present. Exit 1, nothing written, if any check fails (a previous output is deleted first).

Controls of the extractor: `controls.py` (Windows, 2026-10-25) gives it a source that differs from the committed text in one way and expects exit 1 and no output file (a stale output file is planted first and must be removed): one comment byte changed (blob hash), `pool_ensure` renamed (found 0), `pool_ensure` defined twice (found 2), the write-back threshold line changed and the chunk arithmetic changed (expected text missing), `run_parallel` without its closing brace and `pool_ensure` without its closing brace (block too long), the `CHECK` macro of kernels.cuh changed, `EXPERTS` of moe.cuh changed, a nonexistent `--src` directory, a nonexistent `--repo`, another commit (7cbb532). Positive control: an unchanged copy read with `--src` exits 0 and writes files byte-identical to the shipped ones. Result: all 15 as expected (the first run of the script had two controls that failed for another reason than the one intended, a duplicate definition that sat next to `run_parallel` and a replace that hit another macro; both were fixed and re-run). The brace-balance check itself was not the one that fired in the two brace controls (the span limit fired first).

`extract.py` also checks, without taking them, the text of the two adaptation sources: the `CHECK` macro (`kernels.cuh:17`) and the enum with `EXPERT_RAW = 17547264` and `EXPERTS = 896` (`moe.cuh:5`).

## Adaptations (each also marked `ADAPTED (n)` in supply_ref.cpp)

1. The CUDA names of the text (`cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaSetDevice`, `cudaMemcpyAsync` with `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaStream_t`, `cudaEvent_t`) are small functions / typedefs that call `sy_ops`. The calls in the text are unchanged.
2. `CHECK(x)` (kernels.cuh:17 prints `file:line: error text` and `exit(1)` when the status is not `cudaSuccess`): here `int e_ = (x); g_ops.check(e_, #x); if (e_ != 0) exit(1);`. The text handed to `ops.check` is the preprocessor's stringization of the argument, which is what `supply.h` lists as the 11 sites; `smoke_ref.c` and the driver check each text against that list.
3. `ekey` uses `cfg.experts` instead of the compile-time `EXPERTS` (decode.cuh:201); `EXPERT_RAW` is `#define EXPERT_RAW g_expert_raw`, a `long` set from `cfg.expert_raw` (an enum constant in the original: the text uses it in `size_t` and `off_t` expressions, identical for values below 2^31). Built with `-fwrapv` so the int overflow of `ekey` wraps like the Rust version.
4. `g_vq` (decode.cuh:211), `g_chunks` (:207), `g_prefill` (:500) are plain globals set from `cfg`; `g_slab` is `#define g_slab (const_cast<unsigned char *const *>(g_cfg.slab))`, the caller's table (the original array `g_slab[93]` of :504), read at each use.
5. `SlotCacheSel` (decode.cuh:326-401) becomes two classes. `PoolSel` forwards `find`, `touch`, `acquire` to `ops.sc_*` (a `std::unordered_set` becomes an int array as in decode.cuh:332); `HostSel` forwards `find`, `worth`, `remove`, `release` to `ops.sc_*` and its `acquire` to `ops.host_acquire`, because the original passes `g_pend_keys` (the planner's set of pending keys) as `prot` and that set lives in the planner. The null-handle rules of the original are kept for the methods it guards (`find` NULL, `touch` nothing, `worth` true, `remove` NULL); `acquire` and `release` pass a NULL cache through (the original creates a cache there, which this ABI cannot do).
6. Minimal `CardPool` and `HostCache` with only the members `pool_ensure` touches; `sy_ensure` rebuilds them from the caller's `sy_pool` / `sy_host` for the duration of the call and copies `st` back at the end. `h_eptr` is a `const unsigned char **` instead of the engine's `std::vector`.
7. `g_pend_keys` is an empty `std::unordered_set<int>` that only gives the text something to pass to `HostSel::acquire`, which ignores it.
8. `pend_take` / `arr_wait` (decode.cuh:506-517, 521-534) belong to the planner; here they call `ops.pend_take` / `ops.arr_wait`.
9. Harness only: `sy_init` copies ops and cfg, resets `g_wb_skipped` (contract), may be called again.
10. Harness only: the ABI entry `sy_ensure` (message `supply: sy_ensure before sy_init` and `exit(1)` before the first `sy_init`; the original has no such state), the copy of `std::vector<int>` from the caller's array, `sy_wb_skipped`.

Counters and objects the original has and the harness drops: `g_sc_calls`, the `Timed` struct is kept (it is in the text), `act_gate` / `g_act_hold` are not used by `pool_ensure` (they are in the copy queue).

## Build (box `gpu`, `~/rustmig/supply/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

```
python extract.py                              # Windows, in ref/: reads git (--repo, --commit 4d8d891), writes the two .inc files next to it
powershell verify/sync.ps1 -Cmd 'ref/build.sh && ref/smoke_ref'   # copy the crate to the box and build
```
`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -fwrapv -pthread -I../include -I../../slotcache/include supply_ref.cpp -o libsupply_ref.so` (no `-march`, no fast-math), then `smoke_ref` with gcc `-O1 -std=gnu11 -Wall -Wextra -pthread`. The slot cache header is the existing read-only `~/rustmig/slotcache/include`. `nm -D` exports exactly `sy_init`, `sy_ensure`, `sy_wb_skipped`.

## Smoke test (`smoke_ref.c`)

No real slot cache and no CUDA: hand-written fakes (scripted caches: a find table, a queue of acquire results with their victims, a not-worth list; an ops table that logs every call and does the copies with `memcpy`; a real file of known bytes for `pread`). The expected log of each scenario was derived by hand from `supply.h` and the original text before the first run. Scenarios: S1 all card hits; S2 all from RAM with the card queue on (`stream_sync(0)`, `pend_take` before `remove`, the held slot released after the final sync); S3 all from the drive with one reader; S4 mix (card hit, RAM hit, two drive reads, two victims: one written back, one not worth); S5a RAM capacity 0, S5b NULL RAM cache; S6 prefill with a slab (no `pread`, `fd` -1, sources `slab + (e & 3) * raw`); S7 duplicate expert (second occurrence is a card hit); S8 card queue with a victim (`arr_wait(victim)` right after the acquire); S9 forced chunks 3 and 4 (the 4th starts at 12288 >= 10000 and is not issued); S10 computed chunks with real threads (readers 16, `pool.readers` 4, 2 experts: nchunk 2, chunk 8192, 4 jobs on 4 threads: each chunk once, bytes right, main-thread head and tail of the log); S11 eight experts on four threads; F0 call before `sy_init`; F1 EOF in a partially present expert (message, log before it, exit status 1); F2 bad fd; F3 CUDA error through `check` (nothing is called after it); F4 `check` that returns with a status: exit 1 at once. Each scenario also checks `h_eptr`, the stats (`needed`, `hit_card`, `hit_host`, `from_drive`, `written_back`, `wb_skipped`) and `t_ensure >= t_read + t_copy`.

Result (box, 2026-10-25): `PASS (73 checks, 0 failed)` with `libsupply_ref.so` md5 `7532a0ef7291735fb1dd5a1d53984390`; the same 73 checks pass against the Rust library (`../smoke_rust.c`). The build prints no warning (two warnings of the first build, a `memcpy` of a class type and a misleading indentation in the smoke test, were fixed; the object code did not change). History of the test: S10 first counted the jobs by the substring `SD(5);kd;`, which is not adjacent under threads (the fake logs each call separately); it now counts `SD(5);`.

## Not covered

One scripted world per scenario (the differential driver in `../verify` runs the same library against the real slot cache); out-of-contract calls (NULL pool cache, `pool.readers` > 32, an expert outside `0..experts-1`, a NULL slab entry with prefill); the original's undefined behaviour on overflow of `ekey`; reader jobs with more than 4 threads in the smoke test.
