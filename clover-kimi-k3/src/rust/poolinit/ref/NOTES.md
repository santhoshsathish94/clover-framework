# poolinit C++ reference (ref/)

Reference implementation of `../include/poolinit.h`: the engine's `HostCache::init` and `CardPool::init` (with `struct ExpertStats` and `enum { MAX_READERS = 32 }`), extracted unchanged from `src/code/gpu/decode.cuh` **as committed at 4d8d891**, run inside a thin harness that exposes the header's C ABI. It is the original's text behind a harness, with the adaptations listed below. It is not an independently verified implementation.

Files: `extract.py` (Windows, python, reads the git blob), `poolinit_wall.inc` and `poolinit_extracted.inc` (generated, do not edit), `poolinit_ref.cpp` (harness + wrappers), `build.sh` (box), `smoke_ref.c` (hand-derived scenarios, fake ops, real memory for the host cache).

## Extracted verbatim (bytes of the committed blob, LF; the working-tree files are not read)

The source is `git show 4d8d891:clover-kimi-k3/src/code/gpu/decode.cuh`, not the working-tree file: other people are editing the working tree, and with `core.autocrlf=true` a Windows checkout has CRLF while the blob has LF (`git ls-files --eol`: `i/lf w/crlf`). So the extracted text is LF; the same text with CRLF line ends (what a Windows checkout holds) has the sha256 values in the last column. This differs from the earlier crates (planner: CRLF working-tree bytes); the content is the same, only the line ends differ.

Commit `4d8d891a2c60e937bc67cd23354cbf7303db9858`. Blobs: `decode.cuh` git object `c5e7185b5bef42e601380e1ebb744915de09f33e`, sha256 `fc7682e9bc6a043af4ebe59b1a7cb27f750ff7e6cde4dcb8beeaed5294955dee` (45669 bytes, 779 LF, 0 CR); `kernels.cuh` `6c777cb363a9cfffa51a3b942b20c7738f5ba221` sha256 `0ce0a33e...277a`; `moe.cuh` `42b1aaa49ae2235ac838eddcd07bdb5556caffea` sha256 `4af5c825...d28f`.

| block | file lines | bytes | content | sha256 (LF) | sha256 (CRLF form) |
|---|---|---|---|---|---|
| D0 | decode.cuh 202 | 121 | `wall_now_()` (the clock the original init uses) -> `poolinit_wall.inc` | dcce1b36...4269 | f09a9cfc...3351 |
| D1 | decode.cuh 403-471 | 3576 | `struct HostCache` (with `CHUNK_SLOTS` and `init`), `struct ExpertStats`, `enum { MAX_READERS = 32 }`, `struct CardPool` (with `init`), contiguous, 69 lines -> `poolinit_extracted.inc` | e0488c0c...2373 | 4031b581...0e20 |

md5 of the generated files: `poolinit_wall.inc` e99e0b459bda0020c0a27e87e0178bbc, `poolinit_extracted.inc` e74fbdc06817017c376d713f5557abea (identical on Windows and on the box; the sync to the box never touches `.inc` files).

Read but not written (replaced by the harness; the checks prove the replaced text is what the harness assumes): `kernels.cuh` CHECK macro (exact text, sha256 d77d3585...0da8) and the `moe.cuh` line with `EXPERT_RAW = 17547264` and `EXPERTS = 896`. The comment lines above `struct HostCache` and `SlotCacheSel` are not taken.

Markers: start lines found by `startswith`, each required to match exactly one line; end = the first line exactly `};` after `struct CardPool {`. Exit 1, nothing written (previous outputs are deleted first), if a marker matches 0 or >1 lines, an end marker is missing, a block is implausibly long, braces (counted per character; comments and strings are not parsed) do not balance, expected text is missing, the block does not hold exactly two `};` lines, **the ten `CHECK(` calls are not at the lines 428, 429, 454, 455, 459, 461, 463, 464, 466, 467** (the numbers the header and the Rust crate carry), the CHECK macro or the constants differ, or the blob contains a CR.

Checks (2026-10-25): (1) run on the real commit: exit 0, hashes above. (2) Independent slice: a separate python snippet cut lines 403-471 and line 202 out of the blob by number; both equal the generated files byte for byte (same sha256). (3) Negative controls on temporary git repositories made from the blobs with one change each, `--repo <tmp> --rev HEAD`: C0 unmodified copy: **exit 0**, two files written. C1 `struct HostCache {` renamed: **exit 1**, `expected exactly one line matching 'struct HostCache {', found 0`, no output file. C2 one line inserted before the first CHECK: **exit 1**, `CHECK sites are at lines [429, 430, 455, ...]`, no file. C3 `MADV_HUGEPAGE` changed to `MADV_NOHUGEPAGE`: **exit 1**, `D1 lacks expected text`, no file. C4 the CHECK macro's `exit(1)` changed to `exit(2)`: **exit 1**, `kernels.cuh CHECK macro is not the expected text`, no file. C5 the decode.cuh blob with CRLF: **exit 1**, `contains CR characters`, no file. C6 nonexistent revision: **exit 1** (git's usage text is the last line), no file. C7 a directory that is not a repository: **exit 1**, `not a git repository`, no file. (C6's message is git's, not the script's; the exit status and the absence of output are what was checked.)

## Adaptations (each also marked `ADAPTED (n)` in poolinit_ref.cpp)

1. `EXPERT_RAW` and `EXPERTS` are the configuration's values (`size_t g_expert_raw`, `int g_experts`, behind `#define`); in the original they are enum constants (moe.cuh:5). The text uses them as `size_t / EXPERT_RAW`, `(size_t)n * EXPERT_RAW`, `(double)EXPERT_RAW`, a `size_t` argument, `EXPERTS * sizeof(void *)` and `h_eptr.assign(EXPERTS, ...)`; for `expert_raw < 2^31` (the real value and every value of the tests) a `size_t` variable gives the same values as the enum.
2. The CUDA runtime is replaced by the ops table (`cudaSetDevice`, `cudaMemGetInfo`, `cudaMalloc` as a template over `T **` like the real header, `cudaHostRegister`, `cudaHostAlloc`, `cudaStreamCreateWithFlags`, `cudaEventCreateWithFlags`, `cudaGetErrorString`); each passes the address of the caller's variable straight to the op; flags are `PI_*` macros named like the CUDA ones (values 0, 1, 1, 2 read from `/usr/local/cuda/include/driver_types.h` on the box).
3. `CHECK`: the kernels.cuh macro with `__FILE__` replaced by `g_check_file` (cfg.src_file or "decode.cuh") and `__LINE__` replaced by `__LINE__ + 402`, because the extracted block is a separate file whose lines start at 1 (the first version of the harness used `#line 403` before the `#include`, which does not carry into the included file; the smoke test caught it: it printed lines 52, 53, 57 ...).
4. `SlotCacheSel` is minimal: `add_spare` forwards to `ops.sc_add_spare(rs, p)`. The original's `R()` (creates the cache on first use) and the C++-cache branch (`cpp.spare.push_back`) are not reproduced: the cache is the caller's handle.
5. Static checks that the header's mirrors equal the original: `MAX_READERS == PI_MAX_READERS`, `HostCache::CHUNK_SLOTS == PI_CHUNK_SLOTS`, size and field offsets of `ExpertStats` equal `pi_expert_stats`.
6. Argument checks the original does not have (NULL `ops` / `cfg` / `out`, a NULL function pointer the call uses, `expert_raw == 0`): `poolinit: <what>\n`, `exit(1)`. Same texts as the Rust crate.
7. The out struct is filled from the original's object at the end of the call; and, if the process exits in the middle (a failed CHECK, "cannot map", "only N slots"), by an `atexit` handler registered at the start of every call, so that a handler registered before the call sees what the original object holds. The object is `new`-ed with value-initialisation (arrays zero); only non-null `rstream` / `bounce` entries below `readers` are copied in the exit case, so entries that were never assigned stay as the caller left them (a fake that returned a NULL handle successfully would not be copied in that case; the driver's fakes never do). The handler is disarmed before the call returns. **This is a harness mechanism, not original text:** it is how the contract's "stored at the moments the original assigns the members" is made observable for the reference.
8. `h_eptr.assign(EXPERTS, nullptr)` stays in the extracted `CardPool::init` and runs on the harness's private object; in the contract that line is the caller's, so nothing of it reaches `out`.
9. `wall_now_()` comes from its own extracted file (it is declared at line 202, before the other block); the two files are included in file order.

Wrappers: `pi_host_init` copies the ops table into a global (the text calls `g_ops` through the shims), makes a `HostCache`, sets `c.rs = cfg->cache`, calls `init(cfg->bytes)`, copies `cap_slots` / `ready` to `out`. `pi_pool_init` likewise with `CardPool::init(device, reserve_bytes, nreaders)` and copies `dev, slots, readers, ready, base, copy, rstream[0..readers), bounce[0..readers), written_back, d_eptr`; `out->st` is never written.

## Build (box `gpu`, `~/rustmig/poolinit/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

```
python extract.py --outdir .        # Windows, from the repository (reads the blob of 4d8d891)
ssh gpu 'cd ~/rustmig/poolinit/ref && ./build.sh && ./smoke_ref'      # after copying include/, ref/ (with the .inc files untouched by sed)
```
`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -Wall -Wextra -pthread -I../include -I../../slotcache/include poolinit_ref.cpp -o libpoolinit_ref.so`, then `smoke_ref` with gcc `-O1 -std=gnu11 -Wall -Wextra -pthread`. The build printed no warning. `nm -D` exports exactly `pi_host_init` and `pi_pool_init`. Hashes: `libpoolinit_ref.so` md5 `070e70c16e120a0e7613afde8169d102` (the same after every rebuild of the final source; earlier builds with the `#line` version and the one-time `atexit` registration differed: 5aa2299a...); `poolinit_ref.cpp` md5 `82e9208e097d294e25fb6d5969b396d3`, `smoke_ref.c` md5 `12c182030a687c620fd0977e4d06aa56` (box, CRs stripped by `sed`).

## Smoke test (`smoke_ref.c`)

Hand-written fakes (recording ops with arenas, a failure at the k-th CUDA call, an `atexit` dump in the forked child), real memory for the host cache. The expected values were derived by hand from decode.cuh before the first run (derivations in the comments): A a card pool with 3 readers (free = reserve + 100 experts + 5000 bytes: 100 slots, the 112 calls in order, addresses of the spares, streams, bounce slots, event, pointer table, untouched entries and `st`); A2 exactly 64 slots with 40 readers (clamped to 32, 134 calls); A3 negative readers; B 63 slots and B2 free below the reserve (exit 1, the original's text, `out` already holding `dev` and `slots`); E the k-th CUDA call failing for k = 1..12 (lines 454 455 459 461 463 464 463 464 463 464 466 467, events and spares seen at the exit), E2 no `src_file`, E3 `expert_raw == 0` and a NULL op; C a host cache of 196 slots (6 chunks of 2 MiB, 4 slots dropped: calls, spare order chunk 5 slot 31 down to chunk 0 slot 0, page-aligned distinct zero regions, the summary line `K3GPU: host expert cache of 192 slots (0.0 GB) pinned in <time> s`); D no chunk (31 slots and fewer than one slot); H-E set_device failing (line 428, `cap_slots` 196 not yet trimmed, `ready` 1) and the third register failing (line 429).

Result (box): `PASS (45 checks, 0 failed)`. History: the first run had 17 failures. Cause 1, a harness bug: `__LINE__` inside the included file (adaptation 3). Cause 2, a test-order artefact: the `atexit` handler of the reference was registered once, so it ran after the test's own handler; fixed by registering at every call (adaptation 7).

## Not covered

- A real CUDA runtime, the real `HostCache::init` on the engine's 561 MB chunks (the contract's `expert_raw` is a parameter; with the real constants only the arithmetic and the card pool through fake ops are exercised, see `../NOTES.md`).
- Mutants of the harness (`poolinit_ref.cpp`) were not run; no sanitizer run of the reference.
- `SlotCacheSel` (adaptation 4) and `h_eptr` / `ready` of the engine's `CardPool` are not part of the reference.
- Out-of-contract values: negative `experts` (the original's `assign` would throw), a chunk table too large to allocate, `expert_raw >= 2^31` (the original's enum is an `int`).

## Contract notes for the owner (header not edited by the reference)

1. `__FILE__` in the CHECK messages is a build property of the engine; the header makes it a configuration string. The line numbers are decode.cuh@4d8d891's: they describe that commit, and will be wrong as soon as the engine file changes (the numbers are constants in `host.rs` / `pool.rs` / the harness).
2. The 15 helper threads are always started, even for 0 chunks (the original does).
3. The original does not unmap or unregister anything; neither implementation does.
4. `n_chunks` is derived from `cap_slots`, so up to 31 slots of `bytes / expert_raw` are dropped silently; the summary line shows the trimmed number.
