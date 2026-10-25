# engcfg C++ reference (ref/)

Reference implementation of `../include/engcfg.h`: the engine's configuration / summary / card_of / force_routes text **as committed at 4d8d891**, extracted unchanged by `extract.py` into `x/*.inc`, behind thin `extern "C"` wrappers with plain-data stand-ins for everything that touches CUDA, `dlopen` or the loaders. It is the original's text behind a wrapper, not an independently verified implementation.

Files: `extract.py` (Windows, python; reads the git blob, not the working tree), `x/*.inc` + `x/MANIFEST.txt` (generated, do not edit), `engcfg_ref.cpp` (wrappers and stand-ins), `ecref.h` (reference-only extras: the composite `configure()` run), `build.sh` (box), `smoke_ref.c` (hand-derived scenario), `extract.out.txt` (raw output of the real extraction).

## Extraction (what is taken, byte for byte)

Source of truth: `git -C clover-framework show 4d8d891:clover-kimi-k3/src/code/gpu/backend-gpu.cu` (git blob 7c5fccb4efa2a36e683a997116473cfdc83d0a33, 709 lines, sha256 `cc1950a108900a62f42f9e278012753ecbe64f2928e7b95fe6958e7b9207c3f3`) and `.../decode.cuh` (blob c5e7185b5bef42e601380e1ebb744915de09f33e, 779 lines, sha256 `fc7682e9bc6a043af4ebe59b1a7cb27f750ff7e6cde4dcb8beeaed5294955dee`). The committed blobs have **LF** line endings (the Windows working-tree copies are CRLF through checkout conversion, so `Get-FileHash` of the working file is a different value); the `.inc` files are therefore LF and contain exactly the blob's bytes. No file under `src/code` was read or touched.

`extract.py` fails with exit 1 and an empty `x/` if: `git show` fails; a blob's sha256 differs from the pin; a slice's first line does not start with / last line does not end with the expected text; a function slice's braces (counted per character, comments and strings not parsed) do not balance as one block; a written slice's sha256 differs from its pin (a slice without a pin is an error unless `--print-pins`); a written file differs when read back. Controls run on Windows: a copy of the blob with `needs` -> `need` in the PREFILL message (`--src-dir`): `FAIL: ... has sha256 1a9bfb4c..., expected cc1950a1... (not the blob this table was written for)`, exit 1, no output files; `--repo C:\nonexistent`: `FAIL: git show ... failed: fatal: cannot change to ...`, exit 1. Independent re-hash with .NET (not python) of the line ranges 50, 374-387, 465-485, 487-524, 492-495, 500, 454, 185-187 and 526 of the blob: all equal to the pins.

| slice (`x/NAME.inc`) | blob | lines | bytes | sha256 |
|---|---|---|---|---|
| decls | backend-gpu.cu | 31, 43, 45-46, 48, 56, 58, 83, 99, 101-102, 175-176, 295-298 | 1290 | 1edc85067e6e9298e67b7d9583911d11bacb0176c232af220bf52bd569169463 |
| decls_dc | decode.cuh | 207, 211, 311, 437, 486, 495-496, 500, 523 | 792 | 01bd1bb55762e078a77e41ebe732bc4ba3888c73d023c0f68049ee59ae4cea5b |
| env_int | backend-gpu.cu | 50 | 114 | 7ba6ef7e1db7fdfd61f4fe8e191628c558b7e0863c37bf0a6ab3c097f14d508b |
| card_of | backend-gpu.cu | 526 | 151 | 00f53c2ec30bfc6922d679ba8a75cd6e02753d45fcaf98d38b6f8c18c2cbdcc2 |
| force_routes | backend-gpu.cu | 374-387 | 957 | ea17f5ae377c42adc49f1b3aa35778193281ee35c5c45564bac2f4d67aef81da |
| print_summary | backend-gpu.cu | 465-485 | 2518 | 70d7ecd88d61a7760bc18bfc7b77d9b8edd8e785c9e4f95878b21a955e08137e |
| configure | backend-gpu.cu | 487-524 | 2129 | a9b1c2804c71bbdd793eb40b32a4fc800a336bf182cf602ad9cdac5ed7598404 |
| cfg_cards | backend-gpu.cu | 492-495 | 239 | 790a836c7720fc8e71363815663fec5aacb841b8910579836be9ef5077bb7305 |
| cfg_policy / cfg_chunks / cfg_trace / cfg_host | backend-gpu.cu | 497 / 499 / 500 / 501 | 54 / 43 / 162 / 62 | see MANIFEST.txt |
| cfg_rc_head / cfg_lookahead / cfg_pendcap / cfg_vq | backend-gpu.cu | 502-503 / 505 / 506 / 509-511 | 74 / 56 / 50 / 182 | see MANIFEST.txt |
| cfg_rc_msg / cfg_layout / cfg_prefill_flag / cfg_prefill_check / cfg_prefill_msg / cfg_start_msg | backend-gpu.cu | 512 / 514 / 515 / 517 / 519 / 523 | 235 / 43 / 50 / 98 / 185 / 136 | see MANIFEST.txt |
| sel_{copyq,slotcache,routetable}_{parse,lib,nb,announce} (12 files) | backend-gpu.cu | parse 185-187 / 244-246 / 444-446; lib 188 / 247 / 447; nb 225-227 / 261-263 / 459-461; announce 200 / 256 / 454 | 59-289 | see MANIFEST.txt |

34 slices, 146 lines. `x/MANIFEST.txt` lists name, blob, lines, bytes and sha256 of every slice (the same values are pinned in `extract.py`).

## ADAPTED markers (everything in `engcfg_ref.cpp` that is not the original's text)

1. `TOP` / `EXPERT_RAW`: compile-time constants of the engine (moe.cuh); here `#define`d to variables set from the contract's arguments so the extracted text compiles unchanged.
2. `g_rc`: the engine's `std::vector<CallSet>` is replaced by `RcView { size(), empty(), operator[] }` holding the size and the one entry the contract passes; `struct CallSet` itself is the original line (56) in `decls.inc`.
3. `g_pool[c]`: `PoolView { ready, slots, ExpertStats st }` instead of `CardPool` (CUDA); `ExpertStats` is the original line (decode.cuh:437).
4. `g_sc`, `g_rt`: `{ bool on; }` instead of the structs of function pointers.
5. `g_host`: `HostView` with `cap_slots` and an `init(bytes)` that records the byte count; in the composite run it also derives `cap_slots = bytes / EXPERT_RAW / 32 * 32` (decode.cuh:408 and :432) and prints `K3GPU: host expert cache of %zu slots (stand-in)`. The real `init` pins memory and prints a time.
6. Globals are reset to the values of their declarations at the start of `ec_settings_from_env` and `ecref_configure` (the library runs many cases in one process; a real process starts fresh).
7. Line 491 `int have = 0; CHECK(cudaGetDeviceCount(&have));` -> `int have = devices;`.
8. `slotcache_set_policy(p)` is a macro storing `g_policy` (the original also forwards to the Rust slot cache).
9. `ec_settings_from_env` stores `env_int("K3GPU_HOST_GB", 150)` into `host_gb` (the engine evaluates that expression at lines 501 and 523).
10. `ec_print_start_line`: `env_int(...)` inside the extracted line 523 is redefined as `s->host_gb` for that one include.
11. Composite only (`ecref_configure`): `g_built[3]` stands for `K3GPU_*_BUILT`; `CHECK`, `cudaGetDeviceCount`, `slotcache_set_next_use`, `rc_nu_sel`, `route_hook`, `g_route_hook`, `g_force_routes` are stand-ins that record that they were reached.
12. Composite only: `rc_load_sel(path)` gives a table of `strlen(path) % 3` calls (so that empty and non-empty tables both occur); the real one parses a file.
13. Composite only: the three `*_select()` functions run the original parsing slices through `ec_selector_mode` / `ec_selector_lib` / `ec_selector_announce_rust` and, for `rust`, print the announce line as if `dlopen` had succeeded.

The selector slices are statements, not functions: the built variant ends in `return;` (= cpp, already printed) or falls through (= exactly `rust`); `engcfg_ref.cpp` wraps each in a lambda so the original `return;` compiles, and sets a flag after the last line. The extracted `configure()` is included whole (`x/configure.inc`) and is what `ecref_configure` runs.

## Wrappers

`ec_env_int` calls `env_int`. `ec_settings_from_env` includes the statement slices `cfg_cards`, `cfg_policy`, `cfg_chunks`, `cfg_host`, `cfg_rc_head` (it opens a block and the `if (rc && *rc) {` condition; the two closing braces and the copy of `rc` are ours), `cfg_lookahead`, `cfg_vq`, `cfg_layout`, `cfg_prefill_flag` in the original order and copies the globals into the struct. `ec_pend_cap` = slice `cfg_pendcap`. `ec_open_trace` = slice `cfg_trace` (returns `g_trace`). `ec_print_route_cache`, `ec_prefill_start`, `ec_print_start_line` = the message slices with the globals loaded from the arguments. `ec_selector_*` = the selector slices. `ec_card_of` sets `g_layout`, `g_cards` and calls the original `card_of`. `ec_force_routes` fills `g_rc_pos`, the `RcView` and `ecr_top` and calls the original `force_routes` (its function-local `static bool warned` is the process-global flag the contract describes). `ec_print_summary` loads every global from the struct and calls the original `print_summary`.

## Build (box `gpu`, `~/rustmig/engcfg/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

    python extract.py                     # Windows; add --print-pins to review new hashes
    (copy include/ and ref/ to ~/rustmig/engcfg on the box, e.g. tar over ssh; build.sh must have no CRs: sed -i 's/\r$//' build.sh)
    ssh gpu 'cd ~/rustmig/engcfg/ref && ./build.sh && ./smoke_ref'

`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -Wl,-Bsymbolic -Wall -Wextra -Wno-unused-variable -Wno-unused-function -I../include -I. engcfg_ref.cpp -o libengcfg_ref.so` (no fast-math, no `-march`), then `smoke_ref` with gcc `-O1 -std=gnu11 -Wall -Wextra`. The build printed no warning. `-Bsymbolic` keeps the composite code's calls to `ec_selector_*` inside this library even if another library with the same symbols is loaded.

## Smoke test

`smoke_ref.c` (expected values derived by hand from the header text; derivations in its comments; also used unchanged for the Rust library via `../smoke_rust.c`): env_int (12 values), settings (defaults, 20 cards over 16 devices, route cache on with policy 2, `-1` MB, policy 1 keeps the queue off, empty route cache), pend_cap, card_of (both layouts, 3 and 8 cards), selectors (5 children incl. both fatal texts), trace (unset, unwritable), prefill (both), the route cache + start lines, force_routes (a hand-dealt case, the once-only message, the three other exits, pos == size), print_summary (a case with a tie `1.25 -> 1.2`, `0.0005 -> 0.001`, and the three-lines-only case), and static asserts of the struct layouts.
Result on the box: `PASS (61 checks, 0 failed)`, exit 0, for `libengcfg_ref.so` and for the Rust `libengcfg.so`.

## Not covered

One scenario per function (the differential driver does the volume); nothing that needs CUDA; the working-tree engine sources are not read; the stand-ins (`g_host.init`, `rc_load_sel`, the select functions without `dlopen`) are mine, not the original's; `-O2` on gcc 13, not the engine's nvcc host compile.
