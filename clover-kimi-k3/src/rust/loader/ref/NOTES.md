# loader C++ reference (ref/)

Reference implementation of `../include/loader.h`: the engine's loader text, extracted unchanged from `src/code/gpu/backend-gpu.cu` and `kernels.cuh` **as committed at 4d8d891**, run inside a thin harness that exposes the header's C ABI and sends every CUDA / model call to the contract's callbacks. It is the original's text behind a harness, with the adaptations listed below. It is not an independently verified implementation.

Files: `extract.py` (Windows, python), `loader_a.inc` / `loader_b.inc` (generated, do not edit), `loader_ref.cpp` (harness + C entry points), `build.sh` (box), `smoke_ref.c` + `run_smoke.sh` (hand-derived scenarios, recording callbacks, six modes).

## Extracted verbatim (bytes, LF as committed; sources read with `git show 4d8d891:...`, nothing in the working tree is read)

Sources: `backend-gpu.cu` blob sha256 `cc1950a108900a62f42f9e278012753ecbe64f2928e7b95fe6958e7b9207c3f3` (41848 bytes), `kernels.cuh` blob sha256 `0ce0a33edd8ffde2a0f2fc3dbff8b2028cae07ba210874660a123271dde6277a` (29623 bytes). `extract.py` pins both and every block hash below and exits 1 if one differs (`--no-pin` to explore another version).

| block | file lines | bytes | content | sha256 |
|---|---|---|---|---|
| K1 | kernels.cuh 17 | 168 | `#define CHECK(x) ...` | 1089b632...f917 |
| B1 | backend-gpu.cu 31 | 39 | `enum { MAX_CARDS = 8, N_LAYERS = 92 };` | 33e69051...e3d2 |
| B2 | backend-gpu.cu 538-572 | 1489 | `struct Loader`, `g_loader[]`, `pending_mu()`, `pending_cv()`, `g_pending`, `loader_submit`, `loader_wait` | 2515fce1...d667 |
| B3 | backend-gpu.cu 574-610 | 1884 | `open_layer` (37 lines) | f53ed3f0...bbcc |

`loader_a.inc` = K1 B1 B2 (1696 bytes, 37 lines, md5 `1549311f83dc4dc8cc05c5587d9b06ea`). `loader_b.inc` = B3 with the five-line state group replaced by one line (1763 bytes, 33 lines, md5 `d9de653d9b8aeb705b6e890e9117f272`); identical on Windows and on the box. Independent check (a second python slice of the committed blob by those line numbers): `loader_a.inc` equals K1 + lines 31 + 538-572 byte for byte; `loader_b.inc` lines 1-12 and 14-33 equal lines 1-12 and 18-37 of the blob slice 574-610 (the replaced span is source lines 586-590, sha256 `a2ac2952...722b`).

Markers: start lines by `startswith`, each required to match exactly one line; B2 ends at the first line exactly `}` after `static void loader_wait()` and must contain exactly two top-level `}` lines; B3 ends at the first line exactly `}` after the signature; braces counted per character; expected text strings checked in every block; the five state-group lines must be found exactly once with exactly this text.

NOT taken: `configure`, `card_of`, `ensure_work`, `layer_load_*`, `kda_state_alloc`, `forward`, `Handle`, `g_handles`, `g_slab`, any CUDA call.

Negative controls of `extract.py` (`python extract_controls.py`, Windows, 2026-10-25; each case works on a temp copy, outputs go to a temp dir, the shipped `.inc` files are untouched): `static void loader_wait()` renamed -> `FAIL: expected exactly one line matching 'static void loader_wait()', found 0`, exit 1, no output file; `struct Loader {` renamed -> exit 1, no output; one state-group line edited -> `FAIL: the five-line state group of open_layer was found 0 times`, exit 1; `if (rd <= 0) {` changed to `if (rd < 0) {` -> `FAIL: B3 lacks expected text ...`, exit 1; one byte changed with the pins on -> `FAIL: source backend-gpu.cu sha256 ... differs from the pinned ...`, exit 1; a missing source directory -> `FAIL: source file not found`, exit 1; an untouched copy with the pins on -> exit 0; the default git mode -> exit 0 and both outputs equal the shipped `.inc` files byte for byte. Result line: `extract controls: ALL AS EXPECTED`. (Every changed-source case printed `source changed=True`, so none of the failures is a "file not found" in disguise.)

## Adaptations (each also marked `ADAPTED (n)` in loader_ref.cpp)

1. The CUDA names the extracted text uses are fakes: `cudaError_t`, `cudaSuccess`, `cudaGetErrorString` (so `CHECK` compiles unchanged), `cudaSetDevice(card)` -> `cb.set_device`, `cudaHostAlloc((void **)&g_slab[L], n, cudaHostAllocDefault)` -> `cb.host_alloc(ctx, &slab[L], n)`. The fakes always return success: the callbacks own their failure handling (header).
2. `EXPERT_RAW` is `#define EXPERT_RAW g_expert_raw` (a `long` set from `cfg.expert_raw`); the original is the enum constant 17547264 (moe.cuh:5). `g_prefill` is a plain bool set from `cfg.prefill`; `g_slab` is a pointer to the caller's table (`cfg.slab`) where the original has `static unsigned char *g_slab[93]` (decode.cuh:504).
3. `Handle` is reduced to `L, card, k_in, fd` (each an `IntRef`: writes go through to the caller's int, a NULL pointer keeps a local copy only, reads return the local copy), `y.user` and `user` (the caller's opaque handle). `g_handles[L] = h` stores `h->user` into the caller's table through a small proxy.
4. `configure()` is an empty function; `card_of(L)` returns the card of the request (a thread-local set by `ld_open_layer` just before it calls `open_layer`).
5. `ensure_work`, `layer_load_trunk`, `layer_load_expert_tables` call `cb.ensure_work`, `cb.load_trunk`, `cb.load_expert_tables` (the Layer stub carries the caller's handle).
6. **Text replaced, not extracted:** the five lines `h->mla = h->y.mla; if (h->mla) { h->mc.cap = positions; cudaMalloc x2 } else kda_state_alloc(h->kda);` are replaced by `loader_state_alloc(h, positions);` (= `cb.alloc_state`). They are model/CUDA code (several calls and fields of types that stay in C++); extract.py checks their exact text and reduces the adapted block back to the original.
7. Guards the original does not have (header, "deliberate differences"): call before `ld_init`, NULL arguments, card outside 0..7, layer > 92 with prefill, a second `ld_init` ignored. These are written in the harness, identical in text to the Rust crate's; they are not a comparison with the original (which has out-of-bounds access there).
8. `ld_submit(card, fn, arg)` wraps `fn(arg)` in a `std::function<void()>` and calls the extracted `loader_submit`. `ld_wait` calls `loader_wait`. Nothing else in the loader text is touched: the unlocked `if (!g_loader[card]) g_loader[card] = new Loader;` race is still in the reference (header, difference 1).

## Build (box `gpu`, `~/rustmig/loader/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

```
python extract.py                      # Windows; reads the committed blobs with git show; writes loader_a.inc loader_b.inc here
pwsh verify/sync.ps1                   # copies the crate to the box, strips CRs from scripts and C sources there
ssh gpu 'cd ~/rustmig/loader/ref && ./build.sh && ./run_smoke.sh ./smoke_ref'
```
`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -pthread -Wall -Wextra -D_FORTIFY_SOURCE=0 -I../include loader_ref.cpp -o libloader_ref.so`, then the smoke binary with gcc `-O1 -std=gnu11 -Wall -Wextra -pthread`. `-D_FORTIFY_SOURCE=0`: with Ubuntu's default fortification `open()` becomes `__open_2()`, which the differential driver's interposer would not see. No warning printed. No `-march`, no fast-math.

## Smoke test (`smoke_ref.c`, mode argument)

Recording callbacks (a log string), no CUDA. Expected strings were derived by hand from the original text before the first run: `basic` (three jobs on card 2 -> `set_device 2` first, then the jobs in order; no second thread start; three cards on a timed barrier; a job that submits a job and `ld_wait`); `noprefill` and `prefill` (layer 0: bookkeeping at once, handles entry, callback order `set_device, set_device, ensure_work, trunk, state`, fd untouched; layer 7 with a 5-record file: `... tables ... state`, with prefill `host_alloc 32768`, fd opened and readable at an aligned offset, slab = first 4 records, without prefill slab untouched); `missing`, `eof`, `badcard` (child process: exit status 1 and the exact stderr text). Result on the box: PASS in all six modes (5, 9, 10, 2, 2, 2 checks), run after a rebuild with the `build.sh` above. First run passed; that means the expectations and the reference agree, not that either is right: the Rust crate and the differential driver are the cross-checks.
