# copyq C++ reference (ref/)

Reference implementation of `../include/copyq.h`, assembled by `extract.py` from the engine sources as they are in the working tree (checked 2026-10-21; `kernels.cuh` has an uncommitted change elsewhere in the file, not at the extracted line). `copyq_ref.cu` is generated: do not edit, re-run `python extract.py`.

Files: `extract.py` (assembles `copyq_ref.cu`), `build.sh` (box: builds `libcopyq_ref.so`), `smoke_ref.cpp` (dlopen smoke test).

## Extracted verbatim (bytes, CRLF kept; each block is checked to appear exactly once in the output)

| block | file | lines |
|---|---|---|
| `CHECK` macro | `src/code/gpu/kernels.cuh` | 17 |
| `enum { LATENT ... EXPERT_RAW = 17547264 ... }` | `src/code/gpu/moe.cuh` | 5 |
| `enum { MAX_CARDS = 8, N_LAYERS = 92 }` | `src/code/gpu/backend-gpu.cu` | 19 |
| `act_gate()` | `src/code/gpu/decode.cuh` | 389 |
| `struct PfJob` | `src/code/gpu/backend-gpu.cu` | 68-72 |
| `struct PfQueue` | `src/code/gpu/backend-gpu.cu` | 73 |
| `pfq()` | `src/code/gpu/backend-gpu.cu` | 74 |
| `pf_enqueue()` incl. the worker lambda | `src/code/gpu/backend-gpu.cu` | 76-130 |

Markers are line prefixes (`static void pf_enqueue(const PfJob &j)` to the first line that is exactly `}`, etc.). The script exits 1 without writing if a marker matches 0 or more than 1 lines, if the `pf_enqueue` braces do not balance, or if the body no longer contains `pread(`, `cudaMemcpyAsync`, `act_gate();` and the three `->state.store(1` calls. It also exits 1 if the originals of `Pending`, `Arrival` (decode.cuh) and `Rel` (backend-gpu.cu) no longer begin `std::atomic<int> state;`. Negative control run: renaming the `pf_enqueue` marker in a temp copy of the source gave exit 1 and no output file; a missing source dir gave exit 1.

## Glue written for the reference

- Includes, `#include "copyq.h"`.
- `struct Pending`, `Arrival`, `Rel`: state-only (`std::atomic<int> state;`), `static_assert(sizeof == 4)`.
- `static_assert(sizeof(std::atomic<int>)==4 && is_always_lock_free)`.
- `g_cards`, `g_readers` (init 16), `g_prefill`, `g_slab[93]` as plain globals; `g_act_hold` as a small proxy object `ActHold` with `load(order)` over the caller's `const std::atomic<int>*` (NULL = never held).
- `cq_init`: checks `cfg->expert_raw == EXPERT_RAW` (else message + `exit(1)`), copies readers/cards/prefill, copies the 93 slab pointers (`const_cast`), sets the act_hold address.
- `cq_enqueue`: builds a `PfJob` from the `cq_job` (`int*` flags `reinterpret_cast` to `Pending*`/`Arrival*`/`Rel*`; `arr`/`rel` as non-owning `shared_ptr` via the aliasing constructor over an empty owner; empty when the `*_state` pointer is NULL), then calls the extracted `pf_enqueue`.

## Where the glue can differ from the original (specific)

1. **Flag objects.** Original: `Pending` and `Rel` are heap objects with a `slot` member and constructors that set `state` to 0; `Arrival` is held by `shared_ptr` with a constructor setting 0. Here: the caller's `int` is reinterpreted as a 4-byte state-only struct. The worker only touches `->state`, but the cast is not strictly conforming C++ (no object of that type lives there). The ref never initialises the flags: the caller must pass ints that are 0.
2. **Lifetime.** Original `shared_ptr<Arrival>`/`shared_ptr<Rel>` keep the object alive for the worker after the main thread dropped its reference (`g_arr.erase`). Here ownership is none: the caller's ints must stay valid until their flag is seen as 1.
3. **act_hold.** Original is a global `std::atomic<int>`; here a pointer read with `memory_order_acquire` each iteration (same `act_gate` text), NULL treated as 0.
4. **Config globals** are set by `cq_init` and read by workers as plain variables, as in the original (set before workers start). Calling `cq_init` after a queue has started is a race here, not guarded; the original has no equivalent call. `readers = 0` gives a queue with 0 threads (original: same if `K3GPU_READERS=0`).
5. **`expert_raw` check** is new: the original has a compile-time constant; a different value exits instead of being ignored.
6. **`CHECK` messages** print `__FILE__`/`__LINE__` of `copyq_ref.cu`, not of `backend-gpu.cu`.
7. **No validation** was added: `dev` outside 0..7 for kinds 1/2 indexes past the queue array (as the original); any `kind` other than 0 and 1 takes the kind-2 path (as the original); `arr_state == NULL` on kind 1/2 dereferences null (as the original `jb.arr`); `cfg.slab == NULL` with `prefill != 0` reads null pointers (original: zero-initialised `g_slab`).
8. **`host_src`** is `const` in the contract; `const_cast` to the original's `unsigned char *`.
9. **Build.** The original is one translation unit with decode.cuh, model.cuh etc. and the flags of `scripts/run-model-gpu.sh` (not read by me). The ref is built alone with `nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -fPIC -Xcompiler -pthread -shared` (`-pthread` added by me). Line endings: extracted blocks are CRLF as in the source, glue is written CRLF too.
10. One library instance holds one set of queues (`static` as in the original); worker threads are never stopped.

## Run record (box `gpu`, card 7 only, 2026-10-21)

Before the run: `nvidia-smi --query-compute-apps` listed no processes, memory used 0 MiB on all 8 cards, no `model-gpu|k3gpu|g3run` process. `~/rustmig/copyq/ref/build.sh` built `libcopyq_ref.so` with no compiler output; `nm -D` shows `cq_init` and `cq_enqueue`. `smoke_ref ./libcopyq_ref.so 7`: kind 0 (file -> pinned RAM slot), kind 2 with victim copy (pinned slot -> card 7, old card bytes down to `victim_host`, `wb_state`/`rel_state`/`arr_state` all 1), kind 1 (file record 2 -> bounce -> card 7, replacing record 0 on the card): all read-backs byte-identical (17,547,264 bytes each), exit 0.

Not run: `cfg.prefill != 0`; `act_hold != 0` (hold and release); several jobs per queue, FIFO order or concurrent enqueue; failure paths (`pread` error, CUDA error, `expert_raw` mismatch); other cards than 7; the original code compiled and run side by side with this file; timings.
