# routetable C++ reference (ref/)

Reference implementation of `../include/routetable.h`: the engine's route-table text (`struct CallSet`, the route-cache globals, `rc_nu`, `rc_load`), extracted unchanged from `src/code/gpu/backend-gpu.cu`, behind thin `extern "C"` wrappers. It is the original's text behind a wrapper, not an independently verified implementation.

Files: `extract.py` (Windows, python), `routetable_extracted.inc` (generated, do not edit), `routetable_ref.cpp` (wrappers), `build.sh` (box), `smoke_ref.c` (hand-derived scenario).

## The one adaptation

The original's `ekey(layer, e)` is `layer * EXPERTS + e` with a compile-time `EXPERTS` (decode.cuh:201). `routetable_ref.cpp` defines `static int g_experts_rt;` (set by `rt_load` from its `experts` argument) and `static inline int ekey(int layer, int e) { return layer * g_experts_rt + e; }`. Nothing else in the extracted text is changed.

## Extracted verbatim (bytes, CRLF kept; `backend-gpu.cu` sha256 6ef8cb64...6a43, read-only, hash identical before and after)

| block | backend-gpu.cu lines | bytes | sha256 |
|---|---|---|---|
| `struct CallSet`, `g_rc`, `g_rc_pos/g_rc_planned/g_pend_cap`, `g_rc_cur`, `g_uses`, `g_use_ptr` | 52-57 | 411 | 5bb40678646335127fc180a13c7ef2766308c855ad4e316fb3ff0b7c1e7cb59b |
| comment line + `rc_nu` | 58-66 | 371 | ddb64118b39835a80a8915a201634d96809b4601bfdb6957152aa2e2f7a59bb7 |
| `rc_load` | 369-387 | 854 | c5a47fcd220870a603503f670ac9f8afc98dba053d87c5321a8e2b09aa4f0435 |

`routetable_extracted.inc` = A + B + C, nothing added (1636 bytes, 34 lines, all CRLF), sha256 `86328a398cb4ccc1395f8cd40665b786ff2c8b15942462d663b24a10059ccc5f` (same value printed by `sha256sum` on the box copy).

Markers: A from the unique line starting `struct CallSet {` to the line starting `static std::unordered_map<int, size_t> g_use_ptr;`; B from the unique line `static int rc_nu(int key)` (the line above must start with `// next call after the current one that needs the expert`) to the first following line that is exactly `}`; C likewise from `static void rc_load(const char *path)`. extract.py exits 1 (nothing written; a previous `--out` is deleted first) if a start marker matches 0 or >1 lines, an end marker is missing, a block is implausibly long, braces (counted per character, comments/strings not parsed) do not balance or close early, expected text is missing, or the blocks are out of order.

Negative controls (run 2026-10-22 on Windows): temp copy with `static int rc_nu(int key)` renamed `rc_nuX` -> `FAIL: expected exactly one line matching 'static int rc_nu(int key)', found 0`, **exit 1**, no output file; `--src C:\nonexistent` -> `FAIL: source file not found: C:\nonexistent\backend-gpu.cu`, **exit 1**, no output file. (An earlier first run of my own extract.py failed on the real source because the brace check treated the `rc_nu` signature line, whose `{` is on the next line, as a closed block; fixed, then everything above was re-run.)

## Build (box `gpu`, `~/rustmig/routetable/ref`, g++ 13.3.0 Ubuntu 24.04, no GPU used)

```
python extract.py --src C:\personal\oss\clover-framework\clover-kimi-k3\src\code\gpu --out routetable_extracted.inc   # Windows
scp include/routetable.h gpu:rustmig/routetable/include/ ; scp ref/{routetable_extracted.inc,routetable_ref.cpp,smoke_ref.c,build.sh} gpu:rustmig/routetable/ref/
ssh gpu 'cd ~/rustmig/routetable/ref && sed -i "s/\r$//" build.sh && ./build.sh && ./smoke_ref'
```
`build.sh`: `g++ -O2 -std=c++17 -fPIC -shared -I../include routetable_ref.cpp -o libroutetable_ref.so` (no fast-math, no `-march`), then the smoke binary with `gcc -O1 -std=gnu11 -Wall -Wextra ... -Wl,-rpath,'$ORIGIN'`. Build printed nothing, exit 0.

Hashes: `routetable.h` md5 929d801fffeada4a2874dfc4addb7268 (Windows and box equal, copied unchanged); `routetable_ref.cpp` md5 4c12cba1657a6101a3335e2f4d580fee (Windows and box equal); `libroutetable_ref.so` md5 09c4d2b7bd2013b24aa8d3b808fc352e; `routetable_extracted.inc` md5 ec0d604f5b393491522a168e1a1c2588 (box). `nm -D` exports exactly the 8 `rt_*` functions of the header.

## Wrappers (glue written by me)

`rt_load`: sets `g_experts_rt`, clears `g_rc`, `g_uses`, `g_use_ptr`, sets `g_rc_cur = -1`, calls the original `rc_load`, returns a pointer to a static dummy `rt_table`. `rt_free`: clears the same globals and sets `g_rc_cur = -1`. `rt_calls`/`rt_layer` read `g_rc`; `rt_experts`/`rt_counts` return `vector::data()` and `*n = size()`; `rt_set_current` assigns `g_rc_cur`; `rt_next_use` calls `rc_nu`. The handle argument is ignored everywhere (one live table).

## Smoke test

`smoke_ref.c` (expected values derived by hand from the contract text; derivations in its comments): a 7-call trace in `/tmp` (normal lines with counts, no counts, fewer counts than experts, empty line, line ending at a space, `3 5 7 x 9` non-digit stop, last line without newline and with an expert twice), loaded with experts=896; checks of calls, layers, experts, counts; `rt_next_use` over 14 keys with `rt_set_current` values 0, 4, 1, -1, 5, 6, -1 (decreasing steps show the pointer never goes back and that it is per key); free + reload resets (906 -> 0 again); `experts`=100 changes the keys; missing file in a child process.
Result: `PASS (43 checks, 0 failed)`, exit 0.
Controls (/tmp copies on the box, built with `build.sh`, removed afterwards; shipped files untouched, hashes re-checked): M1 `1000000000` -> `1000000001` in the extracted text (2 places): `FAIL (43 checks, 12 failed)`, exit 1; M2 `l[p] <= g_rc_cur` -> `l[p] < g_rc_cur`: `FAIL (43 checks, 11 failed)`, exit 1.

## Not covered

One scenario; no differential test against the engine build; not run under sanitizers or with `-Wall` on the library (the extracted globals `g_rc_pos`, `g_rc_planned`, `g_pend_cap` are unused in this unit, so a `-Wall` warning is expected but was not checked). Parsing paths read from the code and not run: a count after `:` with a sign, leading space or no digit (`strtol(p + 1, ...)` accepts `3: 5` -> 5, `3:-2` -> -2, and `3:x` pushes 0 and consumes the `:`), strtol overflow (clamps to `LONG_MAX`, `(int)` cast gives -1), a file ending in a newline (no extra empty call, from `getline` returning -1 at EOF), embedded NUL bytes, tab characters (only `' '` is skipped between numbers), large files, `layer * experts` overflowing `int` (signed overflow in the original too).

## Contract notes for the owner (header not edited)

1. Line numbers in the header (52, 53-57, 59-66) match the file; the leading comment of `rc_nu` is line 58.
2. The header says the stop is "the first character that is not an ASCII digit" after skipping spaces; the original skips only `' '`, so a tab stops the line. Consistent with the header's word "spaces", but a Rust port using `is_ascii_whitespace` would differ.
3. "read a count with strtol after it" does not say that a count with no digits is recorded as 0 (and the `:` consumed), nor that sign and leading whitespace are accepted (from the code, not run).
4. The first two `strtol` calls also skip leading whitespace including the line's own `\n`, which is why an empty line gives layer 0 and a layer-only line gives no experts; the header states the outcome, not this mechanism.
5. `rt_load` after `rt_free` is not described: the reference resets the current call number to -1 and drops every pointer (checked), which equals a fresh process.
6. The key `layer * experts + expert` aliases when an expert id is >= `experts` (e.g. `experts`=100, layer 1 expert 110 and layer 2 expert 10 share key 210); the header does not mention it. The original has the same aliasing.
