# slotcache C++ reference (ref/)

Reference implementation of `../include/slotcache.h`: the engine's `SlotCache` text, extracted unchanged from `src/code/gpu/decode.cuh`, behind thin `extern "C"` wrappers. It is the original's text behind a wrapper, not an independently verified implementation.

Files: `extract.py` (Windows, python), `slotcache_extracted.inc` (generated, do not edit), `slotcache_ref.cpp` (wrappers), `build.sh` (box), `smoke_ref.c` (hand-derived scenario).

## Extracted verbatim (bytes, CRLF kept; `decode.cuh` sha256 000fa2b1...951f, 665 lines, all CRLF)

| block | decode.cuh lines | bytes | sha256 |
|---|---|---|---|
| comments + `g_policy`, `g_chunks`, comments + `g_nu_fn` | 200-206 | 891 | 31c15dfaae704d1c15af266c8044ecfd03c1cae52d050443c5e67ba402feccec |
| `struct SlotCache { ... };` | 208-285 | 4274 | d6186894d04f4339f5c5bd4f14453c3cf55656f1fb56368430b2ad444105309b |

`slotcache_extracted.inc` = block A then block B, nothing added (5165 bytes, 85 CRLF lines), sha256 `b1392981d7071010513a221c53c7ac6493c6626f2b91524169a622a164aec4f7`. Line 207 (`g_vq`) is not taken. `g_chunks` shares a line with `g_policy` and comes along (unused; `-Wall` warns, the shipped flags do not).

Markers: block A starts at the unique line beginning `// Slots with replacement of the least recently used entry` and ends at the line beginning `static int (*g_nu_fn)(int key) = nullptr;`; block B starts at the unique line `struct SlotCache {` and ends at the first following line that is exactly `};`. extract.py exits 1 (nothing written; a previous `--out` is deleted first) if a start marker matches 0 or >1 lines, an end marker is missing, a block is implausibly long, braces (counted per character, comments/strings not parsed) do not balance or close early, or an expected member is missing.

Negative controls (run 2026-10-21 on Windows): temp copy with `struct SlotCache {` renamed `struct SlotCacheX {` -> `FAIL: expected exactly one line matching 'struct SlotCache {', found 0`, **exit 1**, no output file; `--src C:\nonexistent` -> `FAIL: source file not found`, **exit 1**, no output file. `decode.cuh` hash unchanged afterwards.

## Build (box `gpu`, `~/rustmig/slotcache/ref`, g++ 13.3.0, no GPU used)

```
python extract.py --src C:\personal\oss\clover-framework\clover-kimi-k3\src\code\gpu --out slotcache_extracted.inc   # Windows
scp include/slotcache.h gpu:rustmig/slotcache/include/ ; scp ref/{slotcache_extracted.inc,slotcache_ref.cpp,smoke_ref.c,build.sh} gpu:rustmig/slotcache/ref/
ssh gpu 'cd ~/rustmig/slotcache/ref && sed -i "s/\r$//" build.sh && ./build.sh && ./smoke_ref'
```
`build.sh`: `g++ -O3 -std=c++17 -fPIC -shared -I../include slotcache_ref.cpp -o libslotcache_ref.so` (no `-ffast-math`, no `-march`), then the smoke binary with `gcc -O1 -std=gnu11 -Wall -Wextra ... -Wl,-rpath,'$ORIGIN'`. Build printed nothing. A separate `-Wall -Wextra` compile of the library (output discarded) gave one warning: `g_chunks` defined but not used.

Hashes (box copies compared equal to the Windows files): `slotcache.h` ee985b7f1b306b6616f18e91a5ea0e4cc4cf8de3b37f13d5e6a0eb0ed2b09735 (copied unchanged); `slotcache_ref.cpp` 09381f0d120e97d5f1804bd7bc806063bd98cad582e652900d239738091e2b94; `smoke_ref.c` fd620f33e8c06c25eeab9686dc610ddffa469b1c29c81aaf4aaee163ce426bd3; `libslotcache_ref.so` sha256 dce8b61182c2e76ec5fe3d6d5512be9005cdb4f9e99ee527cf7f3b21d10c38e7 (md5 2d325e28d106ae8f130a4bed659104e0). `nm -D` exports exactly the 17 `sc_*` functions of the header.

## Wrappers (glue written by me)

`sc_cache*` is an opaque pointer to a `SlotCache` (`reinterpret_cast`); `sc_new` = `new SlotCache`. `sc_add_spare`/`sc_release` = `spare.push_back` (release calls the original `release`). `sc_contains` = `map.count`, `sc_size` = `map.size()`. `sc_acquire`/`sc_try_acquire` build an `unordered_set<int>` from `prot[0..nprot)` and call the original. `sc_worth` returns 1/0. Snapshots walk `lru` (looking each key up in `map`), `spare`, `by_next` and write up to `cap` records, returning the total. `sc_set_policy`/`sc_set_next_use` assign `g_policy`/`g_nu_fn`. The extracted logic is untouched.

## Smoke test

`smoke_ref.c` (expected values derived by hand from the contract text, derivations in its comments): policy 0 LIFO spares, find, LRU victim, protected victim skip, remove/release; policy 1 heat decay (`5*0.8f+3 = 7.0f` exactly), lowest-heat victim, tie rule, prot skip; policy 2 with next-use (ordered-set contents, worth, victim choice, try_acquire both outcomes, touch re-keying, function unset -> heat rule erasing the pair); the "too few slots" message and exit status 1 (child process).
Result: `PASS (55 checks, 0 failed)`, exit 0. Negative control: a /tmp copy of the extracted text with `* 0.8f + w` changed to `* 0.9f + w` gives `FAIL (55 checks, 3 failed)`, exit 1 (checks at lines 84, 85, 89). The mutant was built and removed from /tmp; nothing shipped changed.

## Not covered

One smoke scenario only; no differential test against the engine build; `sc_find`/`sc_touch` on absent keys covered once; a key acquired twice, double release, and `prot` duplicates are not tested (outside the contract or trivially a set); not run under sanitizers; no concurrency (contract says single-threaded).

## Contract notes for the owner (header not edited; items 2-4 are from reading the code, not run)

1. The header cites `struct SlotCache` as lines 208-286 and the globals as 203-206; in the file the struct ends at 285 (286 is blank) and `g_nu_fn` is on 206 (203 also declares `g_chunks`).
2. `sc_acquire` in belady state picks victims only from the ordered set. An entry stored while belady was off is not in the set, so if the set is empty (or all its keys are in `prot`) and no spare exists, the original exits with the "every slot is in use by one call" message even when no held entry is protected. The header's wording ("every candidate is in `prot`") does not say this. The reference follows the original.
3. `sc_try_acquire` is documented for belady state; the reference does not check the state (as the original): with no spare and an empty set it returns NULL, with a spare it acquires regardless of policy.
4. `sc_touch` in policy 0 still does the heat update (heat is only read by the heat rule), as the contract's "non-belady state" says.
