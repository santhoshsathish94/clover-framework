# copyq verification record (agent C, 2026-10-21, box `gpu`)

Question: does the Rust queue (`target/release/libcopyq.so`) behave like the C++ reference (`ref/libcopyq_ref.so`) on every contract-visible aspect, and is it as fast?
Nothing outside this folder (local) and `~/rustmig/copyq/verify` (box) was written. md5 of `copyq.h`, `ref/copyq_ref.cu`, `ref/libcopyq_ref.so`, `target/release/libcopyq.so`, `src/*.rs` taken before and after: unchanged.
Library md5 under test: ref `59ef712d...`, rust `e8d77fab...` (full list in `box-out/md5-before.txt`).

## Files
- `diff_driver.cpp`: one library per process, one scenario per run (S1-S5, T1-T3, mkfiles, selftest). stdout is canonical (no times, no addresses), exit 0 PASS / 2 FAIL / 3 driver error.
- `one.sh`, `matrix.sh` (correctness + negative controls), `timing.sh`, `compare.sh`, `tsummary.py`, `make_mutants.py` + `mk_mutants.sh` (mutants are copies in `mutants/`), `rs2.ps1` (local helper: scp + one ssh, output to a file).
- `box-out/`: logs of the final runs (`matrix.log`, `timing.log`, `cpp/`, `rust/`, `M*/`, `run1/` = first run, see below).
- Test files (`/dev/shm/copyq_verify`, `verify/data/rec8.bin`, 8 records = 140 MB each) were deleted afterwards; `diff_driver - mkfiles` recreates them.

## Instrument validation
- SHA-256 in the driver: "abc" and "" match the known vectors; records 0 and 3 (17,547,264 bytes) equal `sha256sum` of the same slices of the file on disk; the two files (shm and disk) have equal `sha256sum`.
- Expected bytes are computed by the driver from (record id, word index); they do not come from either library.
- Negative controls (mutated copies of the reference): all caught, see report.

## What was learned
- First S3 run FAILED on both libraries identically: my expectation "all victim copies finish while the gate is held" was wrong. A card queue has max(2, 16/8) = 2 threads; each takes one job and stops in the gate, so only the first 2 jobs per card have done their victim copy. Model fixed in the driver (and it now also checks the thread count per card queue); the whole matrix was re-run with the fixed driver. First-run logs are in `box-out/run1/`.
- CPU seconds in timing are dominated by CUDA stream-synchronise spinning (about 14 cores busy for ~1 s in T1/T2), the same in both libraries. They measure "cores kept busy while waiting", not copy work.
- Several workers calling `exit(1)` at once (mutant M4, EOF on every job) ended in `timeout: dumped core` / `initialization error` in the reference lib, not a clean status 1. Not examined for the Rust lib (S5 enqueues one job per child).
- Shared pwsh terminal + ssh: two helper calls stalled after scp; one ssh per helper call fixed it.
