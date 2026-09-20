# Context — Kimi K3 benchmark on rented hardware

Handoff record for the `kimi-k3-in-c` measurement work. Written 2026-09-21.
Companion to `kimi-k3-bench-run.sh`, `fareed-khan-kimi-k3-in-c-explanation.md`
and `kimi-k3-local-evidence.json`.

Upstream clone used: `FareedKhan-dev/kimi-k3-in-c` at `ac1584a`, 0 behind origin.

## Intended outcome

Demonstrate to a limited audience that **one CPU box with no GPU is enough for most
of the work**. Kimi K3 is the showpiece for "model size is not machine size"; it is
not the workload. A separate segment has to carry the software-engineering claim.

Secondary, and a by-product rather than the goal: contribute a commodity-hardware
data point upstream (`ROADMAP.md` items 2 and 3).

## What the System showed

### The published speed figures are two campaigns, and they disagree

- `docs/PERFORMANCE.md` and `docs/data/*.tsv` are the **pre-v1.0.0** campaign.
  v1.0.0 (2026-08-07) fused the matmul kernels and took trunk compute to
  **2.48 s/token, from ~20**.
- The current figures are in `docs/data/speed-2026-08.md`: 8 GB → 26.5, 32 GB → 24.2,
  64 GB → 19.8, **128+ GB trunk-resident → 5.59 s/token at 179 GB peak RSS**.
- `docs/TUNING.md`'s preset table still gives ~17 s/token for `server`. Its column is
  headed "expect" and the file is user guidance rather than a measurement record, but
  the v1.0.0 changelog refreshed `k3-doctor.sh` only, so the two now read differently.
  A documentation-synchronisation point worth offering upstream, not a defect.

### s/token is not a static value

The document says so directly: *"The ladder understates sustained throughput. Quote it
as a comparison across budgets, not as the engine's speed."* It moves with generation
length (19.21 at 8 tokens → 10.66–11.79 at 16–32), prompt length, context depth, cache
warm state and device state. Both upstream harnesses run a 5-token prompt at `--gen 8`,
which is the operating point least like real work.

### The noise floor is mostly the device

33% spread on identical configurations. Two runs doing byte-identical work (374.99 GB)
differed 2,709 vs 5,874 MB/s → 29.40 vs 18.37 s/token. *"The variance is not scheduling
jitter or NUMA placement. It is the device."*

### Reference machine vs ours

| | reference | AX102-3-LTD |
|---|---|---|
| CPU | EPYC 7763, 124 vCPU | Ryzen 9 7950X3D, 16C/32T |
| RAM | 228 GiB | 128 GiB |
| NVMe | 3.2 GB/s O_DIRECT, virtualised | 2× 1.92 TB Gen4, bare metal |

`server` (110/13) peaks at ~128 **decimal** GB = 119 GiB, so it fits 128 GiB with ~6 GiB
spare. The 5.59 s/token configuration needs 179 GB and does not fit at all.

## What was ruled out

- **GitHub Actions** — 6 h hard job limit that support cannot raise; 10 GB cache ceiling
  means re-downloading 1.56 TB per job; larger runners need a Team/Enterprise org.
- **Codespaces** — storage an order of magnitude short.
- **Student credits** — Azure for Students is $100 with burstable-only VM quotas; Camber
  is 40 CPU-hours and 50 GB. Not a funding problem, a quota and storage problem.
- **Network-attached storage anywhere** — the benchmark measures device bandwidth.
- **ARM** — engine requires AVX2+FMA.
- **Free academic bare metal (CloudLab `sm220u`, 256 GB + 8× NVMe)** — technically the
  best fit and still open, not pursued because the demo wants a machine we control.

## Decisions

- AX102-3-LTD, Hetzner HEL1: 128 GiB, 2× 1.92 TB NVMe, €39 setup + €0.4471/h excl. VAT.
- **RAID0**, not the installimage default of RAID1: 3.84 TB usable instead of 1.92 TB
  against a 1.67 TB need, and bandwidth where the low rungs are I/O bound. Verify with
  `tools/devbw.py` before the download rather than assuming the striping delivers.
- `--preset auto`, never a forced preset. See the heavy-pin correction below.
- Workload arm is primary; campaign arm is the by-product.

## Corrections — mistakes made and what they cost

Recorded because a repeated mistake is not Growth.

1. **Quoted ladder numbers as the engine's speed.** The document warns against exactly
   this in a sentence I had not read. Root cause: grepping for figures instead of
   reading the source.
2. **Used superseded figures** (16.80 / 24.40 / 29.40) without checking the changelog.
3. **Unit error.** Claimed `server` will not fit 128 GiB. Peak RSS is decimal GB —
   provable because the 8 GB rung logged 8.24 peak under an 8 GiB cgroup cap with
   `MemorySwapMax=0`; had it been GiB it would have been SIGKILLed.
4. **Advised forcing `--preset server` over `auto`. Backwards.** `auto` exists because a
   heavy-pin regression was measured — *51 GB pinned ran slower than 0 GB pinned* — and
   it caps the pin below the RAM ceiling for that reason.
5. **Overstated RAID0.** "80% waiting on disk" is the floor configuration; I/O is 41–61%
   across the ladder and about a third at `server`.
6. **Ranked a user's lived report below the document.** The split-storage guidance, which
   *inverts* the project's headline tuning rule, came from one user with trunk on NVMe and
   checkpoint on SATA. AGENTS.md now states the principle: neither actor possesses
   complete reality.
7. **`split-sweep.sh` takes `[total_gb]` fourth and `[reps]` fifth.** Passing reps fourth
   would have silently swept a 3 GB budget and produced a confident, wrong table.
8. **Assumed an HF token was required.** The repo is public. The real dependency is the
   `hf` CLI with `cache verify`, installed via pipx because PEP 668 blocks pip.
9. **Hetzner billing.** `shutdown -h` is not a cost guard: *"Costs are incurred… until the
   contract is cancelled."* Setup fee is non-refundable once provisioned.
10. **Committed the script `100644`** — Permission denied on a clean clone, the same bug
    upstream fixed in their own repo. Also needed `.gitattributes *.sh text eol=lf`, since
    a CRLF shebang fails on Linux.
11. **A PowerShell-piped byte count reported CRs that were not in the blob.** Verified
    properly with `git ls-files --eol` (`i/lf`). The terminal lies in both directions.

## Limits that bound the demo claim

- **No chat template.** Base-model continuations only: *"asking a question gets the
  question completed rather than answered."* It cannot follow an instruction.
- **No chunked prefill** — ROADMAP item 1. A 21k-token prompt does not complete.
- **No quality benchmark of any kind.** No perplexity, no task evaluation.
- KV cache is ~2.37 MB per position, so context is tight once the trunk is pinned.

## Local results, 2026-09-21 (measured, not inferred)

These establish a method and a set of reference values to carry to the server. They do
not predict the server's speed: this is a 12-thread laptop and the AX102 is not.

Machine: i7-1355U, 10P/12 logical, 15.7 GiB RAM, 214 GB free, MSYS2 MinGW-w64 build.

**`make bench` — first run here, needs no weights.**

| kernel | rate | projection |
|---|---|---|
| bf16 matmul 12288x7168 | 11.33 ms, 15.5 GFLOP/s | trunk 7.30 s/token |
| MXFP4 matmul 3072x3584 | 1.92 ms, 11.5 GFLOP/s | experts 8.48 s/token |

~15.8 s/token of pure arithmetic against the reference box's post-v1.0.0 2.48 s/token on
124 cores. **Carry these two hashes to the server, they must match:**
`bf16 FNV1a = 83c8504a4cb3fac6`, `mxfp4 FNV1a = a231061237b5579d`.

**Checkpoint availability, confirmed without downloading it.** HF repo is public and
ungated (HTTP 200). Live tree: **96 safetensors shards, 1,560,936,091,448 bytes — exact
match** to what `download-model.sh` hard-codes, so its verification will pass. Repo sha is
now `f831ab66814297da540d832a5235f8e904f29d06`; the Jetson result recorded
`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, so the repo moved but the weights did not.

**Gates unlocked by 2.82 MB of tokenizer files** (`tiktoken.model`, `tokenizer_config.json`,
`config.json`, `tokenization_kimi.py`, fetched to `c:\personal\oss\k3tok`):

- `test_cfg real` → **REAL CONFIG: PASS**, all 23 assertions against the released nested
  config: 896 experts / top-16 / 2 shared, 24 MLA + 69 KDA, latent 3584, moe_inter 3072,
  dense_inter 33792, attn_res_block 12, SiTU b1=4 b2=25, layers 92 and 93 both MLA,
  layer 0 KDA. This independently confirms the architecture numbers in the explanation doc.
- `make test TOK_FILES=...` → tokenizer leg **ran** rather than NOT RUN: 163,584 ranks +
  16 added tokens, roundtrip 94,402 bytes -> 28,264 ids -> 94,402 bytes PASS.
- Full weightless suite still green: GATE 1, 1b, 2, 3, "ENGINE MATCHES THE REFERENCE EXACTLY".
- `scale_test` at real dimensions: 69 KDA + 24 MLA = 93, last five 88/89/90 KDA, 91/92 MLA.

**Environment notes.** `k3-doctor.sh` refuses on MSYS2 (Linux-only by design) and runs under
WSL, where `systemd-run --scope --user` works. WSL reports 126 MB/s on `/mnt/c`, which is
the 9p filesystem and not a real measurement of anything.

**shellcheck: clean.** Upstream CI treats it as blocking. Neither apt (needs elevation)
nor MSYS2 (no such package) could supply it; the official static binary from the
shellcheck GitHub release works with no package manager and no elevation, and is kept at
`c:\personal\oss\.tools\shellcheck.exe` (v0.10.0). It found one real issue, SC2015 on the
`pgrep -x k3 && { ...; exit 1; } || true` guard, which reads as if-then-else and is not;
rewritten as an `if`. Now clean, plus `bash -n` passing and `i/lf w/lf`.

**Line endings are a live hazard here.** The working copy silently acquired 323 CRLF
pairs after `.gitattributes` was added, which would have failed on Linux at the first
`for` loop. The committed blob stayed LF throughout. Check `git ls-files --eol` shows
`i/lf w/lf`, not just `i/lf`, before trusting any local validation of this script.

## What remains unknown

- The v1.0.0 streaming figure at a ~128 GB budget. **Nobody has measured it** — the
  published table jumps from 64 GB to trunk-resident at 179 GB.
- Thread scaling. *"`OMP_NUM_THREADS` has never been swept on this engine."*
- Whether RAID0 striping helps in practice on this workload.
- Whether `--spec` and `--save-state`/`--load-state` behave as documented; both are read,
  never executed here.
- Whether the count-extraction patterns in the run script match real output. The
  `s/token` and peak-RSS patterns come from the upstream ladder and are proven; the
  `requests`/`evictions`/`pinned` patterns are derived from README samples and may
  need adjusting on first contact.

## What the next cycle should start from

1. `make test`, `make bench`, `devbw.py` — all free, no checkpoint. Decide go/no-go on
   compute and storage before spending hours on 1.56 TB.
2. `installimage` with `SWRAIDLEVEL 0`, then `loginctl enable-linger $USER` — both
   harnesses hard-exit without a user session bus.
3. Run ARM A first. Stop there if it answers the question.
4. Report counts beside seconds, every repetition, never the best of three.
5. Do not predict a number. Let the box show what it shows.

## Abort criteria, decided now rather than at 2am on a metered machine

- `make bench` FNV1a hashes differ from `83c8504a4cb3fac6` (bf16) or
  `a231061237b5579d` (mxfp4) → **stop**. A bit-exactness failure invalidates every
  measurement that would follow it, and no timing is worth recording until it is fixed.
- `make test` red → **stop**, do not start the download.
- Checkpoint byte total is not 1,560,936,091,448 → **stop**; the weights moved and the
  published figures no longer describe what was fetched.
- A rung exits non-zero for any reason other than OOM → the harness already aborts.
  Do not record it as a data point.
- Billing stops on cancellation in Robot, not on `shutdown -h`. Copy the tarball off,
  then cancel.
