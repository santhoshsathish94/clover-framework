# Context — Kimi K3 benchmark on rented hardware

Handoff record for the `kimi-k3-in-c` measurement work. Written 2026-09-21.
Companion to `kimi-k3-bench-run.sh`, `fareed-khan-kimi-k3-in-c-explanation.md`
and `kimi-k3-local-evidence.json`.

Upstream clone used: `FareedKhan-dev/kimi-k3-in-c` at `ac1584a`, 0 behind origin.

## Intended outcome

Measure what can actually be learned from running a very large model on ordinary, controllable
hardware. The CPU machine established that model size and machine size are separable, but it is
not the destination. Kimi K3 is the systems experiment, not the workload. The next question is
how the same model behaves when storage, CPU memory, and a small GPU are treated as one
heterogeneous system.

Secondary, and a by-product rather than the goal: contribute a commodity-hardware
data point upstream (`ROADMAP.md` items 2 and 3).

## Plain summary, read this first (rewritten 2026-09-23)

A 2.78-trillion-parameter model, 1.56 TB of weights, on one 16-core box with 124 GiB of RAM
and no GPU. Every figure below is measured on that machine; where a number was derived it
says so.

**Start of campaign: 9.40 s per generated token. Now: 5.311 s.** A 64-token answer takes
487.4 s. Output is byte-identical throughout — the same prompt produces the same token ids at
every configuration ever tested, including the untuned default, which is the only reason any
of these comparisons mean anything.

### The findings, in the order they were found

| # | finding | measured effect | kind |
|---|---|---|---|
| 1 | one NVMe negotiated PCIe **x2 not x4**; Hetzner found the cables loose | **18%** (601.8 -> 493.6 s) | hardware fault |
| 2 | **16 threads beats 32** — the box has 16 physical cores | **8.1%** | setting |
| 3 | **prefix reuse** via `--save-state` / `--load-state` | **2.06x** on turn two | feature already present |
| 4 | **the expert cache returns ~0%** on ordinary prompts, so its RAM can go to the trunk | **6.1%**, on 4.1 GB LESS memory | setting, found by tracing |
| 5 | **thread binding** stops migration between the two CPU dies | **4.9%**, spread +-0.05 -> +-0.01 s | setting |
| 6 | **expert reads split into 1 MiB chunks** so the batch can rebalance | **5.7%** | our code change |
| 7 | **batched bf16 GEMM**, bit-exact, wired into q/k/v | **~1%** on spec decode, 0.1% on plain | our code change |

Items 4-7 are now four commits (`e655eb0`, `7c5f6d0`, `7919565`, `9977044`), each building
independently with zero warnings. Measured together across five payloads: **16.8% mean, with
a spread of 0.4 percentage points**, output identical in every pair.

**Two corrections to earlier versions of this summary.** Finding 4 was written as "the expert
cache returns 0.00% at any capacity"; that was measured on a NON-repetitive prompt and does
not generalise — on a repetitive prompt a 10 GB cache retains 63.35% of requests and cuts
expert I/O 2.3x. And the GEMM was quoted at 5.58x; that is the discarded one-accumulator
prototype. The bit-exact kernel that shipped is **2.39x** at T=16.

### What a single token actually costs

Measured on the clean run at commit `9977044`, steady state:

| | bytes moved | rate | time | share |
|---|---|---|---|---|
| trunk streamed from RAM to CPU | 108.81 GB | 47.9 GB/s | **2.27 s** | **43%** |
| experts read from SSD | 25.83 GB | 13.8 GB/s | **1.87 s** | **35%** |
| experts streamed from RAM to CPU | 25.83 GB | 47.9 GB/s | 0.54 s | 10% |
| everything else | — | — | ~0.63 s | 12% |
| **total** | **~160 GB** | | **5.31 s** | |

**To produce one token the machine moves about 160 GB.** It is not computing hard; it is
hauling. The CPU sits at roughly 6% of its arithmetic peak throughout. **The largest single
term is memory bandwidth, not storage** — a fact that decides most of the hardware questions
below.

### How much of the machine is now in use

| resource | using | available | |
|---|---|---|---|
| SSD read | 13.8 GB/s | 14.32 GB/s | **96%** (was 78%) |
| memory bandwidth | 47.9 GB/s | 47.9 GB/s | **100%, at the wall** |
| CPU arithmetic | ~64 GFLOP/s | ~1,075 GFLOP/s | **6%** |
| DIMM speed | 3600 MT/s | 4800 MT/s rated | 75%, capped by 4 dual-rank modules on 2 channels |
| disk capacity | 1.6 TB used | 1.8 TB | **94%, 106 GB free** |

### Ceilings that bound whole directions of work

- **Storage can never be worth more than 35% of a decode step** in single-stream decode,
  because that is all the expert read costs. Realistically a doubling of drives buys ~17%.
- **The expert CACHE can never be worth more than 17.0% of a decode step** on an ordinary
  prompt, at any capacity and under any policy, measured by replaying the real access trace.
  Prefetching is a different question and is not bounded by this.
- **PCIe 5.0 x16 is 64 GB/s against this box's 47.9 GB/s of DRAM.** Streaming weights to an
  accelerator cannot beat keeping them in DRAM; any GPU case rests on the 108.81 GB trunk
  being *resident* in device memory, not streamed to it.
- **Long context is bounded by RAM, not disk.** An 1861-token prompt needs 4.41 GB of KV
  cache and is refused at both t108/c10 and t114/c2.


### What to look for next, in value order

1. **Overlap the SSD read with the DRAM stream.** They are 2.08 s and 2.27 s, they use
   different buses, and today they are strictly sequential — the budget sums exactly to the
   measured step. Perfect overlap would give ~3.2 s/token, **a bigger prize than everything
   found so far combined.** Measure the per-layer timeline before touching code.
2. **Scan-resistant cache policy.** LRU returns 0.00%; admit-on-second-touch + LFU returns
   16.35% on our trace. ~0.46 s/token, cannot change output, costs no memory.
3. **A packed GEMM**, which also unlocks batching and any accelerator argument.
4. **Trunk precision.** Attention alone is 72.40 GB/token of the 108.81. The routed experts
   are already 4-bit while the trunk is 16-bit; `--tf-check` measures the accuracy cost
   directly.

### What was believed and turned out to be false

Kept because each cost real time and would otherwise be re-tried:

| believed | truth |
|---|---|
| "startup is 64% of a run, ~380 s" — *this was priority #1* | the engine's own counter says **19.34 s** for a whole 64-token run; my tracer timed from the wrong origin |
| a 38x kernel slowdown "did not reproduce" | it reproduces exactly **at 32 threads**; four exonerating retests all varied the thread count |
| engine gets 9.29 GB/s, 1.25x unexplained | arithmetic error mixing two time windows; really 11.14 GB/s and the gap is **fully explained** by the read barrier |
| batching flips the bottleneck to compute | no GEMM exists, so batching re-reads the trunk per position; per-token DRAM measured flat at 1.06x |
| MRU would beat LRU on a cyclic scan | 1.08%, worse than random; LFU won at 10.96% |
| 24 threads would beat 16 | 1.1% slower |
| the 96 MB V-Cache die would help | it is the **slowest** placement, -10.6% |
| DRAM traffic would be 139.3 GB/token | measured 114.0, 18% high |
| async I/O would close the storage gap | psync, libaio and io_uring agree within 1% |

**And one method error worth more than any of them:** I built a cost model that matched
measurement to within 8% and treated the agreement as proof of mechanism. It fit because a
term I invented was close in size to a term I had omitted. *Closing to within 8% is not
proof.* One line of source — the kernel takes a single activation vector — refuted it.


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

## Measured on the AX102, 2026-09-22 (the rented machine)

Ryzen 9 7950X3D 16C/32T, 124 GiB (132.0 GB reported available), 2x KIOXIA KCD8XRUG1T92
in md RAID1. Repo pinned to `ac1584a`, same commit as the laptop.

**Gates all passed.** `make test` reference-exact; tokenizer parity now runs with the real
vocabulary (163,584 ranks, roundtrip 148,284 B -> 42,364 ids -> 148,284 B PASS); `make bench`
hashes identical to the laptop. Checkpoint verified byte-exact at 1,560,936,091,448 and
checksum-verified against Hub metadata.

**Generation-length sweep**, workstation preset, 15-token prompt:

| gen | total s | s/token | GB read |
|-----|---------|---------|---------|
| 8   | 189.7   | 23.71   | 478.50  |
| 16  | 280.7   | 17.54   | 900.99  |
| 32  | 463.1   | 14.47   | 1745.98 |
| 64  | 836.9   | 13.08   | 3435.97 |
| 128 | 1590.2  | 12.42   | 6815.93 |

Fixed startup ~97 s, marginal ~11.6 s/token. The 23.71 at gen 8 is 51% startup overhead;
quoting it as the engine's speed would have been the ladder mistake all over again.

**Trunk-pinning sweep** (trunk_gb / cache_gb, budget held under the 5% guard):

| trunk | pinned | GB/token | s/token | peak RSS |
|-------|--------|----------|---------|----------|
| 60    | 47/93  | 52.81    | 11.73   | 95.3     |
| 75    | 60/93  | 38.02    | 11.30   | 105.4    |
| 90    | 72/93  | 24.08    | 9.78    | 114.7    |
| 100   | 81/93  | 13.52    | 8.70    | 119.5    |

**The byte model is exact**: 1.15 GB per un-pinned trunk layer (1.148 / 1.152 / 1.147 /
1.127 across the four budgets). Byte counts reproduce to the decimal across independent
runs; s/token carries ~1% noise. Trust bytes, replicate time.

### Complete I/O accounting, which corrected the conclusion above

The `gb_read` column reports **trunk only**. The engine reports expert traffic separately,
and it moves the OPPOSITE way: pinning trunk steals RAM from the expert cache. Full
picture at gen 64, transcribed from the run logs:

| trunk | cache | trunk GB | expert GB | total GB | wall s | I/O share | retention |
|-------|-------|----------|-----------|----------|--------|-----------|-----------|
| 60    | 30    | 3435.97  | 1284.21   | 4720.18  | 840.5  | 119.5%    | 31.35%    |
| 75    | 25    | 2504.38  | 1827.11   | 4331.49  | 819.0  | 107.6%    |  1.37%    |
| 90    | 20    | 1625.99  | 1827.11   | 3453.10  | 725.0  |  90.2%    |  1.09%    |
| 100   | 14    |  960.53  | 1827.11   | 2787.64  | 652.0  |  73.6%    |  0.77%    |
| 108   | 10    |  508.00  | 1827.11   | 2335.11  | 601.8  |  60.1%    |  0.55%    |

`expert_GB` is IDENTICAL to the decimal at cache 25, 20, 14 and 10. Only cache 30 differs.
**The expert cache is a cliff, not a gradient**: below it retention is ~0 and it is dead
weight; at 30 GB it retains 31.35% and removes 543 GB over 64 tokens.

Fitting wall against TOTAL bytes: `wall = 366.5 + GB/9.76`, residuals within ±10 s on
600-840 s, i.e. at the 1% noise floor. **Effective bandwidth 9.76 GB/s against a measured
array peak of 10.8 GB/s, so the machine runs at 90% of its storage capability and IS
storage-bound.** The earlier "storage is not the bottleneck" conclusion came from counting
only trunk bytes, which produced a nonsensical 14.8 GB/s slope — above hardware, and that
should itself have been the tell.

### The engine already measures what I built external profilers for

`k3_trunk.c` tracks `k3_trunk_bind_wall`, `k3_trunk_widen_wall` and `k3_trunk_binds`, and
prints them. From our own logs, already captured:

```
t60  gen64: bind wall 18.73 s over 5952 binds; read 502.99 + widen 6.39 = 509.38 s of
            device work, of which 490.66 s (96%) overlapped compute on the reader thread
t100 gen64: bind wall 20.78 s ... 136.25 s (87%) overlapped
```

Trunk binding is **2-3% of wall clock**; the prefetch hides 87-96% of device work behind
compute. Main-thread bind wall stays flat at ~19-21 s while device work falls 509 -> 157 s,
so cutting trunk bytes cut work that was already invisible. Read the engine's own report
before building anything external.

### Prefill is compute-bound; decode is I/O-bound

Prompt-length sweep at trunk 90 / cache 10, gen 32, transcribed from `probe.out`:

| prompt | approx tokens | s/token | GB read |
|--------|---------------|---------|---------|
| 60 B   | 15            | 12.99   | 855.36  |
| 600 B  | 150           | 25.38   | 855.36  |
| 3000 B | 750           | 75.11   | 855.36  |

**Byte-identical I/O, 5.8x the time.** Prefill batches all prompt tokens through the weights
in one pass, so bytes stay constant while compute scales.

**CORRECTED 2026-09-22.** The "2.8 s of pure compute per prompt token" originally recorded
here was derived by subtracting `gen x 12 s/token` from the total, using a decode rate I had
ESTIMATED rather than measured. Decode is actually **7.49 s/token** (see the decomposition
section below), and a 10-second resource trace shows prefill of 140 tokens takes ~100 s,
so the real figure is **~0.71 s per prompt token** - I was out by 3-4x.

Also corrected: "the disk measured at 0 MB/s during prefill" was wrong twice over. The trace
shows prefill sustains **~1,500 MB/s** throughout. The original claim came from one 5-second
sample that happened to land in a quiet moment.

Consequence: a 2000-token context costs roughly **25 minutes** of prefill, not the hour
recorded here earlier. Still not interactive, but materially less bleak.

### Upstream already documented the allocation rule, and contradicts one of our estimates

`docs/TUNING.md`: *"fill the trunk before you feed the expert cache"*, and a gigabyte of
trunk removes ~1.17 GB/token of guaranteed traffic (we measured 1.15 independently). Its
preset table reports `server` (110/13, ~128 GB) at ~17 s/token and `max` (110/109, ~224 GB)
at ~19 s/token, with *"the extra 96 GB buys nothing outside the noise floor."*

**That contradicts the estimate that +30 GB of RAM would buy ~17% here.** Upstream tested
full-trunk-pin plus a large cache directly; our figure is extrapolated from points where we
never had both. Treat the RAM case as CONTESTED, not established. Unresolved difference:
our box shows 31.35% retention at a 30 GB arena where upstream reports the knee at ~36 GB,
possibly because a code prompt has better expert locality than their trace.

**Time does not follow bytes.** Bytes fell 74%, time fell 26%. Fit: `t = 7.83 + GB/12.6`,
so a **~7.8 s/token floor that reducing bytes does not touch**. Compute is 1.95 s/token
(kernels), so ~2.8 s/token is unaccounted for.

**Storage is not the bottleneck.** `wchan` over 30 s: futex_wait_queue 71.0%,
blk_io_schedule 21.1%, on-CPU 7.8% — corroborated independently by thread state
S 70.8% / D 21.5% / R 7.6%. Two mechanisms agreeing within 0.4 points.

Storage facts, for the record: array peaks at 10.8 GB/s with 8-way O_DIRECT but delivers
~5.5 GB/s under the real workload at 82-83% md2 utilisation; request size is 127 KB, which
is `max_hw_sectors_kb` and therefore a hardware ceiling, not a tunable; `read_bytes/rchar`
= 1.000 exactly, confirming O_DIRECT with no page-cache assistance.

**Hardware fault found:** `nvme1n1` negotiated PCIe **x2** with `max_link_width=4` — a
degraded link, not a slot limit, halving one mirror member's ceiling to ~3.9 GB/s. It shows
as 70 ms latency at 71% utilisation against nvme0n1's 45 ms at 47% for the same bytes.
Not currently the binding constraint, but it is a real Hetzner-reportable fault.

RAID1 read balance under real load is **52/48**. An earlier 66/34 reading was an artefact
of an 8-stream synthetic test and should not be used.

### Frontier bottleneck evolution, and which of it applies here

Read from primary sources 2026-09-22 (DeepSeek V3 arXiv:2412.19437, V3.2 arXiv:2512.02556,
V4.1-Flash release note, Kimi K3 blog). The pattern is that each generation finds the new
binding constraint and changes the ARCHITECTURE, not the hardware.

| gen | what became expensive | what changed |
|-----|----------------------|--------------|
| DS V2 | KV memory, dense compute | MLA latent KV compression, DeepSeekMoE |
| DS V3 | load-balancing overhead | auxiliary-loss-free balancing, MTP, FP8 |
| DS V3.2 | long-context attention compute | DeepSeek Sparse Attention |
| DS V4.1-Flash | KV cache size, input/output asymmetry | Causal Encoder-Decoder: **8B active for input, 16B for output**; KV cache cut to 1/4 HBM and 1/8 SSD |
| Kimi K2->K3 | scaling efficiency, routing at 16/896 sparsity | KDA + AttnRes, Stable LatentMoE, Quantile Balancing, MXFP4/MXFP8 QAT, "no host synchronization on the critical path" |

**Two of these land directly on what we measured.**

1. **DeepSeek split prefill and decode into different architectures** (8B input / 16B output).
   We reached the same structural conclusion by measurement on one CPU: decode is
   storage-bound at 47.9% I/O, prefill is compute-bound at 72.4% matmul. Independent
   convergence, so the asymmetry is a property of the workload, not of our machine.

2. **Kimi's pricing says avoid prefill rather than accelerate it**: $0.30/MTok cache-hit
   input vs $3.00 cache-miss, a 10x gap, with "cache hit rate above 90% in coding
   workloads" via Mooncake disaggregated inference. They also note KDA broke conventional
   prefix caching and they contributed a vLLM fix, because "KDA with prefill cache allows
   us to serve Kimi K3 at a highly competitive token price."

Which frontier bottlenecks exist on this box: KV cache size (yes, 2.37 MB/position),
prefill/decode asymmetry (yes, measured), host sync on the critical path (yes, ~26%
libgomp and 71% futex), flat expert routing defeating cache (yes, Quantile Balancing is
deliberate). Which do not: inter-accelerator communication and expert-parallel balancing,
because we run one node where Kimi recommends 64+ accelerators.

**This re-ordered the queue.** The thread sweep was next; prefix caching should be, because
the engine already implements it (`--save-state` / `--load-state`, upstream claims 3.9x on
turn two) and it attacks the phase we measured as most expensive using the lever the
frontier monetises at 10:1.

### Prefill profile, measured with perf on a `--gen 0` run

```
54.87%  k3_matmul_bf16      dense trunk matmul
17.56%  k3_matmul_mxfp4     expert matmul
~26%    libgomp             OpenMP barrier / spin
 0.51%  k3_router           expert routing
```

**72.4% of prefill is two matmul functions.** Routing is negligible.

**CAVEAT added 2026-09-22.** This profile was taken during a `--gen 0` run, which is LOAD
plus prefill, not prefill alone. The resource trace later showed load is a distinct phase
running at ~90% CPU user for ~147 s, while prefill runs at 56-65% user. So this profile is
contaminated by the load phase and the 72.4% should not be read as pure prefill.

The Amdahl ceiling of 1/(1-0.724) = 3.6x originally quoted here therefore rests on a
contaminated share AND on the wrong prefill/decode split. Both biased the argument AGAINST
a GPU. The corrected split puts prefill at 67.4% of a gen-32 run rather than 45%.

Retracted: an earlier claim that prefill does zero disk I/O. That came from ONE 5-second
sample; 300 seconds of samples show 1,500-2,500 MB/s. The byte evidence stands (60/600/3000
byte prompts all read exactly 855.36 GB) but "no disk during prefill" was wrong.

### Prefix caching works, and the saving is not where I expected

`--save-state` / `--load-state`, measured 2026-09-22 at trunk 90 / cache 10, post PCIe repair.
150-token shared prefix, 15-token follow-up, 32 generated.

| arm | total s | s/token | trunk GB | expert GB | state MB |
|-----|---------|---------|----------|-----------|----------|
| control, prefix reprocessed | 701.5 | 21.92 | 855.36 | 1529.33 | - |
| turn 1, `--gen 0 --save-state` | 471 (wall) | - | 108.81 | 728.62 | 913 |
| turn 2, `--load-state` | **340.4** | **10.64** | 855.36 | **1033.59** | 913 |

**2.06x on turn two.** The engine confirms the hit explicitly: `resuming from turn1.state:
140 prior positions, 20 new`.

**The saving is entirely in EXPERT bytes, not trunk.** Trunk reads are IDENTICAL at 855.36 GB;
expert traffic falls 1529.33 -> 1033.59 GB, a 32% cut of 495.74 GB. Resuming does not avoid
streaming the trunk, it avoids re-routing 140 positions through the experts. I had modelled
this as a saving in prefill COMPUTE; on a storage-bound machine the real prize was bytes not
read, which is worth more here.

**What a cached prefix costs to store**: 914 MB for 140 positions, of which ~626 MB is the
fixed recurrent state across 93 layers. **69% of the file is independent of prefix length**,
so caching is storage-efficient only for long prefixes. 750 tokens would be ~2.4 GB, 10,000
tokens ~24 GB. This is the quantity DeepSeek cut to 1/8 of SSD in V4.1-Flash.

**Prediction scored**: I predicted 420 s and 1.67x. Actual 340.4 s and 2.06x, so I was 19%
pessimistic.

**RESOLVED 2026-09-22.** I flagged that 2.06x exceeded a ~1.8x ceiling and that one of my
inputs must be wrong. It was the decode rate: I assumed 12 s/token, measured 7.49. With the
measured value, prefill is **67.4%** of the control rather than 45%, so the ceiling is
**3.07x** and 2.06x sits comfortably under it at 67% of theoretical maximum. No anomaly
remains; the anomaly was my arithmetic.

**Untested**: upstream's 3.9x claim. With prefill at 67.4% here the ceiling is 3.07x, so
3.9x needs a longer prefix than this geometry provides. The 750-token prompt would raise it.

Script flaw worth remembering: `grab()` reported turn 1 as FAILED because `--gen 0` generates
no tokens and therefore never prints the `N tokens in X s` line the extractor looks for. The
run succeeded; the extractor did not.

### The decomposition, finally measured rather than estimated

Three points at trunk 90 / cache 10, same 150-token prompt, only `--gen` varied, so load and
prefill cancel in the differences. No samplers running; a 10 s `/proc` tracer alongside
measured its own CPU as `00:00:00` over 65 s wall.

| gen | total s | s/token | trunk GB | expert GB |
|-----|---------|---------|----------|-----------|
| 16  | 589.9   | 36.87   | 470.04   | 1116.06   |
| 32  | 716.1   | 22.38   | 855.36   | 1529.33   |
| 64  | 950.6   | 14.85   | 1625.99  | 2355.88   |

`total = F + M x gen` gives **M = 7.49 s/token decode** and **F = 472.6 s fixed**. Against
the independent 701.5 s prefix-test control the fit predicts 712.3 s, a -10.8 s residual
(1.5%, about the noise floor), so the linear model holds across the range.

**I had been assuming M = 12. It is 7.49, so the estimate was 60% too high**, and it
propagated into the prompt-length figure, the Amdahl ceiling, and the prefix-cache ceiling.
All three are corrected above.

### Three phases, visible only in the resource trace

A 10-second trace of one gen-16 run shows sharp boundaries a whole-run total cannot:

```
el   10-141   us 89-92  wa 3-4    read ~1500  RSS 66.7 -> 96.5   LOAD
el  151-242   us 56-65  wa 3-4    read ~1500  RSS flat 97.8      PREFILL
el  252+      us 37-44  wa 20-31  read ~5900  RSS flat           DECODE
```

**Load is ~147 s of the 472.6 s fixed cost**, roughly a third, and it is pure process
startup paid once. A persistent server would remove it entirely. It is not prefill, and I
had been counting it as such.

**Prefill is not CPU-saturated** - 56-65% user with ~32% idle. Only LOAD saturates at ~90%.
And prefill sustains ~1,500 MB/s of disk, so it is not the compute-only phase I described.

The decode transition is unmistakable: read jumps 1,300 -> 5,900 MB/s and I/O wait goes
4% -> 25% within one 10 s sample. That is the storage-bound regime arriving.

### Thread count: 16 beats 32, and decode does not care

Four runs, gen 16, trunk 90 / cache 10, only `OMP_NUM_THREADS` varied.

| threads | total s | s/token | vs 32 |
|---------|---------|---------|-------|
| 32 | 592.0 | 37.00 | - |
| **16** | **543.8** | **33.99** | **-8.1%** |
| 8 | 668.8 | 41.80 | +13.0% |
| 4 | 796.2 | 49.76 | +34.5% |

**Non-monotonic, so none of my three hypotheses were right.** 32->16 improving means SMT
threads are pure overhead; 16->8 degrading means the parallel work is genuine, so it is not
lock contention. The 26% `libgomp` is **idle workers at barriers, and above 16 the extra
workers are actively harmful**. The optimum is exactly the physical core count.

**Output invariance PASSED**: `ids_sha 130f419e7a35` identical at every thread count, and
trunk/expert bytes identical to the decimal (470.04 / 1116.06 GB). The engine is
deterministic across an 8x thread range. This is the correctness gate the plan was missing.

Per-phase, sliced from the 10 s trace by the recorded start/end of each run:

| phase | 32t | 16t | 8t |
|-------|-----|-----|-----|
| load | 380 | 340 | 430 |
| prefill | 100 | 90 | 130 |
| **decode** | **110** | **120** | **110** |

**Decode is thread-insensitive** - flat across a 4x thread range, exactly as a storage-bound
phase should be. The whole-run win comes entirely from load and prefill. This kills the
disaggregation idea I had proposed (different thread counts per phase): there is nothing to
split, because decode has no preference. DeepSeek's 8B-in/16B-out asymmetry is about
parameter activation, not thread scheduling, and the analogy does not carry down to us.

**CORRECTION to the phase numbers recorded earlier.** I previously wrote load ~= 147 s. The
tracer started at 02:44:15 and that run started at 02:40:11, so I read "elapsed 141" as 141 s
into the RUN when it was 141 s into the TRACE - 385 s into the run. Load is ~380 s, not 147.
It reconciles with the independent fit: F = 472.6 s against load + prefill = 480 s.

**Load is therefore ~64% of a gen-16 run** and is pure process startup. That is the largest
remaining target and it needs no hardware.

Caveat on the 4-thread row: decode shows 10 s, which is a detection artefact - the classifier
requires read > 3000 MB/s and at 4 threads decode never sustained that, so it was labelled
prefill. The 4-thread split should not be trusted; 32/16/8 are internally consistent.

## What remains unknown

- **What the ~2.8 s/token floor actually is.** Not storage, not kernel compute. Attention,
  routing, KV update and expert gather all live in there, unmeasured.
- **Whether the 71% futex is idle workers or lock contention.** Those need opposite fixes.
  GNU OpenMP parks idle workers on a futex at barriers, so the observation is equally
  consistent with both. The queued thread sweep discriminates: flat s/token across
  4/8/16/32 means idle workers; degradation with more threads means contention.
- Thread scaling. *"`OMP_NUM_THREADS` has never been swept on this engine."* Still true.
- Whether `--spec` and `--save-state`/`--load-state` behave as documented; both queued,
  neither executed yet.
- Prompt-length cost is not separated from generation-length cost. A 1-token prompt gave
  15.69 s/token at gen 8 where a 15-token prompt gave 23.71; I let two variables move at
  once and cannot attribute the difference.

Resolved since the last cycle: the ~128 GB budget figure is now measured; RAID0 is moot
because the workload is not storage-bound; tokenizer parity runs and passes.

## What the next cycle should start from

1. **Do not sweep pinning further.** 122.8 GB RSS against a 125.4 GB guard is the wall, and
   returns are flattening (13.5% -> 11.0% -> ~4%). More of this optimises the thing Reality
   already said is not the constraint.
2. Run the thread sweep and read it as a discriminator, not a tuning exercise.
3. Separate prompt length from generation length. I conflated them once already.
4. Report counts beside seconds, every repetition, never the best of three.
5. Do not predict a number and then go looking for it. Record the prediction first so it
   can be scored either way — that is what caught the bytes-versus-time error.

## Mistakes made on the machine, 2026-09-22

- **Predicted time would follow bytes.** It did not: bytes -74%, time -26%. Hours of
  storage instrumentation were aimed at a bottleneck that was never there. The column that
  exposed it, `gb_read`, was added almost as an afterthought.
- **A 38x kernel regression that did not reproduce.** 3.7 GFLOP/s on the first run after a
  build; four later runs gave 137-142. Nearly written up as an upstream finding.
- **Instrumentation not verified before use.** A thread sampler claimed 20 Hz and ran at
  4.6 Hz while forking ~17,000 processes into the machine it was "not perturbing"; a
  `/proc` path split took field 6 (`stat`) instead of field 5 (the TID), producing 4,096
  threads and negative CPU time.
- **`--preset auto` refuses on this box** — it asks 126.43 GB against a 0.95 x 132.01 =
  125.41 GB threshold. The refusal message prints `need - have` rather than
  `need - 0.95*have`, so a legitimate refusal reports a *negative* shortfall and reads like
  a bug. Our campaign script used `auto` at all eight measurement sites and would have
  hard-aborted after the 1.56 TB download.
- **Our own campaign preflight is wrong**: it compares 1,680 GB needed against *free* space
  without subtracting what is already on disk, so it aborts instantly once the checkpoint
  and trunk exist. Still unfixed.

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

---

# Activity decomposition, 2026-09-22

Method change, at the operator's direction: stop asking "what outcome will this produce"
and instead split the run into every distinct activity, then examine each one on its own
for what could be improved. The reason given, and it turned out to be the right one:
*nothing in the system is inherently insignificant, and building from the bottom makes
things easier to understand and grow.* Everything below came from doing that, and most of
it contradicts what the campaign had concluded from the top down.

## The baseline being decomposed

`--trunk-gb 108 --cache-gb 10`, `OMP_NUM_THREADS=16`, `--incremental`, prompt 17 ids,
post-PCIe-repair. Step 0 costs 77.86 s and does three unrelated things at once. Steps 1+
cost **6.26 s** and do one thing, with a spread of 6.20–6.37 across 63 steps. Only the
second is worth attributing, and no profile before this one had ever looked at it.

## What a steady decode step actually spends itself on

`perf record -F 99`, flat, attached to all threads, 40,985 samples, window placed after
step 2 had printed so that neither prefill nor the first-touch trunk load is inside it.

**Non-interference check, which is why the window placement is recorded:** steps inside
the perf window averaged 6.30 s, steps outside it 6.26 s. 0.6% apart. The measurement did
not move the thing being measured.

| symbol | self CPU |
|---|---|
| `k3_matmul_bf16` | **58.04%** |
| `k3_matmul_mxfp4` (routed experts) | 15.69% |
| libgomp (barrier / spin) | 22.19% |
| `k3_router` | 1.21% |
| kernel | 2.31% |

Kernel at 2.31% is not a contradiction of an I/O-bound engine. O_DIRECT reads block in
uninterruptible sleep and perf cannot sample a sleeping thread, so ~2.5 s/step of expert
read is invisible here **by construction**. This is a distribution of CPU cycles, not of
wall clock, and quoting it as wall clock would be the same error as the retracted prefill
profile.

## The kernels, measured rather than reasoned about

`bench_kernels` at real K3 dimensions, two reps per point:

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| bf16 GFLOP/s | 12.3 | 23.0 / 24.5 | 36.1 / 43.0 | 49.9 / 54.6 | 64.3 / 60.3 | 67.4 / 60.7 | 67.7 / 72.6 |
| mxfp4 GFLOP/s | 14.7 | 29.1 | 55.9 / 53.3 | 69.4 / 78.5 | 46.0 / 104.9 | **109.1 / 106.4** | **3.8 / 3.7** |

1. **bf16 saturates.** 32x the threads buys 5.6x the throughput, 18% of linear. It is not
   waiting on FMA units. Rewriting its double-precision accumulator — which exists on
   purpose, to keep the AVX2, NEON and scalar paths bitwise identical under `test_ops` —
   would therefore buy close to nothing. That accumulator looked like the obvious target
   and is not one.
2. **mxfp4 peaks at 24 threads, not 16.** The engine runs at 16. Untested on the engine.
3. **CORRECTION, and it reverses a recorded finding.** This file says above: *"A 38x kernel
   regression that did not reproduce. 3.7 GFLOP/s on the first run after a build; four
   later runs gave 137-142."* That was wrong. It reproduces exactly — 3.8 and 3.7 — **when
   thread count is held at 32**, which the four exonerating runs never did. It is a real,
   repeatable 28x cliff at full SMT, not a cold-start artefact. I dismissed a true finding
   by varying the wrong thing while re-testing.

## The ceiling those kernels are running into

Streaming-read benchmark, 16 GB buffer, well past the 128 MB L3:

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| GB/s | 41.1 | 47.2 | **47.9** | 47.5 | 46.1 | 45.8 | 45.4 |

Memory bandwidth **peaks at 2–4 threads and declines thereafter**, and one core alone
reaches 86% of the peak.

`dmidecode`: 4 x 32 GB DDR5, **rated 4800 MT/s, Configured Memory Speed 3600 MT/s**. Four
dual-rank modules on a two-channel AM5 board cannot train higher. So:

- theoretical at 3600 MT/s dual channel = 57.6 GB/s; measured 47.9 = **83% efficient, i.e.
  the kernel is at the wall**
- theoretical at the rated 4800 MT/s = 76.8 GB/s, **1.33x more**

**The 128 GB of capacity is bought by filling all four slots, and that purchase costs a
quarter of the memory bandwidth.** Capacity and bandwidth are in direct tension on this
board. Every earlier note in this file counted the capacity and none counted the bandwidth.

**This also invalidates the bench as a predictor of engine speed.** The bf16 bench reports
64–72 GB/s of weight traffic, which is *above* DRAM speed; it can only do that because its
176 MB working set partly fits in the 7950X3D's 128 MB L3 and is re-read. The engine gets
no such reuse — every trunk weight is touched exactly once per token. So the bench's
"1.76 s/token" projection is optimistic and the engine's real figure is DRAM-limited.

## A ground-up cost model that closes

Built only from measured device rates, for one steady decode token:

| activity | bytes moved | at | time |
|---|---|---|---|
| routed experts, NVMe → RAM | 25.83 GB | 9.2 GB/s | ~2.50 s |
| trunk, DRAM → FMA units (56.74 G params x 2 B) | 113.5 GB | 47.9 GB/s | ~2.37 s |
| routed experts, DRAM → FMA units | 25.83 GB | 47.9 GB/s | ~0.54 s |
| attention, router, lm_head, libgomp barriers | — | — | ~0.85 s |
| **predicted** | | | **~6.26 s** |
| **measured** | | | **6.26 s** |

The model closes to within its own noise. The two dominant terms are **comparable in
size**, and one of them is not storage:

> **Streaming the pinned trunk out of DRAM costs almost as much per token as reading the
> experts off NVMe. Pinning the trunk in RAM did not make it free — it only moved it from
> one bus to a faster one.** The whole campaign treated this as a storage problem.

## CORRECTION: the "380 s load, 64% of the run" figure

Claimed in the original plain summary (since rewritten) and in the phase-split table under
"Measured on the AX102", and used to justify "eliminate the load" as the top priority. It
cannot be right in this configuration. The engine's own counter reports **bind wall 19.34 s
over all 5,952 binds of a 64-token run**, and step 0 in total is 77.86 s. A 380 s load is
not compatible with either number.

The figure came from slicing my own tracer, which reads elapsed time from tracer start
rather than run start — an error already recorded once in this file and evidently not
fully corrected. **Treat 380 s as withdrawn.** Where my tracer and the engine's internal
counters disagree, the engine's counters win: they bracket named code, mine bracket a guess
about which wall-clock interval belonged to which phase.

Confirmed end to end on 2026-09-22: a 64-token run has a prefill of **73–78 s** total, of
which the trunk's first-touch load is a part, against 407 s of wall clock. Startup is
roughly **18%** of a short run, not 64%, and it amortises to nothing on a long one.

## Per-activity verdict

| activity | cost/token | improvable? | by what |
|---|---|---|---|
| trunk DRAM streaming | ~2.37 s | **only by moving fewer bytes** | denser trunk format; `tools/int8_trunk.py` and `qdq_trunk.py` already exist but are draft-only, and upstream keeps these tensors high-precision deliberately (K3 report 4.1.4) |
| expert NVMe read | ~2.50 s | **yes, policy** | 0.55% retention; LRU against a cyclic scan. The trunk already solved this with a pinned prefix; the expert cache did not |
| expert DRAM streaming | ~0.54 s | no | already mxfp4, 4 bits |
| `k3_matmul_bf16` vectorisation | — | **no** | bandwidth-bound, 18% of linear scaling |
| thread count | — | **untested** | mxfp4 peaks at 24, engine runs at 16 |
| SMT / 32 threads | — | **avoid** | reproducible 28x mxfp4 cliff |
| memory configuration | ~0.6 s | not by us | 4 dual-rank DIMMs force 3600 MT/s; 2 modules would allow 4800 |
| `k3_router` | ~0.08 s | no | 1.21% |
| trunk bind path | 0.30 s/step | no | engine counter; 69% of device work already overlapped |

## What this changes about what to do next

The ranked list earlier in this file is wrong at the top. "Eliminate the 380 s load" was
first; the load is not 380 s and startup is amortised by any persistent process anyway.
The two real terms are trunk DRAM traffic and expert NVMe traffic, they are within 5% of
each other, and the cheapest untested lever is a thread count.

## Thread count 16 / 20 / 24 / 28 on the engine — prediction recorded, prediction wrong

Prediction written before the run: *mxfp4 benches ~2x better at 24 threads, but expert
matmul is only ~0.54 s of a 6.26 s step and DRAM bandwidth is marginally worse at 24, so
expect a small gain of 1–3%, and do not be surprised by neutral.*

| OMP_NUM_THREADS | step 0 | steady mean | range | ids_sha |
|---|---|---|---|---|
| **16** | 77.86 | **6.282** | 6.23–6.33 | b7b340f40393 |
| 20 | 81.01 | 6.411 | 6.37–6.44 | b7b340f40393 |
| 24 | 80.04 | 6.351 | 6.33–6.40 | b7b340f40393 |
| 28 | 79.44 | 6.289 | 6.27–6.31 | b7b340f40393 |

Identical output at every thread count, and identical expert bytes (535.63 GB), so this is
a clean speed comparison with nothing traded away. **16 remains best; 24 is 1.1% slower,
not 1–3% faster.** The direction of the prediction was wrong.

**Why the bench misled, and this is the transferable lesson:** `bench_kernels` measures
mxfp4 against a working set it re-reads, so part of it lives in the 96 MB V-Cache. The
engine's expert bytes arrive cold from an O_DIRECT read and are touched exactly once.
A kernel benchmark's optimal thread count does not transfer to an engine whose working
set is cold. The bench was measuring cache behaviour; the engine has none to measure.

The negative result is itself confirmation of the cost model: if any significant part of
the step were FMA-bound, 24 threads would have helped. Nothing did. **Every major term in
this system is bandwidth — NVMe for experts, DRAM for the trunk — and neither responds to
threads or to vectorisation.**

## Remaining levers, given that everything is bandwidth

Only three things can move a bandwidth-bound system, and none of them is tuning:

1. **Move fewer expert bytes.** Retention is 0.55% against a cyclic access pattern that
   LRU cannot serve. The trunk hit exactly this problem and solved it with a pinned
   prefix, saying so in its own startup banner: *"a cyclic scan defeats LRU, so a pinned
   prefix is used instead"*. The expert cache still runs LRU. `tools/sim_cache.py` can
   replay a trace offline under LRU / Belady / pinned, and `--dump-cache-trace` produces
   one, so this is answerable without spending a single expensive run.
2. **Move fewer trunk bytes.** 113.5 GB per token at bf16. Denser storage is the only
   lever, the tooling exists (`int8_trunk.py`, `qdq_trunk.py`) but is draft-only, and
   upstream keeps these tensors in higher precision deliberately. Accuracy cost is real
   and would have to be measured, not assumed — `--tf-check` measures exactly that.
3. **Amortise both across several tokens.** `--spec` verifies a batch of drafted tokens
   in ONE sweep, so the 2.37 s trunk stream is paid once for k tokens instead of k times.
   The engine claims an extra verified position costs ~22% of a serial token. **We have
   never run it.** It is the only lever that attacks both dominant terms at once.

---

# The expert cache is not short of memory. It is running the wrong algorithm.

Trace captured with `--dump-cache-trace` from a real `--incremental` run at t108/c10,
gen 16, 16 threads: **33,469 requests, 17,169 distinct experts**, rescued to
`k3-results/trace/`. This matters because upstream's published trace came from a run that
re-prefills the whole prefix every step and therefore overstates reuse; ours is genuine
incremental decode, so it is the more conservative evidence.

## The pattern is a cyclic scan, and LRU scores exactly zero

| measure | value |
|---|---|
| distinct experts in ONE token | **1,472** (92 MoE layers x top-16) |
| cache capacity | 569 slots (9.99 GB) |
| working set / capacity | **2.59x** |
| accesses with reuse distance **< 569** | **0 of 33,469** |
| median reuse distance | 2,348 |

Zero. Not few. **LRU's hit rate at this capacity is 0.00% as a matter of geometry**, and no
amount of tuning changes it. Every expert is evicted precisely before it is next needed.

The engine already knows this shape. `k3_trunk.c` prints it at startup: *"a cyclic scan
defeats LRU, so a pinned prefix is used instead."* The trunk was fixed. The expert cache
was not.

## Why: two-thirds of the experts are touched once

| touches | experts | share of accesses |
|---|---|---|
| **1** | **11,396 (66%)** | 34.05% |
| 2 | 2,681 | 16.02% |
| 3-7 | 2,499 | 30.34% |
| 8-15 | 541 | 17.10% |
| **16 (every token)** | **52** | 2.49% |

**~593 experts are touched 8+ times and occupy 10.4 GB — the cache is already 9.99 GB.**
The hot working set very nearly fits in the memory we have. LRU cannot find it because
11,396 single-touch experts stream through and destroy the recency signal. This is
textbook cache pollution by a scan, and the textbook fix is not more memory and not a
cleverer eviction order — it is **refusing to admit on first touch**.

## Policies replayed on our own trace, all at the SAME 569 slots

| policy | hits | rate | GB read |
|---|---|---|---|
| **LRU — what ships** | **0** | **0.00%** | 587.7 |
| MRU (evict newest) | 362 | 1.08% | 581.4 |
| Random replacement | 611 | 1.83% | 577.0 |
| Admit-on-2nd-touch + LRU | 2,326 | 6.95% | 546.9 |
| Protect-recurring | 2,372 | 7.09% | 546.1 |
| LFU | 3,668 | 10.96% | 523.3 |
| **Admit-on-2nd-touch + LFU** | **3,953** | **11.81%** | **518.3** |
| Belady (offline ceiling) | 7,738 | 23.12% | 451.8 |

**HYPOTHESIS RECORDED AND FALSIFIED.** I predicted MRU would win, reasoning that on a
cyclic scan the oldest resident is the one needed soonest, so LRU evicts exactly what
Belady keeps. MRU scored 1.08% — worse than random. The reasoning assumed a clean cycle;
the real pattern has only 36.8% token-to-token overlap and a mean of 1.95 touches per
expert, so freezing a set captures almost nothing. Tested, wrong, recorded. LFU — which I
had not considered — wins, because the distribution is skewed and frequency finds the tail
that recency cannot.

## The advantage grows with generation length

Decode passes only, prefill excluded:

| decode tokens | LRU | Admit-2nd + LFU | GB saved |
|---|---|---|---|
| 2 | 0.00% | 0.00% | 0.0 |
| 4 | 0.00% | 5.79% | 6.0 |
| 8 | 0.00% | 13.38% | 27.7 |
| 12 | 0.00% | 15.34% | 47.6 |
| 15 | 0.00% | **16.35%** | 63.4 |

**LRU is 0.00% at every length tested.** The alternative is still climbing at 15 tokens, so
the gen-64 benchmark figure would be higher than 16.35%, not lower.

## Expected effect, stated as a prediction to be scored later

At ~16% fewer expert bytes: 25.83 -> ~21.7 GB/token, 2.81 s -> 2.36 s at the measured
9.2 GB/s, plus a smaller DMA write into DRAM. Step 6.28 s -> **~5.8 s, about 7-8%**.

Three properties make this unusually safe, and they are the reason it is worth doing
before anything involving quantisation:

- **It cannot change the output.** The cache decides only whether bytes come from RAM or
  disk, never which experts the router chose. Bit-exactness is structural, so none of the
  reference hashes or oracle gates are at risk.
- **It costs no memory.** The ghost list holds keys, not bytes — a few hundred KB against
  a 10 GB arena.
- **It is local.** `pick_victim` and `admit` in `k3_cache.c`, roughly 30 lines.

## Also found while reading the retrieval path

Small, and recorded because size is not the test of whether something is worth knowing:

- `k3_expert_ref` rebuilds **six ~90-character tensor names with `snprintf` and does six
  hash lookups on every single fetch** — 8,832 of each per decode step — for a layout that
  is immutable after `k3_st_open`. All 83,328 (layer, expert) refs could be resolved once
  into a flat array. ~0.06% of the step, but it sits in `cache_getmany`'s **serial** phase
  1, delaying when the NVMe queue is filled.
- `pick_victim` linear-scans all 569 slots and, once the cache is full, never hits its
  early exit: 837,568 iterations per step, also in the serial phase.
- The router gate is held **fp32** (`const float *W`), 25.7 MB per layer, 2.36 GB/token of
  DRAM. bf16 would halve it.
- **Attention, not the MoE, is the largest DRAM consumer**: ~75 GB/token of the 113.5 GB
  trunk traffic is KDA/MLA projections, against ~36 GB for all MoE trunk parts.

## Prefetchability, the other half of the question

Expert reads are 2.81 s/token of NVMe that currently overlaps **nothing** — the layer
blocks on `getmany` and then computes. The trunk has an async reader thread and overlaps
69% of its device time; the expert path has no equivalent.

- Consecutive tokens' top-16 overlap **36.8%** per layer, so a speculative prefetch issued
  a token early would be right about a third of the time.
- Within a layer, the shared-expert block (`sh1`, `sh3`, `sh2`, ~264 MB of DRAM work) and
  the down-projection depend only on the layer input `xt`, **not on the routed experts**.
  They are computed *after* the blocking read today and could be issued during it. That is
  ~6.5 ms of the ~27 ms read window per layer, about **9.5%**, from reordering
  independent work in `k3_moe` — no new memory and no accuracy question.

---

# System measurement data, 2026-09-22

Every row below is measured on this machine, not derived. Raw artefacts, scripts and a
manifest with provenance for each number are at `k3-results/evidence-20260922/` (43 files),
so none of this depends on the server still existing.

## Measurement instruments, calibrated before use

`ls_any_fills_from_sys.dram_io_all` is the only usable DRAM counter (consumer Ryzen does
not expose `amd_df`). Calibrated against `membw` at three sizes before being trusted:

| buffer | bytes actually read | count x 128 | error |
|---|---|---|---|
| 4 GiB | 94.5 GB | 94.5 GB | 0.0% |
| 8 GiB | 189.0 GB | 182.3 GB | 3.5% |
| 12 GiB | 283.5 GB | 281.8 GB | 0.6% |

**128 bytes per count, not 64.** A counter used without calibration would have reported
half the traffic and inverted the conclusion.

## DRAM traffic, measured by differencing two runs

gen 16 minus gen 8 cancels startup, the first-touch trunk load and the whole prefill,
leaving exactly 8 steady tokens.

| quantity | per token |
|---|---|
| wall | 6.21 s |
| **DRAM reads** | **114.0 GB** (predicted 139.3, **-18.2%**) |
| expert NVMe | 25.8 GB |
| trunk NVMe | 6.3 GB |

**PREDICTION MISSED BY 18%.** The measured figure sits almost exactly on the trunk alone
(113.5 GB), which suggests the expert bytes are partly served from the 96 MB V-Cache after
the DMA writes them, or that my per-component split is wrong. Unresolved; the total is
measured and the split is not.

**Corrected budget, and it reverses what I said earlier today:**

| term | seconds | share |
|---|---|---|
| **NVMe reads** (32.1 GB at 9.2 GB/s) | **3.50** | **56%** |
| DRAM reads (114.0 GB at 47.9 GB/s) | 2.38 | 38% |
| everything else | 0.33 | 5% |
| total | 6.21 | measured 6.21 |

Storage is the larger term. I had reported DRAM as larger two hours earlier.

## A free 6.1%, taken directly from the trace finding

Because LRU measures **0.00% at every capacity below 25.85 GB**, the expert cache's memory
is provably doing nothing at 10 GB. It can be given to the trunk at no cost.

| trunk | cache | pinned | steady s/step | trunk NVMe | peak RSS |
|---|---|---|---|---|---|
| 108 | 10 | 87/93 | 6.275 | 191.18 GB | 122.79 GB |
| 111 | 6 | 90/93 | 6.082 | 147.25 GB | 122.25 GB |
| 112 | 4 | 90/93 | 6.090 | 147.25 GB | 120.25 GB |
| **114** | **2** | **93/93** | **5.895** | **108.81 GB** | **118.69 GB** |

**6.1% faster on 4.1 GB LESS memory.** Identical output hash and identical expert bytes
(535.63 GB) at every setting — direct hardware confirmation that the cache was worthless.
**New best config: `--trunk-gb 114 --cache-gb 2`, OMP_NUM_THREADS=16.**

## Storage: the array is not the problem, the engine is

Both drives negotiate **PCIe 4.0 x4, 16 GT/s** — the cable repair held. 7.15 GB/s each
isolated, 91% of link. fio at the engine's exact pattern (17,547,264-byte O_DIRECT random
reads):

| pattern | GB/s |
|---|---|
| one drive, any engine | ~7.1 |
| md2 psync x4 | 14.07 |
| **md2 psync x16 — the engine's exact pattern** | **13.95** |
| md2 libaio qd16 | 14.11 |
| md2 io_uring qd16 | 14.11 |
| **real shard file through ext4, psync x16** | **14.32** |
| real shard file, psync x4 | 14.66 |
| trunk.bin, 64 MB blocks x16 | 13.81 |
| **what k3 achieves** | **9.29** |

**The engine extracts 65% of what the hardware delivers on the same files with the same
pattern.** Two things this rules out:

- **Async I/O is not the fix.** psync, libaio and io_uring agree within 1%. The blocking
  pread model is fine.
- **The filesystem is not the fix.** Raw device and ext4 file agree within 3%.

Partial mechanism: the engine waits for **all 16** reads per layer while fio never waits.
Latency is tight (mean 19.55 ms, p99 23.20, max 25.45, stddev 1.35), so the barrier costs
only 1.2-1.3x — giving ~11.5 GB/s, not 9.29. **A further ~1.25x is unexplained.** The read
path itself is clean: one `pread` per expert, `K3_PREAD_MAX` is 1 GiB so nothing splits.
Prime remaining suspect is OpenMP parallel-region overhead — libgomp was 22% of CPU in the
decode profile, and `cache_getmany` opens a region per layer, 5,888 per 64-token run.

**Worth 0.98 s/token, 16.6%, for no hardware.** This is the largest single lever found.

## Where the headroom is, measured

Baseline: **5.895 s/steady token** at t114/c2/16thr. Each row is a measured gap between
what something delivers and what we extract from it, or a measured ceiling on a lever.

| component | what it can do | what we use | headroom |
|---|---|---|---|
| **NVMe array** | 14.32 GB/s at our own pattern on our own files | **9.29 GB/s** | **35%, worth 0.98 s/token** |
| **Expert cache** | 23.12% hit at 9.99 GB (Belady); 11.81% online | **0.00%** | **worth 0.46 s/token** |
| DRAM | 47.9 GB/s, 83% of the 57.6 available at 3600 MT/s | 47.9 GB/s | none; kernel is at the wall |
| DIMM speed | rated 4800 MT/s = 76.8 GB/s | 3600 MT/s = 57.6 | 1.33x, blocked by 4 dual-rank modules on 2 channels |
| CPU cores | 32 threads | 16 | **negative** — 16 beat 20/24/28/32 |
| bf16 kernel | 18% of linear thread scaling | at the wall | none; bandwidth-bound, not FMA-bound |
| Expert-cache capacity | **17.0% of a step at ANY capacity** | 7.3% reachable now | hard ceiling on that whole lever |

Two ceilings worth remembering because they bound whole directions:

- **The expert cache can never be worth more than 17.0% of a decode step**, at any capacity,
  under any policy. That caps an entire line of work.
- **PCIe 5.0 x16 is 64 GB/s, barely above this box's 47.9 GB/s of DRAM.** So streaming
  weights to an accelerator over PCIe cannot beat keeping them in DRAM. Any accelerator
  argument has to rest on the 108.81 GB trunk being *resident* in its memory, not streamed
  to it. That is a property of the model size, and it will stay true as the trunk grows.

## What the data says to do next, in order

All three are software, cost nothing, and reduce demand on the same resources:

1. **Close the 9.29 -> 14.32 GB/s storage gap.** Largest single number found, 16.6%.
   Mechanism only partly known, which is exactly why it is next.
2. **Replace LRU with a scan-resistant policy.** 0.00% -> 16.35% measured on our trace,
   ~7.8%, cannot change output, costs no memory.
3. **Overlap the shared-expert block with the expert read.** ~8%, pure reordering of work
   already proven independent by reading `k3_moe`.

## CORRECTION: the storage gap is 1.29x, not 1.54x, and it is fully explained

**The 9.29 GB/s figure above is my arithmetic error and should not be quoted.** I divided
per-token expert bytes by a per-token time derived from a whole-run total that included
prefill — numerator and denominator from different windows, the same class of mistake as
the withdrawn "380 s load".

Measured directly from the engine's own counters at t114/c2, gen 14:
**535.63 GB of expert reads in 48.2 s = 11.11 GB/s.**

| | GB/s |
|---|---|
| hardware at our pattern (fio, ext4, real shard) | 14.32 |
| **engine** | **11.11** |
| ratio | **1.29x** |
| fio max/mean read latency | **1.30x** |

**The per-layer barrier accounts for the whole gap.** `cache_getmany` waits for all 16
reads, so it pays the slowest of 16 where fio never waits at all; 1.30x expected, 1.29x
observed. There is no unexplained residual. The earlier claim of "~1.25x unexplained" is
withdrawn.

This also downgrades the prize: closing it needs pipelining (begin an expert's matmul while
its siblings are still in flight), not a configuration change, and is worth ~0.83 s/token
rather than 0.98.

## OpenMP environment sweep — hypothesis falsified, but a real 4.9% found anyway

Tested because libgomp was 22.19% of decode CPU and `cache_getmany` opens a parallel
region per layer. All at t114/c2, gen 14, identical output hash.

| variant | steady s/token | spread | expert GB/s |
|---|---|---|---|
| baseline | 5.903 | 5.85-5.95 | 11.11 |
| `OMP_WAIT_POLICY=ACTIVE` | 5.863 | 5.79-5.92 | 11.07 |
| `GOMP_SPINCOUNT=infinite` | 5.886 | 5.83-5.95 | 11.07 |
| both | 5.885 | 5.83-5.93 | 11.07 |
| **`OMP_PROC_BIND=spread OMP_PLACES=cores`** | **5.616** | **5.61-5.62** | 11.07 |
| **all three** | **5.611** | **5.60-5.62** | 11.09 |
| 32 threads + ACTIVE | 5.898 | 5.83-5.95 | 11.09 |

**Thread wake-up is NOT the storage gap:** the expert rate sits at 11.07-11.11 GB/s in
every variant, unmoved by wait policy, spin count or thread count. Hypothesis dead.

**But binding threads to cores is worth 4.9% for free**, and the run-to-run spread collapses
from +-0.05 s to +-0.01 s — thread migration was real and was also the main source of noise
in every earlier measurement on this box. The gain is entirely on the compute side, since
the expert read rate did not move.

**Current best: `--trunk-gb 114 --cache-gb 2`, `OMP_NUM_THREADS=16`,
`OMP_PROC_BIND=spread OMP_PLACES=cores` -> 5.611 s/token**, from 6.275 this morning, all
from configuration and all with byte-identical output.

## Topology that makes the next question obvious

| | CPUs | L3 |
|---|---|---|
| **CCD0** | 0-7, 16-23 | **96 MB (3D V-Cache)** |
| CCD1 | 8-15, 24-31 | 32 MB |

The two dies are not interchangeable, and 16 threads on 16 cores straddles both — half the
work runs with a third of the cache. Worth testing, particularly because the bf16 bench
reached 64-72 GB/s where DRAM measures 47.9, which is only possible if something is being
served from V-Cache.

## CCD placement — prediction correct, and the V-Cache turns out to be worthless here

Prediction recorded before running: *CCD0-only will be neutral to slightly worse, because
the token streams 114 GB out of DRAM and no cache holds that, while one CCD's fabric link
may not sustain what two pull together.* Half right — it was worse, but not for the fabric
reason.

Memory bandwidth first, so the engine result is interpretable:

| pinning | GB/s |
|---|---|
| both CCDs, 16 thr | 46.2 |
| CCD0 only, 16 thr | 46.3 |
| CCD1 only, 16 thr | 45.8 |
| CCD0 only, 8 thr | 47.4 |

**Flat.** A single CCD pulls full DRAM bandwidth, so the Infinity Fabric was never the
constraint and that half of the prediction was wrong.

Engine, gen 14, t114/c2, all with identical output:

| variant | cores | steady | spread | expert GB/s |
|---|---|---|---|---|
| **both CCDs, 16 thr** | 16 | **5.611** | 5.60-5.62 | 11.11 |
| CCD1 only, 16 thr | 8 | 5.970 | 5.96-5.98 | 10.91 |
| CCD0 only, 8 thr | 8 | 6.014 | 6.00-6.02 | 10.93 |
| CCD0 only (V-Cache), 16 thr | 8 | 6.204 | 6.15-6.59 | 10.93 |
| both CCDs, 32 thr | 16 | 5.798 | 5.78-5.89 | 10.82 |

Two findings, both useful:

- **The 96 MB V-Cache is worth nothing to this workload.** At 114 GB/token streamed there
  is nothing to retain. This independently confirms that the bf16 bench's 64-72 GB/s —
  above DRAM speed — was an artefact of re-reading its own 176 MB working set, and that the
  engine never sees that reuse. One less reason to trust the bench as a predictor.
- **CCD0 is SLOWER than CCD1 despite 3x the cache** (6.204 vs 5.970 at the same 8 cores),
  and has the widest run-to-run spread of any configuration measured on this box. On a
  7950X3D the V-Cache die is clock-limited; for a bandwidth-bound workload that is a pure
  loss with no compensating benefit.

What actually matters is **physical core count**: every 8-core configuration lands at
5.97-6.20 and every 16-core one at 5.61-5.80, regardless of cache. Even though every major
term is bandwidth-bound, the non-streaming work — mxfp4 unpacking, rmsnorm, router top-k,
SiTU, attention scores — still needs cores.

**Best configuration after a day of measurement:**
`--trunk-gb 114 --cache-gb 2`, `OMP_NUM_THREADS=16`,
`OMP_PROC_BIND=spread OMP_PLACES=cores`, both CCDs — **5.611 s/token**, from 6.275 at the
start of the day. Byte-identical output (`b7b340f40393`) at every single configuration
tested today, which is what makes the comparisons worth anything.

## `--spec` batched decode — a perfect null, and it measures the prompt not the feature

Never run before today. It is the only lever that attacks both dominant terms at once: one
batched sweep verifies k drafted tokens, so the 114 GB trunk stream is paid once instead of
k times, and `k3_moe_prefill` dedups experts across the batch.

gen 32, best config, against a serial control. **One printed step per outer sweep, so
sweeps < tokens is the acceptance signal** — plain `--spec` prints no counters of its own,
unlike the `--draft-trunk` path.

| variant | tokens | outer sweeps | wall | expert GB |
|---|---|---|---|---|
| serial | 32 | **32** | 248.0 s | 1000.56 |
| `--spec 2` | 32 | **32** | 247.9 s | 1000.56 |
| `--spec 4` | 32 | **32** | 247.9 s | 1000.56 |
| `--spec 8` | 32 | **32** | 247.9 s | 1000.56 |

**Not one draft was ever accepted at any setting.** Wall identical to 0.04%, expert bytes
identical to the gigabyte, output identical.

The drafter almost certainly never *fired* rather than firing and being rejected: a
rejected draft pays a replay sweep and would read as slower, not identical. `spec_draft`
needs a 4-token (then 3-token) suffix that already occurred, with every occurrence agreeing
on what follows, and this run generated ordinary prose — *"This is a classic interview
question that tests your understanding of pointers and linked list manipulation"* — with no
repeated 3-grams. The engine's own comment predicts exactly this, including that it costs
nothing when it misses, which the 0.04% confirms.

**What this does NOT establish.** It measured the prompt, not the mechanism. Three outcomes
remain indistinguishable from this data: drafts fire and help, drafts fire and hurt (the
replay-sweep case upstream measured at 0.91x on code), or the drafter is broken. A follow-up
with a prompt that is repetitive by construction separates them — recorded because
concluding "speculative decode is useless here" from a prompt it structurally cannot match
on would be exactly the kind of overreach this file exists to prevent.

## `--spec` mechanism test: it works, it is exact, and it changes what the bottleneck IS

Repetitive-by-construction prompt (79 ids), gen 24, best config. Prefill is 233.5 s in all
three runs, so the decode phase is directly comparable.

| | tokens/sweep | s/sweep | s/token | GB/token | ids |
|---|---|---|---|---|---|
| serial | 1 | 5.64 | 5.64 | 25.83 | 9f5210b80ca9 |
| `--spec 4` | 5 (full acceptance) | 21.42 | **4.28 (-24%)** | **16.75 (-35%)** | 9f5210b80ca9 |
| `--spec 8` | 7 | 28.14 | **4.02 (-29%)** | **14.72 (-43%)** | 9f5210b80ca9 |

Output hash identical across all three, which is what batched greedy verification promises
and is the first thing that had to hold. So the earlier null was the prompt, not the feature.

**But the marginal cost of an extra verified position is 3.75 s, 66% of a serial token,
against the ~22% the engine's own usage text claims.** That gap is the interesting part,
and the budget for one 7-position sweep explains it from measured rates alone:

| term | seconds |
|---|---|
| expert NVMe, 101.83 GB at 11.11 GB/s | 9.17 |
| DRAM: trunk 114 GB **once** for the sweep, plus experts, at 47.9 GB/s | 4.51 |
| **bf16 compute: 7 x 113.5 GFLOP at 64.4 GFLOP/s** | **12.3** |
| predicted | 26.0 |
| **measured** | **28.14** |

**BATCHING FLIPS THE BOTTLENECK.** At T=1 every trunk weight is read once and used once —
arithmetic intensity 1 FLOP/byte, and the machine is DRAM-bound, which is what every
measurement in this file has shown. At T=7 the weights are still read once but seven times
the arithmetic is performed: intensity becomes 7 FLOP/byte and **compute becomes the
largest single term in the sweep**.

Two consequences, both of which change earlier conclusions in this file:

1. **The docs' ~22% assumes a STREAMING trunk.** Full trunk pinning, adopted this morning
   for a free 6.1%, already removed the per-token trunk read that `--spec` would otherwise
   amortise. The two optimisations attack overlapping costs and **do not compose** — the
   same non-additivity already recorded for the PCIe repair and thread count. Measuring
   each against a fixed baseline and adding them up would have overstated both.
2. **This is the precise condition under which an accelerator pays, and it is measurable
   rather than assumed.** At batch 1 a GPU gains nothing here: the work is
   DRAM-bandwidth-bound and PCIe 5.0 x16 at 64 GB/s is barely above this box's 47.9 GB/s of
   DRAM, so streaming weights across it cannot beat keeping them in DRAM. Batched, the
   limit becomes the CPU's ~64 GFLOP/s of bf16 matmul — and that is exactly the quantity an
   accelerator replaces. **The case for a GPU is a case for batching first;** without
   batching there is no compute-bound term for it to take over.

**Caveat, and it is a large one.** This prompt is repetitive by construction and gets full
or near-full draft acceptance. Ordinary prose got ZERO acceptance in the previous test. The
honest summary is that `--spec` is worth ~29% of decode *when the output is repetitive*,
nothing at all when it is not, and costs nothing either way. Its real-workload value is
still unmeasured, and would need a code-generation prompt to settle.

## RETRACTION: "batching flips the bottleneck to compute" is wrong

I wrote that a `--spec` sweep pays the 114 GB trunk stream ONCE for k tokens, and concluded
that batching makes the engine compute-bound. **The source says otherwise and I should have
read it before publishing the conclusion.**

`k3_matmul_bf16` and `k3_mmw` are matrix-times-VECTOR kernels — they take a single
activation `x`, not a `[T][E]` block. **The engine contains no GEMM anywhere.** In
`moe_prefill_chunk`, the only batched path in the codebase, step 2 genuinely dedups the
routed experts (fetch each unique expert once, apply to every token that chose it), but
step 3 is a plain `for (t = 0; t < T; t++)` calling `k3_mmw` four times per token for `up`,
`sh1`, `sh3`, `sh2`. Attention is the same. So a T-position sweep re-reads all 108.81 GB of
trunk **T times**.

Re-deriving the 7-position sweep with the trunk NOT amortised:

| term | seconds |
|---|---|
| expert NVMe, 101.83 GB at 11.11 GB/s | 9.17 |
| **trunk DRAM, 7 x 108.81 GB at 47.9 GB/s** | **15.90** |
| expert DRAM, 101.83 GB at 47.9 GB/s | 2.13 |
| predicted | **27.2** |
| measured | **28.14** |

This closes *better* than the compute model it replaces (26.0 predicted), and it does so
without needing a compute term at all. Per token it also reproduces both configurations:
serial 2.33 + 2.27 + 0.54 = 5.14 s against 5.64 measured; spec8 1.32 + 2.27 + 0.31 = 3.90 s
against 4.02 measured.

**So the entire `--spec` gain is routed-expert dedup on NVMe. The machine never leaves the
bandwidth-bound regime, and there is no compute-bound term for an accelerator to take over.**

**What this makes sharper rather than weaker.** This is a MISSING GEMM, not a physics limit.
A true matrix-times-matrix kernel would read each weight once per sweep instead of once per
position: 15.90 s -> 2.27 s on a 7-position sweep, about **48% off the sweep**, taking it to
~2.07 s/token against 5.64 serial — roughly **2.7x**. And only after that does arithmetic
intensity rise from 1 to 7 FLOP/byte and compute become the limit. So:

> **The case for an accelerator is a case for a GEMM first.** Without one, batching moves no
> bytes and a GPU inherits the same DRAM-bound problem across a slower bus. With one, the
> workload becomes compute-bound and an accelerator is exactly the right answer. The order
> matters and the measurement establishes it.

Recorded as a lesson about method, not just about GEMM: I derived a budget that closed to
within 8% and treated the agreement as confirmation. A wrong model can still fit when two of
its terms are individually plausible, and the compute term I invented (12.3 s) was close to
the trunk term I had omitted (15.9 s). **Closing to within 8% is not proof of mechanism.**

### Confirmed by counter, not just by reading the source

Same prompt, same 24 tokens, serial vs `--spec 8`, DRAM read counter on the whole run:

| | sweeps | DRAM/token | DRAM/sweep |
|---|---|---|---|
| serial | 24 | 363.4 GB | 363.4 GB |
| `--spec 8` | 6 | **341.2 GB (1.06x)** | **1364.9 GB (3.76x)** |

**Per-token DRAM is flat while per-sweep DRAM scales with positions.** If the trunk were read
once per sweep, per-token traffic would have fallen ~4x. It fell 6%. The 6% is the routed
experts being deduped; the trunk is untouched by batching.

Two caveats on these absolute numbers, both mine to own:

- 363 GB/token here includes a 79-token prefill amortised over only 24 generated tokens, so
  it is **not** comparable to the clean 114.0 GB/token decode figure from the controlled
  gen16-minus-gen8 difference. This run establishes the RATIO, which is unambiguous; the
  earlier run establishes the LEVEL.
- 79 prefill positions x 108.81 GB would be 8,596 GB for prefill alone, yet the entire run
  measured 8,721 GB. The per-token loop reuses the same weight matrix across iterations, so
  the 88 MB shared-expert and 51 MB latent matrices fit inside the 96 MB V-Cache and are
  partially amortised **by accident of cache size, not by design**. The 176 MB KDA
  projections do not fit and get no such help. That is also why the V-Cache die measured
  worthless for single-token decode but is doing quiet work during prefill.

## Measuring the GEMM prize before building one

The estimate was ~2.7x. Arithmetic has been wrong twice today while still closing to within
8% of measurement, so it was measured instead. Standalone benchmark on the real KDA
projection shape (7168 x 12288 bf16 = 176 MB, 56% of all trunk traffic), 8 matrices cycled
so the 1.41 GB working set cannot sit in the 96 MB V-Cache. Both paths share one scalar
inner loop and the engine's double accumulator, so the ratio is the result.

| T | GEMV x T | naive GEMM | **packed GEMM** | naive sp | **packed sp** | packed GF/s |
|---|---|---|---|---|---|---|
| 1 | 0.006 | 0.009 | 0.009 | 0.64x | 0.64x | 18.8 |
| 2 | 0.008 | 0.009 | 0.010 | 0.82x | 0.81x | 37.0 |
| 4 | 0.015 | 0.010 | 0.010 | 1.55x | 1.53x | 72.7 |
| 8 | 0.029 | 0.017 | **0.010** | 1.78x | **3.05x** | 145.9 |
| 16 | 0.060 | 0.055 | **0.011** | 1.09x | **5.67x** | **267.0** |

**The packed GEMM's time is flat: 0.009 -> 0.011 s from T=1 to T=16.** That flatness is
weight amortisation made visible — the matrix is read once and the time barely grows with T.

**DATA LAYOUT ALONE IS WORTH 5x.** The naive version reads `x[t*IN + i]`, putting the T
activations for one `i` 28 KB apart, so every inner step touches T scattered cache lines; it
peaks at 1.78x and then REGRESSES to 1.09x at T=16. Packing them contiguous as `xT[i*T + t]`
gives 5.67x and still climbing. Same algorithm, same arithmetic, same instruction count —
only where the operands sit. This is the clearest instance in the whole campaign of the
principle that a structural choice no one would call "the algorithm" dominates the outcome.

**Batching does flip the bottleneck, but only with a real GEMM.** Packed at T=16 moves
176 MB in 0.011 s = 16 GB/s, far under the 47.9 GB/s ceiling, while reaching 267 GFLOP/s.
GEMV by contrast pins at 47.2 GB/s at every T — exactly the measured DRAM ceiling, which is
independent confirmation that GEMV is purely bandwidth-bound. So the earlier retraction was
right about THIS engine (it has no GEMM) and the original intuition was right about the
physics (batching can flip it). Both halves needed measuring to separate.

**Scaled to the engine**, a packed GEMM would take a 7-position `--spec` sweep from 28.14 s
to roughly 17.6 s, about **2.24x on repetitive decode**. So the 2.7x estimate was close on
the prize and completely silent on the difficulty: the obvious implementation returns 1.09x,
not 2.24x, and the difference is invisible until measured.

**Order of operations, now established rather than assumed:** a GEMM is the precondition for
any accelerator argument. Without one the workload stays DRAM-bound and a GPU inherits the
same problem across a slower bus (PCIe 5.0 x16 = 64 GB/s against our 47.9 GB/s DRAM). With
one, the workload reaches 267 GFLOP/s of CPU compute and an accelerator has something real
to replace.

---

# First engine change: split expert reads so the batch can rebalance

Branch `perf/expert-read-chunking`, commit `cfcf4d1`, committed locally and NOT pushed.
This is the campaign's first modification to upstream code; everything before it was
measurement.

## The change

`cache_getmany` issued one read per expert. With top-k experts and an equal number of
OpenMP threads, `schedule(dynamic)` had nothing to balance — the layer could not finish
until the slowest read did. Dividing each expert's enclosing aligned window into chunks
gives `nw * nchunk` work items, so a thread that finishes early takes more and the tail
costs one chunk rather than one whole expert.

## Result, one binary, only `K3_EXPERT_CHUNK` changed

| chunk | per expert | s/token | expert GB/s | ids |
|---|---|---|---|---|
| unsplit (original) | 1 | 5.602 | 11.14 | b7b340f40393 |
| 4 MiB | 5 | 5.402 | 11.96 | b7b340f40393 |
| 2 MiB | 9 | 5.325 | 12.26 | b7b340f40393 |
| **1 MiB** | **17** | **5.281** | **12.40** | b7b340f40393 |
| 512 KiB | 34 | 5.286 | 12.34 | b7b340f40393 |

**5.7% off a decode step, and 78% -> 87% of what the hardware delivers.** The curve turns at
512 KiB, so 1 MiB is an optimum rather than a floor — past it the extra syscalls cost more
than the improved balance saves.

Gates, all green: `make test` reports **ENGINE MATCHES THE REFERENCE EXACTLY** with GATE 1b
logits bit-identical and GATE 3 incremental 20/20; both bench FNV1a hashes unchanged
(`83c8504a4cb3fac6`, `a231061237b5579d`); `test_cache` 5/5 at every chunk size; generated
ids identical at every point.

## A bug I introduced, and how it was caught

**1 MiB segfaulted on the first sweep.** `cache_getmany` has a second caller I had not
accounted for: `moe_prefill_chunk` passes the batch's *unique* expert set, so `nw` reaches
`K3_MAX_TOPK` (64), not top-k (16). At 17 chunks each that is 1,088 items against 512-entry
index arrays, and the guard was off by one — at `nitem == 512` it clamped `nchunk` to 1 and
then still wrote `owner[512]`.

Fixed with a per-expert budget of `K3_EXPERT_ITEMS / nw`, which cannot overflow by
construction, **and which recomputes the chunk SIZE when the count is capped**. Capping the
count alone would have left the window tail unread, the payload short, and the slot released
as a failed load — silently dropping an expert, which is worse than a crash.

Two lessons, both about method rather than about C:

- **I sized a buffer for the caller I was looking at, not for every caller.** The other one
  was thirty lines away in a file already read twice today.
- **`test_cache` passed the entire time and proved nothing.** Its fixture experts are 1,632
  bytes, so `nchunk` was always 1 and the new path never executed. A green test that does
  not reach the changed code is not evidence. That is why the chunk size became an env knob
  — following the engine's own `K3_NOPREFETCH` / `K3_NOHUGE` convention — so the fixture can
  be forced through the split. Both crash configurations are now permanent entries in the
  sweep as regression checks.

## Before proposing this upstream

- Measured on **one machine, one storage layout**. The gain comes from tail latency across a
  two-drive md RAID1 at PCIe 4.0 x4; a single drive or a higher-latency device may show less,
  and more syscalls could cost more than they save. The env knob matters precisely here: a
  maintainer can reproduce both sides on their own hardware instead of trusting these numbers.
- Wants a longer confirmation run than gen 14 before submission.
- The repository is a shallow clone of `FareedKhan-dev/kimi-k3-in-c`; a real PR needs a fork.

## End-to-end confirmation, gen 64, same binary and same session

Both configurations run back to back rather than compared against an old log, so nothing
between them can have drifted. The morning configuration is REPRODUCED
(`K3_EXPERT_CHUNK` large enough to force the original one-read-per-expert path, unbound
threads, old trunk/cache split), not quoted.

| | this morning | now |
|---|---|---|
| **total for a 64-token answer** | **473.5 s** | **407.0 s** |
| average | 7.40 s/token | **6.36 s/token** |
| steady decode | 6.284 | **5.299** |
| prefill | 77.52 s | 73.17 s |
| peak RSS | 122.91 GB | **118.81 GB** |
| trunk bytes read from disk | 508.00 GB | **108.81 GB** |
| expert bytes read | 1827.11 GB | 1827.11 GB |
| expert read rate | 10.25 GB/s | **13.34 GB/s** |
| I/O share of wall clock | 49.3% | **36.7%** |
| generated ids | `38aff30e2031` | `38aff30e2031` |

**66.5 s saved on one answer: 14.0% faster, 1.16x, 8.11 -> 9.43 tokens/min, on 4.1 GB LESS
memory, with byte-identical output.**

Three checks that make this trustworthy rather than merely favourable:

- **The control reproduces.** 473.5 s against the 474.0 s logged this morning, 0.1% apart.
  The comparison is the change, not drift.
- **The prediction held.** Recorded before running: ~410 s and ~6.4 s/token. Measured 407.0
  and 6.36, within 0.7%. After a day of predictions that missed — MRU, 24 threads, DRAM at
  139 GB, the GEMM amortisation — the cost model is finally predictive.
- **Expert rate reached 13.34 GB/s, 93% of the 14.32 GB/s fio ceiling**, better than the 87%
  seen at gen 14. The same 1827.11 GB moved in 41 fewer seconds.

Expert retention fell from 0.55% to 0.11% because the cache is now 2 GB rather than 10 GB.
Both are zero for practical purposes, exactly as the reuse-distance analysis predicted, and
the identical expert byte count at both sizes is the direct confirmation.

---

# Second engine change: a batched bf16 GEMM

Branch `perf/expert-read-chunking`, uncommitted. The kernel is correct and the wiring is
correct; the measured value so far is **much smaller than projected**, for reasons worth
recording.

## The kernel

`k3_matmul_bf16_batch`: T activations against one weight matrix, read once. **Bitwise
identical** to calling the serial kernel T times — each output element keeps its own
sixteen double accumulators and sees the same fma sequence in the same order. Since the
AVX2 and NEON forms are already bit-identical to the scalar form, matching the scalar form
matches all of them.

- `tests/unit/test_batch.c`: **8/8 shapes bit-identical**, including ragged tails
  (in not a multiple of 16) and `out` below the OpenMP threshold.
- `make test`: **ENGINE MATCHES THE REFERENCE EXACTLY**, logits bit-identical, GATE 3 20/20.
- Both bench FNV1a hashes unchanged.
- Isolated DRAM measurement: serial x16 reads **1925 MB**, batched reads **179 MB** for a
  176.2 MB matrix — **10.8x fewer DRAM reads**, doing exactly what it claims.

**Exactness cost most of the speed.** The prototype reached 5.58x keeping ONE accumulator
per output; the shipping kernel needs SIXTEEN PER ACTIVATION to preserve the reduction tree,
2 KB per output row at T=16, which spills out of registers. Measured on the real KDA shape:
1.32x / 1.46x / 2.04x / 2.29x / **2.39x** at T = 1/2/4/8/16. I would rather have 2.39x that
passes the oracle than 5.58x that does not.

## Wired into the three largest projections

`k3_kda_layer`'s q, k and v — 176 MB each, 36.5 GB of the 72.4 GB of per-token attention
traffic. b, f_a and f_b stay serial: 0.09-0.22 GB/token combined, and f_b consumes f_a's
single shared scratch slot.

## Measured, and the projection was wrong

| | prefill | total | DRAM |
|---|---|---|---|
| serial | 204.15 s | 209.5 s | **4858 GB** |
| batched | 204.36 s | 209.7 s | **3918 GB** |

**DRAM traffic fell 19.4% and the time did not move at all.** That is not a failure, it is a
diagnosis: **prefill is COMPUTE-bound, not DRAM-bound.** 3918 GB at 47.9 GB/s is 81.8 s of a
204 s prefill, only 40%. At T=79 the arithmetic intensity is already high, the FLOP count is
unchanged by batching, and removing bytes therefore buys nothing.

On `--spec 8`, where a 7-position sweep genuinely is trunk-DRAM dominated:
**mean sweep 17.22 -> 16.77 s, 2.6%.** Output identical (`9f5210b80ca9`).

**Why 2.6% and not the 2.24x projected.** Three reasons, all of which the projection ignored:

1. **Only 33.6% of the trunk is wired.** The MoE's up / sh1 / sh3 / sh2 (33.8 GB/token) and
   the MLA projections are still serial.
2. **The serial path already had cache reuse.** Measured in isolation: serial x16 read
   1925 MB where zero reuse would be 2819 MB, so 32% was already being served from L3. The
   available saving was never the full T x.
3. **The place with the most positions, prefill, is the place least able to use it.**

## A TEST-HARNESS BUG THAT WOULD HAVE PRODUCED A WRONG CONCLUSION

The first three A/B runs showed identical time, identical DRAM and identical profiles, and I
was one step from recording "the GEMM does nothing in the engine".

The cause was mine: the harness set `K3_NOBATCH="$nb"` with `nb=""` for the batched arm, and
**`getenv()` returns a non-NULL pointer for an empty string**, so the kill switch fired in
BOTH arms. Every comparison had batching disabled on both sides.

What exposed it was a profile taken with no `K3_NOBATCH` in the environment at all, which
showed `k3_matmul_bf16_batch` at 12.66% of CPU — the kernel was plainly running when the
harness said it could not be. **Two measurements that disagree are worth more than either
alone**; the honest reading is that a null result should be distrusted until something
independent confirms the treatment was actually applied.

Fixed with `env -u K3_NOBATCH`. The 19.4% DRAM reduction only appeared after that.

---

# Full payload matrix (2026-09-22, before any hardware request)

Before asking for more SSDs, re-measure everything together, across payloads that differ in
shape, with every fix applied at once. Runner: `k3-results/bench-matrix.sh`.

Payloads: `p1_tiny` 24 B, `p2_medium` 234 B prose, `p3_repetitive` 399 B, `p4_long` 9.6 KB,
`p5_code` 208 B. Arms are cumulative: `base` -> `pin` -> `bind` -> `chunk` -> `tuned`.

## What "the baseline" even is — three different answers

| reference | trunk / cache | note |
|---|---|---|
| naked default, no flags | **16 GB / 64 GB** | what you get typing the documented command |
| `--trunk-gb auto` | 111 GB / 10.4 GB | **cannot start on this box, see below** |
| tuned | 114 GB / 2 GB | this session |

## AN ENGINE BUG: `--trunk-gb auto` refuses its own plan

```
REFUSING TO START: this needs 126.78 GB and the machine has 132.35 GB
available, a shortfall of -5572078356.48 B.
```

Two defects, and my first reading of it was wrong.

**I called this an inverted comparison. It is not.** The guard in `src/cli/k3_run.c:1087` is
`if (need_b > have * 0.95)`, a deliberate 5% margin, and 126.78 > 125.73 is correct. Reading
the source rather than trusting the message is what corrected me.

1. **The message is wrong.** It prints `need_b - have`, which is NEGATIVE whenever the
   refusal comes from the margin rather than from a genuine shortage. The true shortfall
   against the guard is 1.05 GB. A negative shortfall is nonsense on its face and sends the
   reader looking for a memory problem that does not exist.
2. **Auto and the guard disagree about the margin.** Auto reserves `2 GB + 2%` (4.65 GB); the
   guard demands 5% of MemAvailable (6.62 GB). On a 124 GiB box auto therefore always emits a
   plan its own guard rejects. Auto is unusable at exactly the size it was written for.

Both are worth reporting upstream. The baseline arm here is `--trunk-gb 108 --cache-gb 10`,
which is what auto WOULD have chosen, rounded down until it clears the guard.

## THREE HARNESS FAULTS CAUGHT BEFORE THEY BECAME RESULTS

Recorded because each produced a confident, wrong-looking-right number.

**1. A stale binary answered for a build that never happened.** The compile failed with
`k3.h: No such file or directory`, and `/tmp/test_batch` from an earlier session ran instead
and printed `BATCHED MATMUL IS BIT-EXACT`, 8 PASS. The failure line and the PASS lines were
adjacent in the same output. Fix: `rm -f` the target first, then require `COMPILE OK` before
believing any test result. **Same family as the `getenv("")` bug earlier today — for the
second time in one day, a green result came from code that was never exercised.**

**2. `make clean` deleted the cache fixture, so `test_cache` aborted** while my grep counted
`PASS=0 FAIL=0` and printed it next to the other passes. A zero-of-zero is not a pass. Then I
invoked it without its fixture argument and got four more false aborts. Fix: the parser now
prints the verdict line, and a count of zero is treated as a failure.

**3. My steady-state mean was wrong by 12%.** The summary line
`8 tokens in 68.6 s, 8.57 s/token average` begins with a digit, so `$1 ~ /^[0-9]+$/` matched
it and `$3` was the word `in`, contributing 0 to the sum and 1 to the count. Seven steps at
5.29 became `37.03 / 8 = 4.625`. **It made the engine look 12% faster than it is.** Fix:
require `NF==6` and a numeric `$3`; a `parse` subcommand now prints every accepted and every
rejected row so the parser can be checked against a log whose answer is already known. It
reproduces 5.287 on the calibration log, where the old one said 4.626.

The general rule these three share: **a measurement harness needs its own oracle.** Every one
of these was caught by comparing against a number I already knew, never by the harness itself.

## Result 1: the gain is the SAME on every payload

Plain incremental decode, steady state, seconds per token. `base` = t108/c10 at 32 unbound
threads with both code changes disabled; `tuned` = t114/c2, 16 bound threads, 1 MiB chunks,
batched GEMM.

| payload | prompt | base | tuned | gain | token ids |
|---|---|---|---|---|---|
| p1_tiny | 5 tok | 6.360 | **5.291** | 16.8% | identical |
| p2_medium | ~50 tok | 6.387 | **5.301** | 17.0% | identical |
| p3_repetitive | ~200 tok | 6.507 | **5.430** | 16.6% | identical |
| p5_code | ~50 tok | 6.391 | **5.310** | 16.9% | identical |

**16.8% mean, and the spread across four very different payloads is 0.4 percentage points.**
That flatness is itself the finding: steady-state decode cost is set by the weights, which are
the same every token, not by what the prompt says. Peak RSS also falls, 122.8 -> 118.7 GB.

Token ids are identical within every pair. They are also identical ACROSS configurations:
the naked default's three tokens `[17374, 20829, 10]` are an exact prefix of the tuned run's
eight. Every arm in this matrix, including the 16 GB/64 GB default, produces the same text.

## Result 2: the fixes are NOT additive, and one is harmful alone

Cumulative, each row adding one change to the row above, on p2_medium:

| arm | what it adds | s/token | vs previous |
|---|---|---|---|
| base | t108/c10, 32 unbound threads | 6.383 | -- |
| pin | t114/c2 | **10.461** | **+64% SLOWER** |
| bind | 16 threads, spread, cores | 5.629 | -46% |
| chunk | 1 MiB expert reads | 5.303 | -5.8% |
| tuned | batched GEMM | 5.297 | -0.1% |

**Pinning more trunk while shrinking the cache is actively harmful until the threads are
bound.** The two changes I previously reported as independent wins (6.1% and 4.9%) are not
independent at all: at 32 unbound threads the bigger pin costs 64%, and binding then recovers
far more than its own headline number. Reported separately they mislead in both directions.

This also means **the ablation ORDER decides the attribution.** Adding pin first makes it look
catastrophic; adding bind first would make pin look free. Only the endpoints, 6.383 -> 5.297,
are order-independent. A 64% outlier is more likely a disturbed run than a real effect, so
group H repeats both arms twice, interleaved.

## Result 3: the batched GEMM is worth about 1%, not 2.6%

Speculative decode needs a different metric. Its rows are heterogeneous -- some sweeps cost
27.6 s and read 123 GB, others cost 5.44 s and read 25.8 GB -- so a mean over rows depends on
how many of each happened to occur. **Decode seconds excluding prefill, per token:**

| comparison | decode s | tokens | s/token | gain |
|---|---|---|---|---|
| spec, base -> tuned | 108.67 -> 92.61 | 24 | 4.53 -> 3.86 | 14.8% |
| spec, chunk -> tuned (**GEMM alone**) | 71.71 -> 71.02 | 16 | 4.48 -> 4.44 | **0.96%** |
| plain decode, chunk -> tuned (GEMM alone) | -- | -- | 5.303 -> 5.297 | **0.1%** |

**The batched GEMM earns about 1% where T > 1 and nothing at all where T = 1**, which is what
it should do: at one position per step there is nothing to batch. The 2.6% recorded earlier
came from a single noisier A/B; this is the better-controlled number and it is smaller.

A near miss worth recording: the generated text on the repetitive payload is ` 1 2 3 4 5 1 2 3`,
which begins with a digit and was rejected only because it has eight fields rather than six.
**A six-token numeric output would have been counted as a timing row.** An audit across all 27
logs (69,569 bytes, 18 containing the marker) found no contaminated row, and the audit itself
was validated by relaxing the field test until it did fire.

---

# The fixes as committed, and one clean run on them (2026-09-23)

Everything above was measured with the code changes sitting uncommitted in a working tree.
They are now four separate commits on `perf/expert-read-chunking`, each building on its own
with zero warnings, followed by one isolated run with nothing else on the machine.

## A claim I nearly shipped

The kernel's own comment said **"3.11x at T=8 and 5.58x at T=16"**. Those are the numbers from
the ONE-ACCUMULATOR PROTOTYPE, which was discarded precisely because it does not reproduce
the serial reduction tree. The shipping kernel measures 2.04x and 2.39x. The comment had been
written when the prototype was the thing being measured and never revised when the exactness
requirement halved the gain.

Committing it would have published a performance claim I already knew was false, inside the
file that makes the claim hardest to check. Corrected before the commit, with the prototype's
number kept and labelled as the discarded alternative, because the gap between 5.58x and 2.39x
IS the cost of exactness and is worth a reader's time.

An earlier version of the same failure is one commit away: a hash `cfcf4d1` was reported in a
previous handoff as the commit carrying the chunking change. **No such commit exists.** The
change was never committed at all. Anything that says `cfcf4d1` is wrong.

## The four commits

| commit | change | independently builds |
|---|---|---|
| `e655eb0` | `perf(cache)`: split each expert read into 1 MiB chunks | yes, 0 warnings |
| `7c5f6d0` | `perf(ops)`: batched bf16 matmul, bit-identical, + `test_batch` + Makefile | yes, 0 warnings |
| `7919565` | `perf(kda)`: read q, k and v once per sweep, not once per position | yes, 0 warnings |
| `9977044` | `bench`: time the batched matmul against the serial kernel | yes, 0 warnings |

The kernel and its single call site are deliberately separate commits. `test_batch` proves the
kernel on its own, so `7c5f6d0` is reviewable and testable without the wiring; `7919565` then
carries the end-to-end numbers and the honest note that they are smaller than the kernel
suggests. Splitting them needed the wiring lifted out and put back, done by an exact-string
script that **refuses rather than guesses** if the text is not where it expects -- a silent
no-op would have produced a commit claiming to be kernel-only while carrying the wiring. The
file returned to exactly its original 100,706 bytes, which is the check that it was lossless.

`test_batch` is now in `UNIT_TESTS` and runs inside `make test`. That is a direct fix for the
stale-binary failure: a test only run by hand is a test that can silently not run.

## The clean run

`commit 9977044`, working tree clean, machine otherwise idle, nothing else measured alongside.
Prompt `p2_medium` (46 tokens), 64 generated, incremental.

```
64 tokens in 487.4 s, 7.62 s/token average
PEAK RSS 118.93 GB
```

| quantity | value |
|---|---|
| **steady-state decode** | **5.311 s/token** over 63 steps |
| spread | min 5.29, median 5.31, max 5.33 -- **0.8% peak to peak** |
| prefill, 46 tokens | 152.77 s |
| expert reads | 2055.98 GB over the run, 25.83 GB/token at 13.8 GB/s |
| trunk | 108.81 GB read once in 12.18 s, then pinned; 98.4% bind hit rate |
| I/O share of wall clock | 36.2% |
| generated ids | `599d822ce3c3`, 64 tokens, coherent on-topic continuation |

**5.311 against 5.301 measured for the same payload in the matrix: 0.2% apart on independent
runs of independently built binaries.** That agreement is the reason the 16.8% figure can be
stated at all.

Where a token goes, in steady state: expert NVMe 1.87 s (35%), trunk DRAM 108.81 GB at the
measured 47.9 GB/s = 2.27 s (43%), everything else 1.17 s (22%). **No storage upgrade can
touch the 43%**, which is the single largest term and is memory bandwidth.

---

# Pause and understand the options (2026-09-23)

Written before choosing anything, and deliberately describing each approach on its own terms
first. The failure mode being guarded against is the obvious one: we have been optimising
storage for days, so storage arguments will feel more persuasive than they are.

Provenance is marked on every claim: **[M]** measured here, **[D]** derived from measurements
here, **[A]** assumed or from a spec sheet and NOT verified.

## The one measurement that reframes the whole question

Prefill processes many positions in one sweep. Decode processes one. Same engine, same run,
same weights — so the difference between them is a direct measurement of what batching is
worth on this machine. Subtracting the one-time trunk load from step 0:

| prompt tokens | prefill s/token | decode s/token | advantage |
|---|---|---|---|
| 5 | 3.878 | 5.291 | 1.36x |
| 46 | 3.056 | 5.311 | 1.74x |
| 64 | 2.841 | 5.310 | 1.87x |
| 399 | **2.462** | 5.430 | **2.21x** |

**[M] Monotonic in batch size, and still improving at 399 positions.** Processing tokens
together already costs less than half per token what processing them one at a time does.

The mechanism is not mysterious: the trunk is 108.81 GB and is read once per SWEEP, not once
per token. At one position per sweep, every byte is used exactly once — arithmetic intensity
of 1 FLOP/byte, which is why the CPU idles at 6% of peak while the memory bus is pinned at
100%. **The machine is not slow. It is being asked to do the least cache-friendly thing
possible, one token at a time.**

This matters for hardware because it changes WHICH resource binds. [D] If decode batched at
B=16 the way prefill does, per-token trunk cost would fall from 2.27 s toward 0.14 s, and
expert I/O at 1.87 s would become the dominant term — **only then does more storage become
the main lever.** Buying drives first optimises the resource that is second in line.

## The approaches, each on its own terms

### 1. Two more NVMe drives

**What it does.** More flash channels in parallel, raising aggregate sequential read
bandwidth. It does not reduce bytes read; it raises the rate.

**What it assumes.** That expert reads are bandwidth-limited, on the critical path, and that
the array scales with spindle count.

**Evidence.** [M] Expert reads are 1.87 s of a 5.311 s step, 35%. [M] The engine already
extracts 96% of what the 2-drive array can deliver at its access pattern, so the drives are
genuinely saturated — this is not a software problem any more. [M] When chunking raised the
expert rate from 11.14 to 12.40 GB/s, roughly three quarters of the I/O time saved showed up
as step time, so the read is largely on the critical path rather than hidden behind compute.

**Verdict.** [D] ~17% in single-stream decode, hard ceiling 35%. [M] Separately and
independently, capacity is a real constraint: 94% full, 106 GB free, which blocks holding a
second quantised copy of the trunk. **[A] The lane budget is the catch — 16 free PCIe lanes
is four CPU-attached NVMe OR one x16 accelerator, not both.** Choosing drives forecloses the
accelerator path on this chassis.

### 2. A GPU

**What it does.** Moves matmuls to a device whose local memory bandwidth is 20-60x this
box's. [A] H100 HBM is roughly 3 TB/s against our measured 47.9 GB/s.

**What it assumes — and this is the whole question.** That the weights are RESIDENT in device
memory. An accelerator is a bandwidth machine; it only helps with the bandwidth it owns.

**Evidence.** [M] The trunk is 108.81 GB per token. [D] Streaming it to a device does not
avoid the DRAM read — the bytes must come off DIMMs at 47.9 GB/s before they can reach PCIe
at all, so streaming adds a hop to the existing bottleneck rather than removing it. The
comparison "PCIe 5.0 x16 at 64 GB/s beats 47.9 GB/s DRAM" is the wrong comparison, and an
earlier version of this document made it.

**Verdict.** [D] A GPU helps this workload only if it holds ~109 GB of trunk in its own
memory. That is two 80 GB cards or one 192 GB card, not a consumer part. **Below that
threshold it is not a smaller win, it is no win**, because the binding constraint is untouched.
[M] The remaining compute is ~22% of the step and the CPU is at 6% of its arithmetic peak, so
there is little compute-bound work for an accelerator to rescue.

### 3. More RAM

**What it does.** Holds more resident.

**Evidence.** [M] The trunk is ALREADY 100% pinned — 93/93 layers, 108.81 GB of a 110 GB
packed trunk. More RAM cannot reduce trunk reads, because none of them go to disk. [M] The
expert cache returns ~0% on ordinary prompts, so more cache does not help there either,
though [M] on a repetitive prompt a 10 GB cache retained 63.35% and cut expert I/O 2.3x.
[M] What more RAM DOES buy is context: an 1861-token prompt is refused today for want of
4.41 GB of KV cache plus 1.25 GB of buffers.

**Verdict.** [D] Nothing for decode speed on ordinary prompts. Everything for context length.
**If long prompts matter, this is the only option on the list that addresses them** — neither
drives nor an accelerator do.

### 4. Quantising the trunk to int8

**What it does.** Halves the bytes moved per token on the largest single term. [D] 108.81 GB
becomes ~54.4 GB, 2.27 s becomes ~1.14 s — about 21% of the step, the biggest single-change
gain on this list.

**What it costs, and it is not a small thing.** [D] Output is no longer bit-identical.
Every comparison in this entire document rests on identical token ids; that invariant is what
made a 5% claim believable. Quantising replaces a correctness question with a quality
question, which needs an evaluation harness that does not exist here yet.

**Evidence.** [M] Blocked today regardless: 106 GB free is not enough to hold a second copy
of the trunk alongside the bf16 one.

**Verdict.** [D] Highest single-change gain, and the only one that requires giving up the
oracle. Worth doing deliberately and never by accident.

### 5. Multi-sequence batching in decode (software, no purchase)

**What it does.** Serves B sequences in one sweep so the trunk is read once for B tokens.

**Evidence.** [M] The prefill-versus-decode table above IS this effect, measured: 2.21x per
token at 399 positions. [M] The batched GEMM committed today (`7c5f6d0`, `7919565`) is the
primitive this needs, and its unimpressive 1% today is precisely because T is 1 in decode and
at most 8 under `--spec`. [M] Its DRAM reduction in isolation is 10.8x, which is the number
that matters for amortisation, not the 2.39x compute figure.

**What it would take.** [A] Real work: a scheduler, per-sequence KV state, and the remaining
66% of trunk matmuls wired to `k3_mmw_batch`. The engine has no concept of concurrent
requests today.

**Verdict.** [D] Largest measured lever on the list, costs money only in time, and it changes
the ranking of everything else by moving the bottleneck to storage. **It should be understood
before any hardware is bought, because it determines what the hardware would be for.**

### 6. Predicting expert routing to prefetch earlier

**What it does.** Hides expert read latency rather than shortening it.

**Why the earlier ceiling does NOT apply.** [M] The 17.0% cache ceiling was measured for
CACHING, which needs reuse, and reuse is absent: 0 of 33,469 requests had a reuse distance
under 569. Prefetching needs lead TIME, not reuse. They are different properties and the
trace result does not bound this.

**The real obstacle.** [D] Layer L's routing depends on layer L-1's output, so there is no
lead time to exploit without predicting the routing. [A] Speculative expert prefetch is an
active research area and is unverified here.

**Verdict.** [A] Genuinely open. Not ruled out by anything measured, not supported by
anything measured either.

### 7. Faster DIMMs

[M] 3600 MT/s against 4800 rated. [A] 4 dual-rank modules on 2 channels is the reason, and
3600-3800 is typically the practical ceiling for that population. [D] If 4800 were reachable,
47.9 -> ~60 GB/s would take the trunk term from 2.27 to 1.81 s, about 8.7%. [A] Rented
hardware, no BIOS access. Noted for completeness, not actionable.

### 8. A CPU with more memory channels

**What it does.** Attacks the 43% term directly, which nothing else on this list except
quantisation does. [A] A 12-channel EPYC at DDR5-4800 is ~460 GB/s theoretical against our
measured 47.9 GB/s; [A] real sustained figures are commonly 300-350 GB/s.

**Evidence.** [M] Our own scaling data says the ceiling is the bus, not the cores: the bf16
kernel reaches only 18% of linear scaling across 32 threads, and DRAM throughput peaks at
2-4 threads and then declines. [D] At 350 GB/s the trunk term falls from 2.27 s to ~0.31 s,
roughly 37% off the step from a single change.

**Verdict.** [D] On the measurements, this is the hardware change best matched to the actual
bottleneck. [A] It is also a different machine and a different price, and nothing here has
been verified on such a box. Listed because the evidence points at it, not because it is
being recommended.

## Honest ranking, on evidence rather than appetite

| approach | attacks | size | provenance | costs money |
|---|---|---|---|---|
| multi-sequence batching | the 43% trunk term | [M] 2.21x already visible in prefill | measured | no |
| more memory channels | the 43% trunk term | [D] ~37% | derived from [A] specs | yes, new machine |
| int8 trunk | the 43% trunk term | [D] ~21% | derived | no, but costs the oracle |
| two more NVMe | the 35% expert term | [D] ~17%, ceiling 35% | derived | yes |
| more RAM | context length only | [M] unblocks 1861-token prompts | measured | yes |
| expert prefetch prediction | the 35% expert term | [A] unknown | unverified | no |
| faster DIMMs | the 43% trunk term | [D] ~8.7% | derived | not available |
| consumer GPU | nothing here | [D] no win below ~109 GB VRAM | derived | yes |

**The uncomfortable conclusion, stated plainly:** the work of the last two days optimised
storage and threading, and the largest remaining lever is neither. It is that decode does one
token at a time while the machine is built to move 160 GB per token either way. That is a
software change, it is already half-demonstrated by prefill, and it changes what any future
hardware would be for.

---

# The pin anomaly: confirmed, and two hypotheses killed

One observation had said t114/c2 at 32 unbound threads decodes at 10.461 s/token against
6.383 for t108/c10 — 64% worse while reading FEWER bytes in LESS time on 4 GB LESS memory.
A single outlier that large deserved a repeat before it deserved an explanation.

## Phase 1: it reproduces, and it is unstable

Six runs, interleaved so drift cannot favour either arm, each with a kernel-state monitor
attached (`k3-results/monitor.sh`).

| arm | steady s/token | utime | vol ctx switches | expert GB | thp fallback | psi memory |
|---|---|---|---|---|---|---|
| base-1/2/3 | **6.440, 6.440, 6.439** | 5976 s | 85,966 | 609.5 | 0 | 0.0 s |
| pin-1/2/3 | **10.289, 12.323, 11.043** | 7276 s | 21,265 | 609.5 | 0 | 0.0 s |

**Confirmed: it is real, and it is worse than first measured** — 74% on the mean rather than
64%. Three findings come with it:

1. **Base is stable to 0.02%; pin varies by +-9%.** Whatever this is, it is not a fixed cost,
   and the instability is as much a clue as the slowdown.
2. **Pin burns 21.8% MORE user CPU time.** It is executing, not waiting.
3. **Pin makes 4x FEWER voluntary context switches**, 21k against 86k.

## Two hypotheses killed by measurement, both mine

**Huge-page fallback: dead.** The first guess was that a 111.22 GB allocation fails to find
contiguous memory where 104.73 GB succeeds, silently drops to 4 KB pages, and pays a TLB miss
per stream — which would be invisible in any I/O total. It is wrong, and it was cheap to
disprove: `thp_fault_fallback` and `compact_fail` are **0 cumulative since boot**, across
every run including the slow ones, and the monitor confirms the engine really does back
~100 GB with transparent huge pages rather than merely asking for them.

**Slot contention: dead.** The second guess was that 113 cache slots against 569 starve 32
threads, which would appear as threads blocking on a free slot. The opposite is true: the slow
arm blocks **four times less**. Threads waiting on a resource sleep and yield; these do not.

What survives: same bytes, less I/O, less blocking, more CPU burned, unstable run to run.
That combination points at threads spinning rather than threads waiting — the engine's
OpenMP regions busy-wait by default, so time lost at a barrier is charged as user time, not
as idle. But **which** barrier, and why the config change provokes it, is not yet established.

## Phase 2: it is the TRUNK, not the cache

The two arms changed trunk AND cache together, so the original result could not attribute the
effect to either. Four cells, 32 unbound threads throughout:

| trunk | cache | slots | layers pinned | s/token |
|---|---|---|---|---|
| 108 | 10 | 569 | 87 | 6.449 |
| 108 | 2 | **113** | 87 | **6.437** |
| 114 | 2 | 113 | **93** | **10.110** |
| 114 | 5 | 284 | 93 | 10.851 |

**Cutting the cache from 569 slots to 113 costs 0.2%. Raising the trunk from 108 to 114 costs
57%.** The expert cache is irrelevant to this; the earlier reading that paired it with slot
count was coincidence. The 4x drop in voluntary context switches also tracks the trunk.

Per step the penalty is UNIFORM — every decode step is 9.8-10.7 s against 6.4 s, and both
arms read exactly 25.83 GB of experts per step. Not an occasional stall; a standing cost.

## Phase 3: huge pages are not the mechanism either

`K3_NOHUGE=1` forces 4 KB pages for the trunk and cache arenas, so causality can be tested on
one binary rather than argued from counters.

| trunk | huge pages | s/token | dTLB misses | miss % | IPC |
|---|---|---|---|---|---|
| 108 | yes | 6.425 | 162.9 M | 19.0% | 0.495 |
| 108 | **no** | **6.407** | **787.8 M** | 39.7% | 0.503 |
| 114 | yes | 12.750 | 146.2 M | 20.5% | 0.427 |
| 114 | **no** | **10.922** | **741.7 M** | 39.8% | 0.455 |

**Multiplying dTLB misses by 4.8x changes the speed by 0.3%.** Whatever this engine is limited
by, it is not TLB reach — and at trunk 114, turning huge pages OFF was actually faster. The
huge-page hypothesis is dead by direct experiment, not by inference.

What the counters do say: t114 runs **+4% instructions in +21% cycles**, IPC 0.495 -> 0.427.
Same work, more stalling.

## Phase 4: the memory subsystem is delivering less, and being asked for more

Sampling the calibrated DRAM counter once per second isolates the decode phase from prefill:

| | t108/c2 | t114/c2 |
|---|---|---|
| s/token | 6.408 | 11.678 |
| DRAM per token | 110.3 GB | **127.2 GB, +15.3%** |
| bandwidth over the decode window | 16.71 GB/s | **10.60 GB/s, -37%** |

Subtracting the SSD wait, which is ~1.87 s/token in both arms, the DRAM rate is **24.3 GB/s
at t108 against 12.97 GB/s at t114**. Both are far below this box's 47.9 GB/s, because 32
unbound threads already halve it before the trunk size makes anything worse.

**So t114 moves 15% MORE bytes and moves them 47% SLOWER.**

## Conclusion, and what is still not known

The effect is fully characterised and reproducible, and five hypotheses are dead by
measurement: bandwidth saturation (it does LESS I/O), huge-page fallback (zero, cumulative),
slot contention (4x FEWER voluntary switches), memory reclaim (pgscan, pgsteal, pgmajfault and
PSI all exactly zero), and TLB reach (4.8x more misses costs 0.3%).

What the evidence supports: **at 32 unbound threads the memory subsystem is already running at
roughly half its capability, and raising resident footprint from 86% to 89% of RAM roughly
halves it again, while simultaneously raising DRAM traffic 15% — consistent with last-level
cache thrash as a larger working set competes across two CCDs.** With 16 bound threads the
effect vanishes completely: the same t114/c2 config runs at 5.63 s/token, the fastest of all.

**What is NOT established** is which of cache thrash, page placement across channels, or DRAM
row locality actually produces it. Separating those needs L3 miss and per-CCD counters, and it
is not worth the machine time right now: the practical consequence is already settled.

## Phase 5: thread count and binding, separated — and this CORRECTS the section above

The paragraphs above said "with 16 bound threads the effect vanishes" and drew a practical
rule from it. That rested on a SINGLE earlier run which changed thread count and binding
together — the exact two-variables-at-once mistake this whole investigation exists to
correct, committed inside the write-up of the correction. Seven cells, separated:

| trunk | threads | binding | s/token | involuntary ctx switches |
|---|---|---|---|---|
| 114 | 8 | free | 6.373 | 1,465 |
| 114 | **16** | free | **5.941** | 1,100 |
| 114 | 24 | free | 6.176 | 1,102 |
| 114 | **32** | **free** | **11.863** | **4,026** |
| 114 | 16 | bound | **5.647** | 571 |
| 114 | **32** | **bound** | **6.161** | 2,095 |
| **108** | 16 | free | **6.309** | 1,129 |

**Three corrections follow, and all three matter.**

**1. It is an INTERACTION, not a single cause.** 32 threads bound is fine (6.161). 16 threads
unbound is fine (5.941). Only 32-and-unbound collapses (11.863). Neither variable alone
explains anything; the earlier attribution to binding alone, and the interim reading that it
was thread count alone, are both wrong.

**2. The trunk pin is GOOD, and the headline rule above was backwards.** The control says it:
at 16 unbound threads, t114 runs at **5.941** against t108's **6.309** — the larger pin is
**5.8% FASTER**, exactly the direction the original reasoning expected, because more resident
trunk means less disk. The pin only inverts under the 32-unbound pathology. "`--trunk-gb 114`
is only safe with threads bound" is withdrawn; the honest rule is below.

**3. Binding is a separate, smaller, real effect.** At 16 threads it is worth 5.6% (5.941 ->
5.647), which matches the 4.9% measured independently earlier. It is not the thing that
rescues t114; it is its own modest win that ALSO happens to rescue the 32-thread case.

Involuntary context switches track the pathology exactly: 4,026 at 32-unbound against roughly
1,100 everywhere healthy, and 2,095 at 32-bound. That is the scheduler migrating threads it
should not, which fits the +15% DRAM traffic and -37% bandwidth measured in Phase 4: 32
threads on 16 physical cores, free to migrate across two CCDs, destroy locality on a 111 GB
working set. Binding removes the migration; fewer threads remove the pressure.

**The corrected practical rule:** the pathology is **32 unbound threads**, not the trunk size.
Use 16 threads bound (5.647, best measured). If thread count must be 32, bind them — that
alone recovers 11.863 -> 6.161. The trunk pin at 114 is a genuine ~5.8% win at sane thread
counts and should be kept.

**Still not established:** whether the underlying mechanism is CCD-crossing cache thrash, page
placement, or DRAM row locality. Phase 4 narrowed it to the memory subsystem and Phase 5
narrows the trigger to scheduler migration, but separating those three needs per-CCD and L3
counters. The practical answer no longer depends on it.


---

# Expert prefetch prediction: viable, bounded at ~17%, and second in line

Investigated before building anything, on the trace already captured
(`/root/k3trace/expert_trace.bin`, 33,469 requests, 15 clean decode tokens).

## The dependency, read from the code rather than assumed

`k3_moe` routes, down-projects, then calls `getmany` and immediately uses the result:

```
k3_router(idx, wt, xt, ...)                 /* needs layer L-1's output   */
k3_mmw(z, xt, w->down, ...)                 /* small                      */
w->src->getmany(w->src, w->layer, idx, nk)  /* BLOCKS                     */
for (j...) compute with idx[j]
```

**Within a token there is no lead time at all.** Expert I/O and compute are serial per layer:
37.4 ms of compute then 20.3 ms of I/O, 57.7 ms total. Getting lead time requires predicting
layer L's experts before layer L-1 finishes.

## Correcting my own earlier reading

The reuse-distance analysis said "0 of 33,469 requests had a reuse distance under 569" and
that was used to conclude expert caching is worthless. Correct, and the wrong question for a
prefetcher. **One decode token is 92 MoE layers x 16 = 1,472 requests, so the same layer's
experts recurring on the NEXT token sit at distance ~1,472 — which that analysis never
examined.** A 569-slot cache cannot reach it. A predictor does not need to.

Measured: the previous token's experts at the same layer predict **36.84%** of this token's,
and it is strongly layer-dependent — 12-17% in the early layers, 63-72% in the late ones.

## A methodology error caught in my own analysis

The static hot set first scored **45.91%**, computed from the same 15 tokens it was tested on.
Training on the test set. Split into train on 7, score on 8:

| hot set size | honest | oracle (leaked) |
|---|---|---|
| 16 | **27.19%** | 40.75% |
| 64 | 45.35% | 76.03% |
| 128 | **45.98%** | 97.50% |

At M=128 the leaked figure is 97.5% against an honest 45.98%: almost entirely memorisation.
The honest curve plateaus near 46% no matter how many bytes it is given.

## More accuracy can be SLOWER, because bandwidth is the constraint

The drives already run at 96% of what the array can deliver. A prefetch that fetches more
experts buys hit rate with bytes, and beyond ~29 experts per layer it no longer fits in the
37.4 ms compute shadow and becomes the critical path itself.

| predictor | hit | per layer | s/token | gain | drive busy |
|---|---|---|---|---|---|
| perfect, the ceiling | 100% | 37.4 ms | 3.441 | **+35.2%** | 54% |
| **union of last 2** | 48.6% | 47.8 ms | 4.402 | **+17.1%** | 91% |
| previous token | 36.8% | 50.2 ms | 4.622 | +13.0% | 66% |
| static hot-24 | 33.0% | 51.0 ms | 4.694 | +11.6% | 86% |
| union of last 4 | **57.6%** | 62.0 ms | 5.701 | **-7.3%** | 100% |
| static hot-64 | 45.3% | 92.4 ms | 8.502 | **-60.1%** | 100% |

**`union-4` predicts BETTER than `union-2` and runs SLOWER.** That is the whole shape of this
problem in one line: on a bandwidth-bound machine, a more accurate predictor that reads more
bytes is a worse predictor.

## Verdict

**Viable, worth about 17%, and it should not be built yet.**

- The ceiling is 35% and requires a perfect predictor; the best realistic candidate reaches
  **17.1%** at 1.63x the expert bytes, with the drives 91% busy.
- Multi-sequence batching is **2.21x already measured** in prefill, costs no hardware, and is
  the same primitive already committed. Prefetch is the smaller prize by an order of magnitude.
- **The two interact badly and it is worth saying so.** Batching amortises the trunk, which
  shrinks the compute per layer — and compute is exactly the shadow a prefetcher hides I/O
  behind. Batching makes the workload MORE I/O-bound, leaving prefetch LESS room, not more.
  In a batched world the answer to expert I/O is more drives, not cleverer prediction.

**What is NOT established:** 15 decode tokens from ONE prompt. Whether 36.84% token-to-token
persistence holds across prompts, lengths and domains is unmeasured, and every number above
rests on it. A longer, more varied trace is the cheap next step if this is ever picked up.

---

# The int8 trunk: measured, and the prediction held (2026-09-23)

The largest single term in a token is the trunk stream, 108.81 GB at 47.9 GB/s = 2.27 s, 43%.
Everything about a GPU rests on shrinking or relocating it. Rather than keep deriving, the
test was run on hardware already in hand.

**Recorded BEFORE the run, so it could be wrong on the record:** trunk 108.81 -> 54.47 GB
should take 2.27 -> 1.14 s, giving roughly 5.311 -> 4.2 s/token, ids would change, and the
risk was that `k3_matmul_q8` might not sustain the bf16 kernel's bytes/s and eat the gain.

Everything needed already existed: `K3_WI8` wired into `k3_mmw`, `k3_matmul_q8`, and
`tools/int8_trunk.py` (written for the speculative draft, but the trunk reader handles the
`I8R` dtype unchanged). Pack took ~7 minutes and produced **54.47 GB, 1437 tensors quantised**.

| | bf16 baseline | **int8 trunk** | change |
|---|---|---|---|
| steady-state decode | 5.311 s/tok | **4.133 s/tok** | **-22.2%** |
| prefill, 46 tokens | 152.77 s | **116.26 s** | -23.9% |
| whole run, 64 tokens | 487.4 s | **376.6 s** | -22.7% |
| **peak RSS** | 118.93 GB | **64.57 GB** | **-54.4 GB** |
| expert I/O | 164.2 s | 163.3 s | unchanged, as expected |
| token ids | `599d822ce3c3` | `00d538a039be` | diverge at token 21 |

**Predicted saving 1.13 s, measured 1.178 s — 4% apart.** The q8 kernel keeps up; the risk
did not materialise. This is the first prediction in the campaign that was recorded in
advance and then confirmed, rather than a budget assembled afterwards to fit.

## Quality

**Tokens 0-20 are identical to the exact model**; divergence begins at index 21 and the two
greedy sequences then follow different but equally coherent paths. The int8 continuation
reads normally. Position-wise "agreement" past the divergence point is meaningless and is
not quoted here. **A real quality claim needs perplexity or a benchmark, and we have neither.**

## What this changes

1. **The quantisation thesis is confirmed by measurement, not argument.** 22.2% for a format
   change, no new hardware, no new code.
2. **Storage is now the dominant term.** The budget is trunk 1.14 s (28%), expert SSD 1.87 s
   (45%), expert RAM->CPU 0.54 s (13%), other ~0.58 s (14%).
3. **RSS fell to 64.57 GB.** The whole model now fits a machine half the size of the AX102 —
   which reopens the hardware question from scratch.
4. **GPU sizing now has a measured basis rather than an estimate:** the int8 trunk is
   **54.47 GB**, so it needs 64 GB+ of VRAM to be fully resident; a Q4 trunk at ~27 GB would
   fit a 32 GB card and is marginal on 24 GB.
5. The batched GEMM is inactive on this path — `k3_mmw_batch` falls back to serial for
   non-bf16 — so the 22.2% is achieved WITHOUT it.

**The cost is the oracle.** Every comparison before this one rested on byte-identical output.
This one does not, and cannot. That is a deliberate, recorded change of footing, not a drift.

---

# The MXFP4 trunk: built 2026-09-23

## A wrong proposal, caught by reading the file

I proposed "run the Q4 trunk with `tools/qdq_trunk.py`". **That tool cannot do it.** Its own
docstring: it writes the dequantised result back as ORDINARY bf16 in an identical
container, same offsets, same dtypes. It is a *quality* probe; the output is still 108.81 GB
and runs at exactly bf16 speed. I had gone from the filename in my notes rather than the file.
Thirty lines of reading caught it before a pack and a run were spent measuring nothing.

Nor is there a 4-bit weight type: `enum { K3_WF32 = 0, K3_WBF16 = 1, K3_WI8 = 2 }`, and
`k3_wsz()` returns 2 for bf16 else 4, so the helpers did not model sub-byte weights at all.
**Q4 required new engine code**, not a tool invocation. Direction to build it was given
explicitly after the cost was restated.

## What made it small

- `k3_matmul_mxfp4` **already existed and was proven** — the routed experts use it, measured
  at 147.3 GFLOP/s. No new kernel.
- The trunk manifest already carries **per-tensor dtype**; the int8 container had already
  demonstrated adding one end to end (`I8R`). That was the worked example to copy.
- `docs/notes/int8-draft-container.md` records the three-part recipe (format, kernel,
  dispatch) from when `I8R` was added. Following it removed most of the guesswork.

## The one real design problem

`k3_mmw` passes a SINGLE `W` pointer; `k3_matmul_mxfp4` needs `packed` and `scales`
separately. `I8R` solved its version by inlining the scale per row, but the mxfp4 kernel
indexes two contiguous 2D arrays and would have needed a variant to match that.

Solution taken: lay one matrix out as **`[packed: rows*(in/2)][scales: rows*(in/32)]`**, one
allocation. Both sub-arrays are then contiguous with exactly the strides the kernel already
expects, so `scales = W + out*(in/2)` and **the kernel is used unchanged**.

Second problem: the binder recovers `rows` for `I8R` from `nb - take`, but for MXFP4
`nb == take*17/32` for *any* shape, so rows is **not recoverable**. It is not needed:
verified that all 1437 BF16 2D tensors have `cols % 32 == 0` (zero exceptions), and when 32
divides C, `r*(C/32) + c/32 == (r*C + c)/32` exactly. So the row-major tensor and one flat
row assign identical scales, and the dequant passes `rows=1`. The precondition is what makes
this legitimate, and the packer enforces it by only converting tensors that satisfy it.

`res_proj` is `[1, 7168]` — 2D and divisible by 32, and it is a `reqw` (fp32-wanted) tensor,
so the dequant path is genuinely exercised, not dead code.

## The scale rule: my "fix" was measured and rejected

The OCP rule `e = floor(log2(amax)) - 2` puts a group's max at `4m ∈ [4,8)`, so any group
with mantissa above 1.5 **clamps its own largest weight to 6** — 30-44% of groups on
realistic data. I proposed raising the exponent for those groups to avoid the clamp.
Measured, that is WORSE:

| data | spec rule | "noclamp" fix | best-of-two | groups clamped |
|---|---|---|---|---|
| gaussian | **11.51%** | 11.57% | 11.16% | 30% |
| heavy-tailed t(4) | **13.19%** | 13.90% | 12.72% | 42% |
| laplace | **12.65%** | 13.19% | 12.27% | 44% |

Halving the resolution of all 32 elements costs more than clamping one. **Spec rule kept.**
Best-of-two is ~3% relatively better and was not taken: it doubles encode cost for a
difference unlikely to move token quality. Revisit only if quality is the blocker.

**Relative L2 is ~11.5%, about 30x the int8 path's measured 0.37%.** That is the real risk in
this experiment, and it is a property of 4-bit, not of this implementation.

## Verification before spending 100 GB

Against `tests/fixtures/mxfp4.json`, which holds REAL released-checkpoint bytes:

1. **decode convention — PASS**, 229,376/229,376 elements exact (nibble order and E8M0 both
   confirmed; the fixture also records the swapped-nibble trap).
2. **encode -> decode round trip — PASS**, 229,376/229,376 exact.
3. byte-identical to the checkpoint: no. 1,933 of 7,168 scale bytes are exactly 1 lower than
   the checkpoint's own, and every value still round-trips exactly, so the difference is a
   finer scale choice, not a defect.

New `tests/unit/test_mxfp4w.c` checks the part that would fail silently — that `k3_mmw`
derives the scales pointer where the packer put it — by comparing the dispatch against the
kernel called directly with explicit pointers, requiring **bitwise** equality. Five shapes
including real trunk widths, all PASS, all outputs nonzero so the comparison is not vacuous.
Not yet wired into the Makefile (see the note on the stale local checkout below).

Regression state after the change: **build clean, zero warnings**; `make test` green; the
full-model oracle still reports ENGINE MATCHES THE REFERENCE EXACTLY; kernel hashes
unchanged at `83c8504a4cb3fac6` / `a231061237b5579d`.

## Recorded BEFORE the measurement

trunk 108.81 -> ~28.9 GB (17/64 of bf16); trunk time 2.27 -> ~0.60 s; total 4.133 -> **~3.6
s/token**, about 32% below the bf16 baseline. Token ids will differ and should diverge
EARLIER than int8's token 21. The risk that matters is not speed but whether 11.5% rel-L2
leaves a usable model.

## Measured

The container matched the prediction exactly: **28.94 GB** against ~28.9 predicted, 1437
tensors converted, 1018 copied through, and **zero** MXF4 tensors with a wrong byte count.

| | bf16 | int8 | **MXFP4** |
|---|---|---|---|
| steady-state decode | 5.311 | 4.133 | **3.965 s/tok** |
| vs bf16 | — | -22.2% | **-25.3%** |
| **vs int8** | — | — | **-4.1%** |
| prefill, 46 tokens | 152.77 s | 116.26 s | 108.07 s |
| whole run, 64 tokens | 487.4 s | 376.6 s | 357.8 s |
| **peak RSS** | 118.93 GB | 64.57 GB | **39.13 GB** |
| trunk packed | 108.81 GB | 54.47 GB | 28.94 GB |
| **diverges from bf16 at token** | — exact | 21 | **3** |

**The speed prediction was WRONG: 3.6 predicted, 3.965 measured, 10% off.** Halving the
trunk a second time returned only 0.168 s where the bandwidth model said 0.53 s.

**Why, and it is the finding of this cycle.** The trunk is pinned, so per-token cost is a
DRAM read. int8 moves 54.47 GB at the machine's measured 47.9 GB/s. MXFP4 should move 28.94
GB in 0.604 s; it effectively takes ~0.97 s, about **30 GB/s** [D, derived from the measured
0.168 s saving, not observed directly]. Unpacking nibbles and applying an E8M0 exponent per
32 elements costs enough that the kernel is no longer purely bandwidth bound, so it gives
back most of the bandwidth it saves. This is the same risk named before the int8 run, where
it did not materialise; at 4 bits it does.

**The quality cost is severe.** Divergence at token 3 against int8's token 21. The output
stays fluent and topical — it goes off into PostgreSQL buffer pools rather than Linux LRU —
but it is plainly a different model, consistent with the 11.5% rel-L2 measured up front.

## Verdict

**int8 is the better operating point for throughput ON THIS CPU.** MXFP4 buys 4.1% for a
large quality loss.

**SCOPE CORRECTION (creator, 2026-09-23).** An earlier draft of this section said "further
quantisation of the trunk is a dead end for speed". That is wrong as written. It is a
property of THIS CPU, not of the format. The reason MXFP4 disappointed is that nibble unpack
plus an E8M0 exponent per 32 elements is expensive relative to a CPU's memory bandwidth. A
GPU inverts that ratio: unpack is close to free across thousands of ALUs while VRAM
bandwidth is the binding constraint, which is exactly the regime where 4-bit pays. **The
compounding on a GPU can look completely different, and nothing here measures it.** The
honest statement is: on a 7950X3D with 47.9 GB/s of DRAM, shrinking the trunk below int8
returns little. Whether that holds anywhere else is UNKNOWN and cannot be known from this
machine.

**MXFP4 is the right choice if RAM is the binding constraint.** At **39.13 GB peak RSS** the
whole model runs in under a third of what bf16 needed.

**Where the time now sits, and it is not the trunk.** Expert streaming was 157.8 s of the
357.8 s run and is essentially unchanged across all three formats, because experts were never
quantised here — they already ship MXFP4. Storage is the term to attack, and the PCIe
evidence in `k3-results/hetzner-ticket/pcie-evidence.txt` says `nvme1n1` is negotiating
**x2 instead of x4** (boot log: 31.506 Gb/s available against 63.012 capable; 28.8 ms latency
against nvme0's 13.9 for comparable volume). Chasing the trunk further while the dominant
term runs on half its lanes is the wrong order of work.

---

# llama.cpp, read from source at last (2026-09-23)

**What was wrong before this.** Every earlier statement in this file about llama.cpp came
from reading ABOUT it on the web — including the "26.90 tok/s at `--n-cpu-moe 25` on an RTX
3090" figure, which was passed on as fact. The repository had never been cloned. That is a
Context failure by this framework's own rules: a published page is an artifact to evaluate,
not the system. **Treat every llama.cpp number in the earlier sections as unverified.**

Now cloned at `c:\personal\oss\llama.cpp`, HEAD `e6ab7c1`. From source:

- `common/common.h:1131` — `LLM_FFN_EXPS_REGEX = "\.ffn_(up|down|gate|gate_up)_(ch|)exps"`.
- `-cmoe` (`common/arg.cpp:2756`) pushes ONE override sending every matching tensor to
  `ggml_backend_cpu_buffer_type()`. `-ncmoe N` (`llm_add_n_cpu_ffn_overrides`) builds
  per-layer patterns `blk\.{i}\.ffn_..._exps` for i in 0..N-1, so it is the FIRST N layers.
- Both are sugar over the general `-ot <regex>=<buffer type>`.
- Resolution happens at LOAD time in `llama-model-loader.cpp:1235`, selecting a
  `ggml_backend_buffer_type` per tensor. It is placement, not runtime scheduling.
- Overriding to CPU while mmap is on logs a warning recommending `--load-mode none`.

**The detail that matters and that the web summaries did not give.** Shared experts are named
`blk.%d.ffn_{gate,up,down}_shexp` (`llama-arch.cpp:466-468`) and **do not match the regex**,
so they stay on the GPU. Routed experts are `GGML_OP_MUL_MAT_ID`; shared experts are plain
`GGML_OP_MUL_MAT`.

So the rule llama.cpp actually implements is: **everything dense and always-active on the
fast device, only the sparse indexed streamed experts on the slow one.**

**Why this matters for K3.** That is precisely K3's existing trunk/expert boundary. The trunk
is attention (66.5%) + shared experts (22.3%) + routed latent proj (8.7%) + dense + router —
all dense and always-active. The routed experts are the 1.45 TB streamed part. **K3 already
draws the same line a widely used engine draws deliberately**, which is evidence the
trunk-on-GPU design is sound in principle. It says nothing about the throughput it would
reach.

---

# What is NOT known, and cannot be known from this machine

Stated plainly because the measured sections above are easy to over-read:

- **No GPU has ever been involved.** Every number in this file is a 7950X3D with 124 GiB of
  DRAM at 47.9 GB/s and two NVMe drives, one of them at x2. The GPU thesis is UNTESTED.
- **CPU behaviour does not transfer.** The MXFP4 result is the clearest example: the format
  lost on compute here and would plausibly win where bandwidth dominates. Any extrapolation
  from these numbers to a GPU is a hypothesis, not a finding.
- **The llama.cpp throughput figures quoted earlier were never reproduced**, on any machine.
- **Quality is one prompt, 64 tokens, greedy, no perplexity and no benchmark.** "Diverges at
  token 3" is measured; "MXFP4 is unusable" is not established.
- **One payload** (`p2_medium`) for the int8 and MXFP4 runs.
- The ~30 GB/s effective MXFP4 trunk rate is DERIVED from the 0.168 s saving, not observed.
  `bench_kernels` timing `k3_matmul_q8` against `k3_matmul_mxfp4` at trunk shapes would
  settle it and has not been run.

---

# NEW DIRECTION (creator, 2026-09-23): pursue the heterogeneous picture

## How this is to be framed — corrected by the creator

An earlier draft of this section set the three approaches against each other, said the
efficiency discipline of games is "absent" from AI, and that the frontier "looked only at the
GPU". **That framing is wrong and is not what Clover does.** The creator's correction:

> Clover is not against any system. Each system is unique and the solutions can vary
> depending on their reality, so we should not compare ours to theirs. The underlying
> philosophy is common: all are improving their systems.

So: no league table, no winner. Each of these is a sound answer to the reality its builders
actually faced, and ours is a different reality, not a better one.

- **Frontier labs** build for fleets of accelerators and train at a scale where that is the
  sensible shape.
- **Fareed Khan's `kimi-k3-in-c`** answers "what if the machine is one CPU with 8 GB" — and
  answers it: 2.78T parameters, 1.56 TB checkpoint, 8.24 GB peak RSS, no BLAS, no framework,
  no GPU.
- **llama.cpp** answers "what if the machine is a mix" — placement per tensor across whatever
  devices are present.
- **The games industry** solved a related problem decades ago in its own way: fit a world far
  larger than the hardware, by streaming, level of detail, compression, cache locality and
  fixed budgets. That prior work is worth respecting and learning from, not claiming.

## Our path, stated accurately

**We started from our own system.** We explored Fareed Khan's engine and it gave us real
experience — the kind that only comes from running the thing. We are now exploring the next
step in order to move forward rather than stay where we are. That is the whole story, and it
should not be told as though his work is the origin of the direction.

The heterogeneous picture is what we want to pursue, informed by how the games industry
solved fitting large things into fixed hardware, and referring to the work in llama.cpp.

## What the experience actually taught

Evidence from this campaign, much of it from being wrong:

- Quantising the trunk paid **exactly as modelled** down to int8 (-22.2%, prediction held to
  4%) and then **stopped paying** at 4 bits (-4.1%, with a large quality cost), because this
  CPU ran out of compute rather than bandwidth. A device property, not a format property.
- The same MXFP4 pack meets **native FP4 tensor cores** on a GPU, where the ratio inverts.
  One machine cannot answer what the other answers easily.
- **No single device suits the whole model.** The 28.94 GB trunk wants VRAM; 1.45 TB of routed
  experts can never be resident anywhere and must stream. That is heterogeneous by
  construction — and it is the same boundary K3's own trunk/expert split already draws.

## The machine being added: GPU-Server GEX45-1

GEX45-1, HEL1, EUR 214.00/mo + EUR 209.00 setup. RTX PRO 4000 Blackwell SFF with **24 GB
GDDR7** and 5th-gen **native FP4** tensor cores; i5-13500 (6P+8E, 20 threads); 64 GB DDR4;
**2 x 920 GB NVMe** (confirmed by the creator; the published base listing shows a smaller
disk option, so the ordered configuration is the one that counts).

Two limits, stated before it arrives:

1. **The next server is intended to hold the full checkpoint across two disks.** Split (not
   mirrored — settled direction) is the storage experiment. The exact placement and read
   scheduling still need to be measured on the box. The 1.45 TB routed-expert pool remains
   storage-backed even though it is already MXFP4.
2. **The MXFP4 trunk does not fit the card.** 28.94 GB against 24 GB VRAM, short by ~5 GB.
   Partial placement is the answer, which is what per-tensor placement exists to do.

So GEX45-1 is not a smaller version of the same experiment. The question it exists to answer
is **what actually needs to be on the GPU for each token** — which parts of the trunk, which
experts, held where, with reads for the next token overlapping computation of the current
one. Sharding across the two disks is part of the experiment rather than a workaround for
capacity. **This is not an architecture claim; it is the next thing to measure.**

### Capacity, with the confirmed disks

2 x 920 GB split gives **1.84 TB raw**, roughly **1.7 TB after filesystem and OS**. The
checkpoint is 1,560,936,091,448 bytes, so it fits with about 140 GB to spare. That headroom
decides which packed trunk can go with it:

| trunk | size | fits alongside the checkpoint? |
|---|---|---|
| MXFP4 | 28.94 GB | yes, comfortably — and the card has native FP4 |
| int8 | 54.47 GB | yes |
| bf16 | 108.81 GB | only just; not worth the risk |

**Plan to ship the MXFP4 trunk to that box, not the bf16 one.** Verify the real usable figure
with `df` on arrival rather than trusting this arithmetic.

## Storage direction, settled and previously mishandled

**Split, not mirror.** This was directed earlier and I wrongly recorded it as awaiting
go-ahead; it is settled. On the AX102 it matters more than capacity: `nvme1n1` negotiates
**x2 instead of x4**, so a mirror spreads reads evenly across a healthy drive and a crippled
one. Independent filesystems would let placement be chosen. **Open and narrow: whether
"split" means RAID0 striping or two independent filesystems** — capacity is the same, but a
stripe parallelises automatically while independent drives leave placement to us.

## What this direction requires next

**Settled by the creator, 2026-09-23:**

- **The engine changes are intended for upstream.** The four commits, the int8 container work
  and the MXFP4 trunk path are to be offered to
  [`FareedKhan-dev/kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), not kept
  local. Nothing has been pushed anywhere yet.
- **We are not done.**

Outstanding work:

- Prepare the engine changes for upstream contribution: commit them, check them against the
  project's CONTRIBUTING rules, and separate what is genuinely useful to the project from
  what was only scaffolding for our measurements.
- Verify the GEX45-1 specification on the machine rather than from the product page.
- Reproduce a llama.cpp placement figure ourselves before quoting one ever again.
- Run `bench_kernels` on `k3_matmul_q8` against `k3_matmul_mxfp4` at trunk shapes, to turn
  the derived ~30 GB/s into a measured number.

### Upstream scope, read from the project's own documents (2026-09-23)

`CONTRIBUTING.md` says to read `docs/ROADMAP.md` first because it lists what is deliberately
not planned. Doing so changed the contribution plan before any of it was prepared.

**Explicitly NOT planned upstream, quoted:**

> **A precision dial for the trunk.** The trunk is streamed losslessly rather than
> quantised, and that is a design decision, not an omission. Post-hoc int4 measures ~17%
> mean relative weight error on K3 attention tensors against ~1% for int8; streaming costs
> time, which more memory buys back, while rounding costs accuracy, which nothing buys back.

> **GPU support.** Out of scope for this project.

So **the MXFP4 trunk path would be declined**, and the author had already measured the same
wall we hit before we started. Our 11.5% relative L2 and his ~17% mean relative weight error
are different metrics reaching the same conclusion independently — which is corroboration of
the finding, not a reason to submit it. The int8 and MXFP4 work stays ours; it serves the
heterogeneous direction, which is out of scope there by design. That is a clean separation,
not a rejection.

**What IS open upstream and already matches work we have done:**

- **Roadmap #2, re-run the published campaign under the replicating harness.** "The measured
  noise floor is 33%, and almost every published figure is a single sample." The harness
  exists (`benchmarks/memory-ladder.sh`, `benchmarks/split-sweep.sh`, 3 repeats, mean/sd/
  spread); what remains is running 12 ladder rungs and 12 splits and replacing the
  single-sample tables in `docs/data/`. We have the hardware for exactly this.
- **Roadmap #3, thread scaling.** "`OMP_NUM_THREADS` has never been swept on this engine."
  We swept it: 8/16/24/32 threads, bound and free, seven cells, and found 32-and-unbound
  collapses to 11.863 s/token against 5.647 at 16 bound. This is a direct answer to an open
  roadmap item.
- **Two engine defects found during this campaign**: `--trunk-gb auto` cannot start because
  auto reserves 2 GB + 2% while the admission guard demands 5%; and the refusal message
  prints a negative shortfall. Both are small, in scope, and independent of our direction.
- The four existing perf commits (expert read chunking, batched bf16 matmul, KDA sweep read
  reduction, the matmul bench).

**Blocking condition on all perf submissions:** CONTRIBUTING requires **at least three runs
per arm, all reported**, because run-to-run spread is 33%. **Every performance figure we
have is a single run per arm.** They are not submittable as they stand. The project also
prefers counts over seconds — bytes read per token, evictions, pinned layers — which our
trunk sizes already are and which are the stronger claim anyway.

### Replicated A/B, three runs per arm, interleaved (2026-09-23)

`ac1584a` (before the four commits) against `9977044` (all four), **same tuned config on both
arms** so this isolates the code. Interleaved BASE/HEAD/BASE/HEAD/... so drift cannot land on
one arm. Two separate binaries, md5s confirmed different. No profiler, no sampler, nothing
else on the machine; load average tracked `OMP_NUM_THREADS` exactly.

| run | decode s/tok | prefill s | total s | peak RSS GB |
|---|---|---|---|---|
| BASE-r1 | 5.631 | 184.25 | 539.0 | 118.92 |
| BASE-r2 | 5.627 | 168.64 | 523.1 | 118.93 |
| BASE-r3 | 5.628 | 168.53 | 523.1 | 118.93 |
| HEAD-r1 | 5.304 | 152.96 | 487.1 | 118.93 |
| HEAD-r2 | 5.305 | 151.29 | 485.5 | 118.93 |
| HEAD-r3 | 5.307 | 151.32 | 485.7 | 118.93 |

| metric | BASE mean (sd, spread) | HEAD mean (sd, spread) | gain |
|---|---|---|---|
| decode | 5.629 (0.002, 0.07%) | 5.306 (0.001, 0.05%) | **+5.74%** |
| prefill | 173.81 (9.04, 9.04%) | 151.86 (0.96, 1.10%) | **+12.63%** |
| total | 528.4 (9.18, 3.01%) | 486.1 (0.87, 0.33%) | **+8.01%** |

Every run is reported, not just the means, and every one parsed at exactly 63 decode steps.

**Three things the replication showed that a single sample could not:**

1. **Steady-state decode is one of the most reproducible measurements on this machine** —
   sd of 0.002 s, spread 0.05-0.07%. The project's CONTRIBUTING states a 33% run-to-run
   noise floor and requires three runs because of it. That figure appears to describe
   prefill and wall clock, not steady-state decode. A 5.74% decode gain against 0.07% noise
   is roughly an 80x margin.
2. **The changes reduce variance as well as time.** Prefill spread falls 9.04% -> 1.10%,
   total 3.01% -> 0.33%. Plausibly the concurrent chunked reads making I/O more predictable,
   but that is an inference, not a measurement.
3. **A first-run warm-up effect exists and inflates the BASE prefill spread.** BASE-r1 was
   the very first run of the whole campaign at 184.25 s; r2 and r3 were 168.64 and 168.53.
   Excluding each arm's first run, prefill is 168.59 -> 151.31, a **10.25%** gain with
   spreads of 0.07% and 0.02%. Reported both ways rather than picking the flattering one.

**What this does NOT establish:**

- **One configuration only** (`--trunk-gb 114 --cache-gb 2`, 16 threads bound). A change can
  help at one memory budget and do nothing at another; `benchmarks/memory-ladder.sh` exists
  for exactly this and has not been used.
- **The bundle, not the commits.** Three functional changes were tested together. If one
  carries the gain and another does nothing, this cannot tell them apart, and the project
  would be taking on maintenance for whichever does nothing. The batched matmul should by
  design affect prefill only, since decode is single-token and falls back to the serial
  path — the large prefill gain is consistent with that but does not prove it.
- One payload, one prompt length.

### The objection that reshaped the contribution (creator, 2026-09-23)

> "the maintainer can dismiss this because the server setup could have accounted for this
> and not the code"

Correct, and stronger than stated: **one of this machine's two NVMe drives negotiates PCIe
x2 instead of x4.** A change that improves read concurrency could be helping precisely
because our storage is degraded, and do nothing on a healthy machine. A full-model wall-clock
number from this box is dismissible as a property of the box.

The project's own CONTRIBUTING already says the answer: "Where you can, measure **counts
instead of seconds** ... They are immune to scheduling noise and make a much stronger claim."

Sorting the three changes by how exposed each is:

| change | evidence available | exposed to the objection? |
|---|---|---|
| batched bf16 matmul | microbenchmark, no checkpoint, no storage | **No.** Reproducible anywhere. |
| KDA sweep reads | a count: q/k/v read once per sweep, not once per position | **No**, but not yet demonstrated by measurement. |
| expert read chunking | effective device throughput | **Yes.** The mechanism IS queue depth. |

**`bench_batch` on 7168 x 12288 bf16, 16 threads bound, no weights required:**

| T | serial x T | batched | speedup |
|---|---|---|---|
| 1 | 0.0061 | 0.0043 | 1.41x |
| 2 | 0.0076 | 0.0057 | 1.35x |
| 4 | 0.0149 | 0.0073 | 2.03x |
| 8 | 0.0279 | 0.0129 | 2.16x |
| 16 | 0.0562 | 0.0238 | **2.36x** |

Consistent with the 2.39x recorded earlier for the shipping kernel, and with `test_batch`
proving the output is bit-identical this change stands without trusting anything about our
hardware.

**Defect found in our own commit while doing this:** `9977044` added
`tests/unit/bench_batch.c` and **no Makefile rule**, so `make bin/bench_batch` fails. We
shipped a file the build cannot build. Must be fixed before offering it.

**Consequence for how to contribute.** Lead with what the maintainer can verify himself;
present the chunking change with the storage disclosure attached rather than buried, and
state plainly that its benefit is a property of the device queue. Do not lead with a
full-model timing from a machine with a known degraded link.

### The `auto` defect, root cause found 2026-09-23

`--trunk-gb auto` is the setting the CLI itself labels **"Recommended"** in its preset list.
It cannot start on this machine, and reading `src/cli/k3_run.c` shows why.

- auto reserves `2 GB + 2%` of available, about 4.6 GB here.
- With full residency reachable it sets `trunk_gb = 111.0` and gives the entire remainder to
  the expert cache, so the plan consumes ~100% of what is left after its own reserve.
- The admission guard then requires the total to be under **95%** of available
  (`if (need_b > have * 0.95)`).

With 132.10 GB available: reserve 4.64, usable 127.46, trunk 111.0, cache 16.46, need 127.46
against a ceiling of 125.50. **Refused.** auto's reserve (2% + 2 GB) is smaller than the
guard's margin (5%), so on any machine with enough RAM for full residency the recommended
setting refuses to start. The refusal message then prints the shortfall as a negative number.

This is worth more to the project than a few percent of throughput: it blocks the documented
happy path, it is deterministic, and the fix is to reconcile the two margins rather than to
tune anything.

### Baseline choice for an end-to-end claim

The engine's literal defaults are `trunk_gb = 16.0, cache_gb = 64.0` (k3_run.c:693). Our
earlier "base" of t108/c10 was our own choice and was never the shipped default. Comparing
our tuned setup against the literal defaults would be a strawman, because nobody with 124 GB
would run 16 GB of trunk. The honest baseline is the project's own best shipped option:
`--preset server` (110/13, described as "Fastest") with **default threads**, since the CLI
never calls `omp_set_num_threads` and therefore gets 32 unbound threads on this 2-CCD part.

## A hazard worth not repeating

The local checkout at `c:\personal\oss\kimi-k3-in-c` is **BEHIND** the machine's repo — the
server's tracked Makefile lists `test_batch`, the local one does not. Files are edited
locally and scp'd, so sending a stale file silently reverts committed work. After this
transfer I diffed every sent file against server HEAD and confirmed only the intended lines
changed. **Diff before sending, or fetch first.** This is also why the new test is not wired
into the Makefile: that would mean sending a stale Makefile.























