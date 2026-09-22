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

## Plain summary, read this first (2026-09-22)

Started at **9.40 s per generated token** on a 16-core, 124 GiB, no-GPU box streaming a
1.56 TB checkpoint off NVMe. Four things were found, in this order:

| # | finding | measured effect |
|---|---------|-----------------|
| 1 | one NVMe negotiated PCIe **x2 instead of x4**; Hetzner found the cables loose and reseated them | **18%** (601.8 s -> 493.6 s, identical config) |
| 2 | **16 threads beats 32.** The box has 16 physical cores; the extra SMT threads add barrier participants, not throughput | **8.1%** (592.0 s -> 543.8 s) |
| 3 | **prefix reuse** via `--save-state` / `--load-state` | **2.06x** on turn two (701.5 s -> 340.4 s) |
| 4 | **~64% of a short run is process startup**, loading 98 GB of weights before any work begins | ~380 s of a 592 s run |

In human terms, roughly: a short question moves from ~19 to ~15 minutes, a 300-word
explanation from ~46 to ~38, a 2000-token file review from ~92 to ~76, and prefix reuse
takes a repeat turn further again. Still not interactive - a hosted model is ~500x faster -
but it is frontier-scale on hardware you own with nothing leaving the box.

**Caveat that matters: 18% and 8.1% were measured SEPARATELY, at different settings. The
combination has never been run.** Any "now" figure above is arithmetic, not observation.

### What to look for next, in value order

1. **Stop reloading.** 64% of a short run is startup a persistent process removes outright.
   Largest single win available, needs no hardware.
2. **Measure the combination** - PCIe fixed, 16 threads, best memory split, one run.
3. **Batching** - never tested here, and the only lever that attacks bytes per token directly.
4. **Context capacity** - 2.37 MB per position decides whether a real codebase fits at all.

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
