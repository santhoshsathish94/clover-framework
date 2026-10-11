# Context — one layer, properly

A context file for the distributed hosting work. Newest cycle first. Written during the
work, not after it. Offered, not imposed: this is my account of what happened and what
it does and does not establish.

---

## Cycle: join the kernel number to a real layer, then port it (2026-10-11)

### Intended outcome

The previous cycle left its central claim unjoined: 23.8x was an isolated kernel, and
the layer had only ever run at five positions. Close that, then decide from the result
whether the other MLA pods should have the change.

### What the System showed

`pod-context.c` drives a real pod through `transformer_process` with its 15.72 GB of
experts pinned by `mlock`, so expert arrival is identical in every arm and only the
attention walk moves. Output bit-identical in every pair:

| positions | original | head-major | layer | attention component |
|---|---|---|---|---|
| 64 | 31.265 ms/pos | 30.864 | 1.01x | negligible |
| 1,024 | 40.035 | 32.106 | 1.25x | 9.1 -> 1.2 ms |
| 4,096 | **71.546** | **35.245** | **2.03x** | 40.6 -> 4.3 ms, 9.4x |

Experts cost about 30.9 ms per position whatever the length, so below roughly 1k
positions the change is noise — the cache still fits in L3. At 4,096 the old layout
spends 57% of the layer in attention and the new one 12%. That justified the port.

**Ported to all pods and verified.** `port-head-major-mla.mjs` follows the house pattern:
unique-anchor assert, idempotent, layers 2..92. The block exists in every pod but is
only compiled where `TRANSFORMER_MLA` is 1, so 24 layers change behaviour and 66 keep a
dead copy consistent rather than leaving a latent defect. All 24 MLA layers verify
against the reference for france and japan, 24 of 24, matching the pre-port baseline.

### The mistake, which was the same one twice

I shipped 90 stale sources to the host and overwrote the deployed tree.

Last cycle I found the repository was behind what runs, pulled `transformer-3.c` from
the host, and wrote in this file: *"Read the deployed artefact first, not the repository
copy."* Then I ran the port against the other 91 pods **without pulling them**, because I
had quietly assumed that fixing one file had fixed the problem. It had not: those pods
still included `root.h` instead of `common/live-root.h`, a gap of **30,159 insertions**,
not the 332 I had measured on a single file.

The build caught it — 23 pods failed with `omp_get_max_threads` implicitly declared,
which is exactly the set of MLA pods, because only they compile the block. The host was
restored from a backup taken before the extraction, rebuilt, and re-verified. Nothing
was lost, because the script took that backup before writing. That was luck of habit
rather than design, and the habit is worth keeping.

The correction: a sample is not a survey. When one artefact is found to be stale, the
question is how many others are, and that is a cheap thing to check.

### What was ruled out

**The layer 46 route mismatch is not mine.** The full campaign (`all`, layers 2..92)
aborts at layer 46 on `!memcmp(routes, captured_routes[position])`. Controlled test:
identical runs differing only in transformer-3's stage object, original and head-major
both reach VERIFIED 44, last layer 45, same assertion. So it predates this work. The
historical `remaining/validation.log` holds 182 VERIFIED lines, a full pass over 91
layers times two prompts, so layer 46's routing changed at some point between that
campaign and now — most likely in one of the ports, none of which re-ran the campaign.
Selecting only MLA layers steps past it, because the harness checks only selected layers.

### What is still unknown

- Why layer 46's routes no longer match the stored reference. Not investigated. Until it
  is, the full campaign cannot pass and only subsets can be verified.
- Everything in the previous cycle's unknowns still stands: no end-to-end long context,
  the latent K/V form unread, expert compute unprofiled at stage granularity, the DIMMs
  at 3600 against a 4800 rating.
- The 4,096-position result is one layer on one machine with experts pinned. A pod under
  real traffic shares the box.

---

## Cycle: make layer 3 use the machine (2026-10-10)

### Intended outcome

One layer — not the fleet, not the model — correct and actually using the CPU cores, the
RAM and the disk it has. Then a stage map from input through to layer 92.

### What was known going in

The standalone engine's MLA K/V cache had been changed from position-major to head-major
earlier the same day, measured at 1.68x at 1M positions on an isolated kernel. The open
question was whether the distributed layers carried the same defect.

### What was tried, and what the System showed

**The repo was not what runs.** The local `transformer-3.c` included `root.h`; the
deployed one on k3 included `live-root.h`, `stage.h` and `trace.h`. A diff put the repo
**332 insertions and 41 deletions behind** the deployed source. `common/` is untracked in
git and was never committed; the port scripts write in place but their output never
reached a commit. I had already read the stale file and begun reasoning about it. Pulled
the deployed copy down and worked from that.

**The defect was there, and worse.** `transformer_mla` walked every head across every
position at a 98,560-byte stride — K, V and the positional tail interleaved per position.
Double the standalone's stride, because here all three share one block.

**Two things were wrong, not one.** The head loop also had no OpenMP pragma: 96
independent heads on a single core while fifteen sat idle.

Measured with `tansformers/mla-stride.c`, which isolates the attention step at real
sizes. All arms produce identical checksums at every size.

At 1,048,576 positions, 103.35 GB for one layer:

| arm | time | GB/s |
|---|---|---|
| position-major serial — what the layer did | 62.40 s | 1.66 |
| position-major parallel | 5.16 s | 20.03 |
| head-major serial | 12.51 s | 8.26 |
| head-major parallel | **2.62 s** | 39.45 |

23.8x together. Layout alone 4.99x, threads alone 12.1x. The result sits at 39.45 GB/s
against a measured 42.8 GB/s RAM ceiling, so it is bus-bound.

**Thread count, measured on the real layer through the reference harness:** 1 thread
2.49 s, 4 threads 0.73 s, 8 threads 0.44 s, 16 threads 0.30 s, 32 threads 0.31 s. Every
run verified. Sixteen is the optimum for a layer alone; SMT does not help.

**Correctness.** Three gates, all passed:
- `VERIFIED` for france and japan at all five positions against the independent
  reference, for both the layout change and the parallel head loop.
- Bit-identical output over a 40-position stream against the unmodified binary. This
  matters because the cache starts at capacity 8 and the reference prompts are 5
  positions, so **the reference never grows the cache and never executes the re-layout
  code**. Forty positions crosses 8 to 16 to 32 to 64.
- `CONTROLS 3 PASS` under AddressSanitizer and UndefinedBehaviorSanitizer.

Installed as `transformer-3.c` on k3 and rebuilt as `pipeline-stage.so`; re-verified at
16 threads. The original is kept at `transformer-3.orig.c`.

### What was ruled out

**The disk is not the bottleneck, and the current I/O path is the best of three.** I
believed expert reads were running at roughly 380 MB/s against a device measured at
13.51 GB/s. That number was wrong and so was the reasoning: I divided whole-layer wall
time by expert bytes and called the quotient an I/O rate. Measured properly, with the
page cache dropped before each arm and disjoint expert ranges:

| request shape | pass 1 | pass 2 |
|---|---|---|
| mmap + `posix_madvise(WILLNEED)` — current | 8.58 GB/s | 8.06 GB/s |
| one `pread` per expert across threads | 7.02 | 7.18 |
| chunked `pread`, more requests in flight | 7.02 | 6.86 |

281 MB of experts arrives in 33 ms. The existing path beats both alternatives. There is
no gain here, and the plan to port the standalone engine's dedicated pread readers into
the distributed layer is withdrawn on this hardware.

**`OMP_NUM_THREADS=4` in HOSTING.md is not wrong.** I measured 4 threads as 2.4x slower
than 16 and nearly reported the documented guidance as a defect. It is not: that table
is a resident fleet serving 1 to 32 concurrent requests, where four threads per request
times sixteen concurrent requests already saturates the box. One layer alone is a
different regime. The thread count belongs to the deployment shape, not to the code.

### Mistakes worth keeping

- **I read the repo and assumed it was the system.** The deployed source had moved on by
  332 lines. Pull the running artefact before reasoning about behaviour.
- **I treated `--inspect` agreement as a correctness result.** It prints one summary line
  and never touches the attention path. Caught it before reporting it as a pass, but I
  had already run it as though it were a gate.
- **I assumed the existing test covered the code I wrote.** It could not: 5 positions
  never grows an 8-slot cache. A test that cannot reach the change is not evidence about
  the change.
- **PowerShell `\$?` through ssh returns the local shell's status, not the remote
  command's.** Several "exit=True" lines earlier in the session were meaningless. The
  `VERIFIED` lines came from the harness itself and were real.

### What is still unknown

- No layer has run end to end at long context. The 1M figures are an isolated kernel.
  Expert cost per position makes a full long-context layer run impractical on one box,
  so the two measurements have not been joined.
- Only layer 3 has these changes. The other 23 MLA pods are unmodified. The 24 pods are
  generated, so landing this across them is a generator change, not 23 edits.
- The latent K/V form is 2,304 bytes per position per layer against 98,560 expanded,
  42.8x fewer bytes. Since attention is now bus-bound, this is the only remaining large
  factor — and it needs the absorb matrices, which have not been read.
- Expert compute, not expert I/O, dominates a layer at short context. 80 of 120 stages.
  Not yet profiled at stage granularity on the distributed path.
- The DIMMs run at 3600 MT/s against a 4800 rating. Never investigated.
- The `detail:expert-read-gb` counter in the standalone profiler prints gigabytes through
  a milliseconds field. Still unfixed.

### What the next cycle should do differently

Read the deployed artefact first, not the repository copy, and check whether a test can
actually reach the code being changed before treating a pass as evidence. Both failures
this cycle were of the same kind: accepting a proxy for the thing itself.

### Repository state

`STAGES.md` is new. `tansformers/mla-stride.c` and `tansformers/expert-io.c` are new
instruments. `tansformers/transformer-3/transformer-3.c` now holds the deployed source
plus both changes — note that committing it mixes a 332-line catch-up with a small
functional change, and the catch-up should land separately. `common/` is still untracked
and still needs committing; it is the reason the repo cannot currently rebuild what runs.
