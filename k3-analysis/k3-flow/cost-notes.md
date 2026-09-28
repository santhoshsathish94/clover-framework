# Cost notes — where the time actually goes

**Parked, not being acted on.** Recorded during the walkthrough of how the model works, to
come back to.

## The observation

For the embedding table, the cost is **not finding the row — it is reading it**.

Finding is `base + row × 7168 × 2`: one multiply, one add, about two CPU cycles, address
fully predictable so the processor can prefetch. Hashing would also be O(1) on average but
with tens of operations and an unpredictable load, and it would solve a problem that does
not exist here because token ids are already dense from 0 to 163,839.

Reading is **14,336 bytes** — 7168 values at 2 bytes each, widened to float one at a time.
That is roughly 224 cache lines, and it dwarfs the address arithmetic entirely. Both
lookup schemes would pay it identically.

**For the embedding specifically this does not matter at all.** One lookup per token
against a step measured at about 7 seconds; it is below the resolution of anything
recorded in this investigation. It is written down because the *principle* generalizes,
not because that read is worth touching.

## The same principle, where it does matter

The engine reads far more than it computes with, at every scale:

| what | size | measured |
|---|---|---|
| one embedding row | 14 KB per token | negligible |
| one trunk layer | 605 MB | 93 of them |
| whole trunk | **54.47 GB** (int8) | read **once**, 5.5 s at ~9,865 MB/s |
| 16 experts, one layer | 281 MB per token | 17.56 MB each |
| experts, one decode step | **12–25 GB** | after the cache warms |
| experts, prefill | 99.7 GB (5 tok) → 831 GB (225 tok) | |

I/O share of wall clock ran **22% to 45%** across the variation set.

### Corrected: the trunk is NOT re-read per token

This file first said the trunk is re-read in full on every token. **That is wrong for the
configuration every run used**, and the engine's own statistics say so:

```
v1, 1 token :  binds  93, hits   0 ( 0.0%), reads 93,  read 54.47 GB in 5.52 s
v2, 8 tokens:  binds 744, hits 651 (87.5%), reads 93,  read 54.47 GB in 5.50 s
               "reads 93 against 93 the walk owes (8 passes, 93 pinned)  -- exact"
```

Eight passes, 744 binds, **93 reads**. Trunk seconds are 5.4–5.7 in every run regardless of
whether it generated one token or eight. All 93 layers were **pinned** because
`--trunk-gb 60` exceeds the 54.47 GB int8 trunk.

So the trunk is a **one-time 5.5 s startup cost**, not a recurring one. The re-reading
described in the engine's streaming comments happens only when the budget cannot hold the
trunk — a regime **none of these runs entered**, so there is no data on it here.

The consequence: **all recurring per-token I/O is experts.**

### But the trunk IS re-read every token — from RAM, not disk

The correction above is about the *medium*, not the fact. Pinning means the 54.47 GB is
read from disk once. It does **not** mean the CPU stops touching it: every matmul in every
layer must stream its weights through the cores, so all 54.47 GB crosses the memory bus on
every single token.

At this machine's measured **47.9 GB/s** of DRAM bandwidth that is **~1.14 s per token** of
pure memory traffic before a single useful multiply. Against decode steps of 7–11 s, it is
10–16% of each one.

So the original instinct was not wrong about the re-reading. It was wrong about where the
re-reading happens and therefore about what could be done with it.

### Where the wall clock actually goes — 24 runs

Splitting total time into disk I/O and everything else:

| | share of wall clock | range |
|---|---:|---|
| **compute and memory traffic** | **64.0%** | 53–78% |
| expert reads from disk | 29.2% | 20–38% |
| trunk read from disk | 6.8% | one-time, 5.4–5.7 s in all 24 runs |

**The system is compute-bound, not I/O-bound.** That is the opposite of where this line of
inquiry started.

Honest limit on that split: "compute" here is *everything that is not disk I/O*. It
contains the genuine arithmetic, the DRAM traffic for the pinned trunk described above,
and the dequantization of experts out of the cache. The current instrument cannot separate
them. What it can say is that 64% of the time is not waiting on the disk.

Consistent with the replication finding: disk seconds are stable to ±1% between identical
runs while this compute portion varies −18% to +14%.


## What is already known that bears on reducing it

Measured during this investigation, not assumed:

- **Quantization already moves it a lot — but it changes the output.** bf16 108.81 GB,
  int8 54.47 GB, MXFP4 28.94 GB of trunk, read once at startup. **Not the same output:**
  same prompt, greedy decode, 64 tokens — int8 first diverges from bf16 at generated token
  **21** (22/64 positionally matching), MXFP4 at token **3** (9/64). All three are fluent and
  on-topic; the divergence is numerical, not sampling noise (identical configs replay
  byte-exactly across 19 replications). **No quality evaluation has been run**, so this is
  measured *difference*, not measured *degradation*. MXFP4 does not bind on upstream
  `a2ad8e5` — that binder lives on the archived `mxfp4-trunk` branch.
- **Routing breadth, not sequence length, drives expert reads.** At an identical 12
  positions, repetition read 115 GB and random tokens read 186 GB.
- **More positions means more agreement between them**: in-layer expert diversity falls
  77.2% → 58.8% → 45.2% → 14.3% as the prompt grows from 5 to 225 positions. Long context
  saturates, which is what makes it affordable at all.
- **Adjacent-step expert reuse is 42–65%**, and that is the figure the cache can exploit.
  The all-step intersection is 0.5% and is misleading.
- **Each token needs ~1472 experts; the cache held 1708 slots** — about one token's worth.

## Which trunk which runs used — read this before comparing any figure here

Two run families, two trunk formats. Conflating them produced a wrong "correction" on
2026-09-24 that had to itself be corrected.

| family | location | trunk | size | disk load | DRAM per token |
|---|---|---|---|---:|---:|
| **v-runs** (observation, traces) | `/root/k3flow` | int8 | 54.47 GB | ~5.5 s | 1.14 s |
| **benchmark runs** (A–F, CLEAN) | `/root/k3results` | bf16 | 108.81 GB | 12.2 s | 2.27 s |

Every figure in *this* file is from the v-runs and so is int8. Every figure in
`CONTEXT-kimi-k3-benchmark.md` is bf16. Both are correct; they are not interchangeable.

## The working set principle — one slice, sixteen experts

**At any instant, a layer needs its own trunk slice and its 16 chosen experts. Nothing
else.** Not the whole trunk, and not the 896-expert pool it chose from.

Per layer, bf16 trunk, measured from `trunk.json`:

| | | |
|---|---:|---|
| trunk slice, MLA (24 layers) | 0.844 GB | |
| trunk slice, KDA (68 layers) | 1.268 GB | |
| trunk slice, dense (layer 0) | 2.341 GB | |
| **trunk slice, mean** | **1.170 GB** | 1.1% of the 108.81 GB trunk |
| 16 routed experts (MXFP4) | 0.281 GB | |
| **working set** | **1.451 GB** mean, 2.622 GB worst | |

Against what is *available* to that layer — its slice plus all 896 experts, 16.89 GB — the
working set is **11.6× smaller**. Against the whole 1,554.8 GB model, **1,071× smaller**.

The trunk is already stored to support this: one `trunk.bin`, but `trunk.json` carries
`file_off` / `nbytes` / `run_start` per layer, and the engine issues exactly **93 ranged
reads**, one per layer, no matter how many tokens follow.

### The caveat that stops this being misapplied

Needing one slice at a time does **not** reduce per-token traffic. All 93 slices are needed
*within the same token*, because every layer runs for every token. So:

- **peak residency** = 1.45 GB — this is what the principle buys
- **per-token traffic** = 108.81 GB trunk + 26.11 GB experts — unchanged by slicing

Splitting `trunk.bin` into 93 files would change neither number. What the small working set
buys is that a 24 GB GPU can hold ~15 layers resident, and a handful of such cards can hold
the entire trunk with nothing streaming at all — the configuration in which the 2.27 s DRAM
term disappears rather than being re-paid over a slower bus.

The same holds on the expert side and matters more there: reading 16 experts rather than
896 is the difference between 0.281 GB and 15.72 GB per layer per token. The engine already
does this. The open cost is that the 16 are not known until the router runs mid-layer, which
is why they cannot be prefetched the way the trunk can.
- **The two shared experts are in the trunk**, 126 MB per layer, read every token and never
  streamed. They do about as much work as all 16 routed experts combined.
- **The squeeze and expand are shared by all 896 experts** in a layer (24.5 MB each), which
  is why an individual expert is only 17.56 MB.

## Open question to come back to

How to reduce the reading cost. Nothing decided, nothing attempted. The tension, as
corrected above: the trunk is read once and pinned, so the recurring cost is entirely the
expert pool — 1.45 TB against 124 GiB of memory, with 281 MB arriving per layer per token.

Worth remembering when this is picked up: **wall clock is not a measurement on this
machine** (see the reverse replication — compute time varies −18% to +14% run to run while
bytes read are exact). Any cost claim should rest on bytes, not seconds.

## The sharper version of that question

**Can you know which 16 experts a layer needs before you have computed your way to that
layer?**

First, what is *not* the problem. The engine already reads only the experts it needs, and
the arithmetic closes exactly:

```
v1: 5,683 distinct (layer, expert) pairs × 17,547,264 bytes = 99,721,101,312 = 99.72 GB
the engine reported reading                                                   99.72 GB
```

Against 1.45 TB in the pool, that is 0.007%. Selective loading is the existing design, not
something to add.

The problem is **when** the identity is known, not whether it is used. Layer L's router
reads layer L's input, which requires layers 0…L−1 to have finished. So the 16 become
known microseconds before they are needed, and 281 MB has to arrive before the layer can
proceed. Contrast the trunk, whose layer order is fixed 0…92 for every token — that is
exactly why the engine can prefetch trunk layer L+1 while computing layer L and hide the
read entirely. Expert identity is data-dependent, so the same trick does not apply.

What the engine does instead: hand all 16 to the cache in one call so the drive gets queue
depth (the code notes that one-at-a-time gives *"a queue depth of one against a drive that
needs depth to reach its rated bandwidth"*), and keep experts across tokens, where
adjacent-step reuse runs 42–65%.

### This is partly answerable from data already on disk

No new runs needed to start. The traces hold every routing decision — 460 in v1, 20,700 in
v6 — with layer, position, the 16 ids and their weights. Questions that could be asked of
what is already saved:

- How well does layer L−1's chosen set predict layer L's, within the same token?
- Is there a stable popular set per layer that covers most picks, so it could be pinned?
  Some evidence against: v1 used 62 distinct experts per layer from 5 positions, but v6
  used **515 of the 896 per layer** across 225 positions, so over a long prompt more than
  half the pool is touched.
- Does the routing bias, which steers selection but not weights, make the choice more
  predictable than the scores alone?

### A caveat on hit rates when this is picked up

A reported 100% cache hit rate can mean the prefetcher had just read it from disk, not
that a read was avoided. The engine says so itself in v1: *"of those hits: 5599 came from
the batch prefetch, i.e. read from disk this token; TRUE resident hit rate 0.00%"*. Judge
by bytes read, not by hit rate.

## Both questions answered — from data already on disk, no new runs

### Q1: the wide reads cost nothing. There is nothing to reduce.

The engine separates its I/O into trunk, experts and model tables. Across **every** run:

```
model tables  0.0 s          in all 19 runs, without exception
trunk         5.4 – 5.7 s
experts      14.7 – 98.7 s
```

The 7168-wide table reads — one embedding row in, the lm_head projection out — do not
register at all. The embedding row is 14 KB; the lm_head is ~2.35 GB of resident memory
traffic per token, and even that disappears against the disk reads.

**The cost is entirely trunk and experts, and it is disk, not memory.** Effort spent on
the token's own 7168 numbers would be spent on 0.0 s.

### Q2: no, the 16 cannot be known in advance. Three routes tested, two dead.

**Route 1 — the token itself. Dead.** v3 is twelve copies of one token in one sentence. If
routing were decided by the token, all twelve positions would choose the same 16.

```
layers where all 12 positions chose identical experts   0 of 92
layers where all 12 chose DIFFERENT sets               91 of 92
mean overlap, position 0 against the others            5.0%   (random ≈ 0.9%)
```

Five percent is above chance but means fifteen of sixteen experts differ. Routing reads
the context-laden residual, and by the time a token reaches layer 40 its residual has
absorbed forty layers of attention from everything before it. The same word in the same
sentence is, to the router, a different thing at every position.

**Route 2 — the previous layer. Dead, and exactly dead.**

```
                layer L -> L+1 same-index overlap
v1 factual      1.54%
v4 code         1.93%
v6 prose        1.69%
random baseline 1.79%
```

Indistinguishable from chance. Expected in hindsight: every layer has its own router and
its own 896 experts, so index 42 in layer 5 has no relationship to index 42 in layer 6.
Worth having measured rather than assumed.

**Route 3 — the previous token, same layer. Real, and already fully exploited.**

```
v2 factual x8   38% 23% 46% 52% 50% 42% 30%   mean 40.1%
v3 repetitive   48% 55% 56%                   mean 53.2%
v7 nonsense     31% 43% 41%                   mean 38.5%
```

Then the decisive comparison. A token needs 1472 experts (92 layers × 16), which is
25.8 GB if nothing is reused. Against the bytes the engine actually read:

| step | routing overlap | reads avoided |
|---:|---:|---:|
| 1 | 38% | 3.4% |
| 2 | 23% | **23.4%** |
| 3 | 46% | **45.5%** |
| 4 | 52% | **52.5%** |
| 5 | 50% | **50.1%** |
| 6 | 42% | **41.6%** |
| 7 | 30% | **29.9%** |

**Six of seven match to within half a percentage point.** The cache is already converting
essentially all of the available reuse into avoided reads. Step 1 is the exception — 38%
overlap but only 3.4% avoided — because the cache was still filling after prefill.

### Conclusion, stated plainly

There is **no unexploited predictability** in the routing on this evidence. The only
structure is temporal, it is worth 40–53%, and the existing cache captures it almost
perfectly. A prefetcher built on layer-to-layer or token-identity prediction would be
building on noise.

This does not close the cost question in general, but it closes the *routing prediction*
route to it. And with the trunk correction above, what remains is narrower than it looked:
the trunk is a one-time 5.5 s cost, the expert reads are already at the floor the routing
allows, so the recurring I/O is at its minimum **for this architecture and this cache
size**. Anything further has to change one of those two, not the scheduling.

## Routing prediction, tested across prompts (2026-09-26)

The section above tested prediction **within one run**. The creator's hypothesis was
sharper: the router is a deterministic function of the 7168-wide state, so a one-to-one
relationship exists and might be exploitable. Everything below is offline on traces already
on disk — no new runs.

The prize if any of it worked: expert reads are **35% of wall clock and cannot be
prefetched**, because layer L's router reads layer L's own input. The trunk, by contrast,
has a fixed 0…92 order and its prefetch already hides 87–96% of its device time.

### Route 4 — a static popular set per layer. DEAD across prompts.

Top-193 most-used experts per layer, taken from v6 (prose, 228 positions):

| coverage of | | |
|---|---:|---|
| v6 itself | **80.0%** | fitted and tested on the same data — the ceiling |
| v1 factual | 25.0% | |
| v4 code | 23.1% | |
| v5 French | 20.5% | |
| v7 nonsense | 16.4% | |
| **random baseline** | **21.5%** | |

Three of four sit at or below chance. The 80% was pure overfitting to one prompt.

### Route 5 — a learned cross-layer mapping. Real signal, but prompt-specific.

Route 2 above tested whether expert *index* 42 at layer L recurs as index 42 at layer L+1,
which was never going to work. This instead **learns** P(e at L | f at L−1) from
co-occurrence. Trained on v6 positions 0–149, tested on 150–227:

| predictor | hit rate on held-out positions |
|---|---:|
| learned cross-layer co-occurrence | **27.6%** |
| previous position, same layer | 31.2% |
| top-16 most frequent | 20.4% |
| random | 1.8% |

Co-occurrence reduces exactly to the frequency predictor if layers were independent, so
**beating it by 7.2 points is genuine cross-layer signal** — the first evidence of any.

Then trained on v6 and tested on *other prompts*:

| test prompt | co-occurrence | prev-position | frequency | random |
|---|---:|---:|---:|---:|
| v1 factual | 13.9% | **24.8%** | 3.2% | 1.8% |
| v4 code | 5.2% | **32.9%** | 3.6% | 1.8% |
| v5 French | 7.6% | **40.4%** | 3.3% | 1.8% |
| v7 nonsense | 4.3% | **30.7%** | 2.0% | 1.8% |

**27.6% collapses to 4.3–13.9%.** The signal was real and entirely prompt-specific. Only
previous-position survives, and it needs no training because it is a property of the model
rather than of a prompt — which is the same 40–53% temporal reuse the cache already
converts into avoided reads.

### Route 6 — the threshold explanation. Tested and NOT the cause.

Selection is top-16 by the *biased* score while the stored weight is *unbiased*, and the
bias reorders the top pick in 20–28% of decisions. So set overlap should understate
similarity: a 0.003-weight expert near the 16/17 boundary counts the same as a 0.645 one.
Splitting the same transfer test by weight rank:

| weight rank | all layers | layer 12 | layers 53–57 |
|---|---:|---:|---:|
| rank 1–4 (top) | 29.1% | 78.4% | 40.7% |
| rank 5–8 | 28.1% | 74.8% | 36.9% |
| rank 9–16 (tail) | 28.3% | 73.8% | 37.7% |
| random baseline | 27.9% | | |

**The most confident picks transfer no better than the threshold-sensitive tail.** The
instability is not an artifact of where the cut falls; different prompts genuinely route
differently even at high confidence.

### THE EXCEPTION: layer 12

Averaging 92 layers hid this. Per layer, top-250 from v6 against unseen prompts:

| layer | distinct on v6 | mean transfer | v1 | v4 | v5 | v7 |
|---:|---:|---:|---:|---:|---:|---:|
| **12** | 387 | **75.2%** | 73.8% | 81.5% | 75.0% | 70.4% |
| 54 | 314 | 49.5% | | | | |
| 56 | 271 | 42.4% | | | | |
| baseline | | 27.9% | | | | |

Only **3 of 92** layers beat the baseline by more than 1.5×, and layer 12 sits 25 points
clear of second place. How small the pinned set can be there:

| pin N | GB | transfer |
|---:|---:|---:|
| 64 | 1.12 | 46.2% |
| 128 | 2.25 | 60.1% |
| **250** | **4.39** | **75.2%** |
| 400 | 7.02 | 85.5% |

It works because **layer 12 routes narrowly for every prompt**, not just the fitted one:
53 distinct experts on v1, 77 on v5, 123 on v4, 128 on v7, 387 on v6.

**The obvious mechanism was checked and rejected.** Layer 12 is a snapshot layer, but
snapshot layers as a class are ordinary: mean transfer 30.4% against 28.3% for all others.
So this is one layer with no explanation, n=1 of 92. Do not build on it without a fifth
unrelated prompt as confirmation.

### What is still untested, and it is the actual hypothesis

Every test above uses expert **indices**. The creator's claim is about the **7168-value
state**, and the traces record only `l2, min, max, mean, fnv` per tensor — no vectors. A
coarse per-position probe (`row_l2`, `row_argmax`) was tried on s160 and returned 16.7%
against 15.3% for a random same-prompt pair; with 2 of 7168 dimensions that is
uninformative either way, not evidence.

The weight-rank result makes the vector test matter more rather than less. If even
high-confidence picks do not transfer, either the states are genuinely that different
between prompts — in which case no predictor can ever work — or the states are similar and
the map is extremely sensitive, in which case every index-based test here measured the
wrong thing. **Those two are indistinguishable without the vectors.**

The run needed: one tap dumping `attn_res.pre_mlp` beside each `router.pick`. 7168 × 4 B ×
92 layers = **2.64 MB per position**, so ~132 MB for 50 positions. Pre-register the bar
before running it: the predictor must beat previous-position (24.8–40.4%) **on an unseen
prompt**, because that baseline is free.

### A separate lever found while doing this

The weight distribution is skewed: rank-1 mean 0.154 (max 0.852), rank-16 mean 0.029
(min 0.000), and the **top 4 carry 43.5% of the total weight**. Fetching only the top 4
would move 0.070 GB per layer instead of 0.281 — a 4× cut in the term that is 35% of wall
clock. It changes the output, so it is a quality trade measurable the same way int8 and
MXFP4 divergence were measured. Against what prediction can offer, it may be the larger
lever.


## The per-layer profile is two sawtooths, not a flat cost (2026-09-25)

Every earlier cost note in this file used a **mean** layer — 1.170 GB of trunk, 0.281 GB of
experts. That average hides the shape, and the shape is the point. Rebuilt by joining
`/root/k3flow/v1.jsonl` (93 layers × every stage, 5 positions) against `trunk.json`.

Arithmetic check first: expert bytes summed from the trace come to **99.721 GB** against the
**99.72 GB** the engine reported for the same run. Two independent counts agreeing, so the
join is sound.

### Sawtooth 1 — bytes, period 4

MLA sits at layers 3, 7, 11 … 87, then **91 and 92 back to back**. Its trunk slice is
smaller than KDA's, so the per-layer cost oscillates every fourth layer:

| | layers | trunk | + experts | total |
|---|---:|---:|---:|---:|
| dense (L0) | 1 | 2.341 GB | none | **2.341 GB** |
| MLA | 24 | 0.844 GB | 0.877–1.12 GB | **1.920 GB** mean |
| KDA | 68 | 1.268 GB | 0.877–1.30 GB | **2.355 GB** mean |

```
min   1.722 GB   layer 83 (MLA)
max   2.566 GB   layer 1  (KDA)
mean  2.242 GB
spread max/min = 1.49x      KDA/MLA = 1.23x
```

The expert half is not flat either: **50 to 74 distinct experts per layer**, mean 61.8 of
the 80 slots that 5 positions × 16 provides — the 77.2% in-layer diversity, per layer.

### Sawtooth 2 — magnitude, period 12

Snapshots fire at 0, 12, 24, 36, 48, 60, 72, 84. The residual is **replaced**, not added to:

| | before | after | collapse |
|---|---:|---:|---:|
| L11 → L12 | 269.55 | 12.34 | 21.8× |
| L23 → L24 | 356.21 | 1.38 | **257.6×** |
| L35 → L36 | 152.71 | 1.84 | 82.8× |
| L47 → L48 | 430.05 | 1.89 | 227.8× |
| L59 → L60 | 482.16 | 13.35 | 36.1× |
| L71 → L72 | 107.46 | 7.32 | 14.7× |
| L83 → L84 | 138.83 | 17.10 | 8.1× |

Mean L2 at snapshot layers **7.14** against **82.56** across all 93; maximum 482.16. Eight
blocks of twelve, each starting near zero and climbing, with the history parked on a stack
that both aggregations re-inject. The stack itself grows 0 → 8, so late layers attend over
nine sources where early ones attend over two.

### What the shape costs

**One five-position pass is 208.533 GB** — 108.812 trunk + 99.721 experts. **One token is
134.64 GB** — 108.812 trunk + 25.83 experts; the two are easy to confuse and the larger
number is not a per-token figure. What that moves is not a 28 KB hidden state: AttnRes
attends over `[stack..., running]` twice per layer, so **150.3 KB mean** crosses each
boundary, 13.8 MB end to end. That is roughly **9,900 bytes read per byte of state
advanced**, and none of it can be skipped or reordered: layer L needs layer L−1's output,
and its 16 experts are not known until its own router has run on that output.

Two consequences that the averaged view could not show:

1. **Any design that gives one layer to one device is unbalanced by 1.49× before it
   starts.** The rate is set by the heaviest stage; the lightest sits idle a third of the
   time. Layer 0 alone is 2.341 GB against a 1.722 GB minimum.
2. **The magnitude sawtooth is why the model tolerates aggressive quantization unevenly.**
   A snapshot layer carries a residual near 1.4 while its neighbour carries 430 — the same
   absolute error means very different relative error at the two. Not measured; recorded as
   the obvious next question rather than a finding.

### Why no single machine does this efficiently

Not a property of our box. A property of the workload:

- **1,555 GB resident** against 124 GiB of RAM or 141 GB of HBM. Nothing holds it; something
  must stream.
- **~1.5 FLOP per byte** at one token — 2.2 GFLOP of arithmetic against 2.24 GB of reading,
  per layer. A CPU wants ~5, a large accelerator wants ~200. Every machine is starved, and
  the faster the machine the more starved it is.
- **The 99.721 GB of expert reads is data-dependent**, so unlike the trunk it cannot be
  prefetched — the engine's fixed 0…92 layer order makes the trunk hint never wrong, and
  nothing equivalent exists for the experts.



