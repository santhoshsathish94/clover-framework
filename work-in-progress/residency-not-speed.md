# Residency, not speed

**Why one machine cannot run a 1.56 TB model, and why many small ones can.**

Exploratory. Every figure below is measured on hardware we rented, or read out of the
checkpoint. Where something is modeled rather than measured it says so, and the last
section lists what has not been tested at all.

---

## The problem, stated in bytes

Kimi K3 is 2.78 trillion parameters. Held for inference:

| | bytes | share |
|---|---:|---:|
| trunk — attention, norms, routers, shared experts | 108.81 GB | 7.0% |
| routed experts — 896 per layer, 92 layers | 1,446.5 GB | **93.0%** |
| **total resident** | **1,555.3 GB** | |

Read out of `trunk.json` and the safetensors headers, not from documentation. One routed
expert is exactly **17,547,264 bytes**; 16 of them per layer across 92 layers comes to
**25.830 GB per token**, which matches to the decimal what the engine reports reading.

The machine we ran it on: one 16-core CPU, 124 GiB of RAM, two NVMe drives. The model is
**12.5× larger than the memory**. It is also 11× larger than the largest single accelerator
that exists.

## What we did about it, and why it was not enough

A full optimization campaign, every change measured against the one before:

| | |
|---|---|
| a degraded PCIe link repaired | 18% |
| thread count set to physical cores | 8.1% |
| prefix reuse on turn two | 2.06× |
| expert cache memory given back to the trunk | 6.1%, on 4.1 GB *less* memory |
| thread binding across the two dies | 4.9% |
| expert reads split into 1 MiB chunks | 5.7% |
| a batched, bit-exact matmul | ~1% |

**Start of campaign 9.40 s per token. End 5.311 s.** Real work, and two of those changes
were contributed back upstream. It did not solve anything.

Because here is where a token's time actually goes, on the best configuration we reached:

| | bytes moved | rate | time | share |
|---|---:|---:|---:|---:|
| trunk streamed from RAM into the cores | 108.81 GB | 47.9 GB/s | 2.27 s | **43%** |
| experts read from SSD | 25.83 GB | 13.8 GB/s | 1.87 s | **35%** |
| experts streamed from RAM into the cores | 25.83 GB | 47.9 GB/s | 0.54 s | 10% |
| everything else | — | — | 0.63 s | 12% |
| **one token** | **~160 GB** | | **5.31 s** | |

**To produce one token the machine moves about 160 GB.** It is not computing hard — the
CPU sits at roughly 6% of its arithmetic peak throughout. It is hauling.

### Every lever we had moved the rate, never the residency

That distinction is the whole document. The 2.27 s trunk term is 108.81 GB divided by
47.9 GB/s of memory bandwidth. No scheduling, no cache policy, no thread count and no
storage change touches it, because the bytes have to cross that bus whatever you do. We
measured this the hard way:

- **More RAM does not help.** The trunk was already 100% pinned, 93 of 93 layers. None of
  those reads go to disk, so holding more changes nothing about them.
- **More cores do not help.** The kernel reaches 18% of linear scaling across 32 threads
  and memory throughput *peaks at 2–4 threads and then declines*. 16 threads beat 20, 24,
  28 and 32.
- **A better cache does not help.** Of 33,469 expert requests, **zero** had a reuse
  distance short enough for the cache to serve. Not few — zero, as a matter of geometry.
- **Capacity fights bandwidth.** The 128 GB of RAM is bought by filling all four slots,
  and four dual-rank modules on two channels train at 3600 MT/s instead of 4800. The
  capacity costs a quarter of the bandwidth.

**The machine is not slow. It is being asked to do the least cache-friendly thing
possible, one token at a time.** Arithmetic intensity is about **1.5 FLOP per byte** — 2.2
GFLOP of work against 2.24 GB of reading, per layer. Hardware wants 5 to 200. Every
machine is starved on this workload, and the faster the machine, the more starved it is.

**So the problem is not that the machine is too slow. It is that the model is not
resident.** That is the thing no amount of tuning on one box can fix.

---

## The measurement that changes the picture

At any instant, a layer needs **its own trunk slice and its 16 chosen experts. Nothing
else.** Not the whole trunk, and not the 896-expert pool it chose from.

| | |
|---|---:|
| trunk slice, MLA (24 layers) | 0.844 GB |
| trunk slice, KDA (68 layers) | 1.268 GB |
| trunk slice, dense (layer 0) | 2.341 GB |
| 16 routed experts | 0.281 GB |
| **working set** | **1.451 GB mean, 2.622 GB worst** |

Against the 1,555.3 GB model, the working set is **1,071× smaller**.

The model is already stored to support this: one `trunk.bin`, but `trunk.json` carries a
byte offset and length per layer, and the engine issues exactly **93 ranged reads**, one
per layer, no matter how many tokens follow.

### And the number that removes storage from the loop entirely

One layer plus **all 896** of its experts is **17.0 GB**.

A 24 GB card holds that with 7 GB to spare. Ninety-three such cards hold the entire
1,555 GB model with **nothing streaming from disk at any point**. The 35% of wall clock
our machine spends waiting on SSD does not get faster — it stops existing.

### What has to move between them: 150 KB

*(First said 28 KB, then 148.4 KB from a five-position estimate. The figure below is
measured directly over 41,624 aggregations.)*

The obvious answer is one hidden state: 7168 values at 4 bytes, 28 KB. That is what I
published, and it is not what this model does.

**AttnRes attends over `[stack..., running]` twice per layer**, and the stack gains an entry
every 12th layer. A stage therefore has to receive the whole stack, not a single vector.
Measured over a real token: **185 aggregations, 1,002 sources attended, stack depth 0 to 8,
mean sources per aggregation 5.37.**

```
payload    layer 1    56.0 KB
           layer 48  140.0 KB
           layer 92  252.0 KB
           mean      150.3 KB      end to end 14.0 MB
```

Against 1.45 GB of local reading per layer that is **1 : 9,406** — not the 1 : 80,000 I
first claimed. The compute those 1,002 passes cost is genuinely negligible (~22 MFLOP, 0.01%
of the token). It matters only because the stack has to cross the wire.

**A 1 Gbit/s port sustains roughly 812 tokens per second of hop traffic.** The pipeline
those cards form produces about 837 tok/s. **So ordinary ethernet does not carry this.** The
link, not the cards, becomes the binding constraint — and the GEX45 this design was costed on
has exactly a 1 Gbit/s port. 2.5 GbE clears it with room; 1 GbE does not. The original
4,360 tok/s figure implied five times the headroom that exists, and the headroom is in fact
negative.

That is still the difference between this shape and tensor parallelism. Tensor parallelism
splits a single layer across devices and must reconcile partial results every layer, which
is why it needs 900 GB/s between cards. Splitting *by layer* moves the stack once per
boundary — three orders of magnitude less, not five.

### And it is not a chain of 93

Counted from the trace, one token is not 93 sequential steps:

```
AttnRes aggregations                     185   layer 0 has no pre-attention one
  sources consumed by them             1,002   each = 1 RMS norm + 1 dot + 1 accumulate
pre-attn / pre-mlp norms                 186
attention ops                             93   (69 KDA, 24 MLA)
router evaluations                        92
routed expert evaluations              1,472   (92 x 16)
shared expert evaluations                184   (92 x 2)
vector-producing operations, total     3,029
```

**The running state is regenerated 185 times, each time from up to nine sources.** And the
eight snapshots pushed at layers 0, 12, 24 ... 84 are read by every later layer, so the
dependency graph is **93 nodes in series plus 400 long-range broadcast edges** — not a line.

The 1,002 source passes cost about 22 MFLOP against 206 GFLOP for the token, 0.01%, so this
is cheap to compute and awkward to distribute. It is the same fact as the 150.3 KB payload
seen from the other side: a stage cannot be handed only its predecessor's output, because it
also needs the eight snapshots that predecessors long past produced.

### What each of those 1,002 transformations actually does

Measured per source across 41,624 aggregations:

```
mean largest weight                0.832
mean effective sources 1/sum(p^2)  1.46    out of 5.37 offered
mean weight on the running state   0.805
running state is the largest       93.1% of the time
```

**The aggregation is overwhelmingly a pass-through.** It is offered five sources and behaves
like one and a half, and the one it usually picks is the state already in hand. Two layers
break the pattern systematically: at **layer 5** the running state is largest only 4% of the
time (mean weight 0.359) and at **layer 10** only 28% — both sit inside the first block, so
both are reaching back to the layer-0 snapshot.

This does **not** shrink the wire payload, because dropping the small weights is lossy. Ship
enough sources for 99% of the softmax mass and the payload falls to 103.5 KB, 1.45x smaller;
but the engine's value here has been bit-exactness, and 99% is a different model.

---

## What the layers actually look like

Measured by joining a real 93-layer trace against the byte counts. Two patterns, and
neither is a smooth climb.

### Bytes: a sawtooth of period 4

MLA attention sits at layers 3, 7, 11 … 87, then 91 and 92 back to back. Its trunk slice
is smaller than KDA's, so the per-layer cost oscillates:

```
min   1.722 GB   layer 83 (MLA)
max   2.566 GB   layer 1  (KDA)
mean  2.242 GB
spread 1.49x                 KDA/MLA 1.23x
```

The expert half is not uniform either: **50 to 74 distinct experts per layer**.

### Magnitude: a sawtooth of period 12

Every twelfth layer the residual is **replaced** rather than added to:

| | before | after | collapse |
|---|---:|---:|---:|
| L11 → L12 | 269.55 | 12.34 | 21.8× |
| L23 → L24 | 356.21 | 1.38 | **257.6×** |
| L47 → L48 | 430.05 | 1.89 | 227.8× |
| L59 → L60 | 482.16 | 13.35 | 36.1× |

Mean at snapshot layers **7.14** against **82.56** across all 93. Eight blocks of twelve,
each starting near nothing and climbing.

**This matters for anyone building the shape**: give one layer to one device and the
arrangement is unbalanced by **1.49×** before it starts. The rate is set by the heaviest
stage. Layer 0 alone is 2.341 GB against a 1.722 GB minimum, and splitting that one layer
is worth more than choosing a faster card.

---

## Why this is a real answer and not a thought experiment

The hardware exists and is rentable today. Hetzner's GEX45, read from their page on
2026-09-25:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell SFF |
| VRAM | **24 GB** GDDR7 ECC |
| bandwidth | **432 GB/s** |
| power | **70 W** |
| network | 1 Gbit/s, unlimited traffic |
| price | **€214.00 / month**, €209 setup, excl. VAT |

24 GB is exactly the size that holds one layer and all 896 of its experts. 432 GB/s is
**9× our CPU's memory bandwidth**, per card, and there are as many cards as you choose to
rent. 70 W is **6.17 GB/s per watt**, against 6.86 for an H200 — within 11%, on a card you
can rent by the month.

### What our own measurements say each term becomes

| term on our machine | share of a token | on this shape |
|---|---:|---|
| expert read from SSD | 35% | **gone** — 896 experts resident per card |
| trunk stream at 47.9 GB/s | 43% | VRAM at 432 GB/s per card, 93 cards in parallel |
| expert stream at 47.9 GB/s | 10% | same |
| interconnect | n/a | 150.3 KB per hop, 1 : 9,406 against local traffic |

The two terms that are 78% of our token both come from *not holding the model*. Hold it,
and they are not optimized — they are removed.

---

## What this is not

Stated plainly, because a document that only lists advantages is not evidence.

- **Per-user latency is poor.** A token must cross 93 layers in order, and with one
  request in flight 92 of the 93 cards are idle. Modeled at roughly **430 ms per token**
  against our CPU's measured 5,895 ms — 13.7× better than what we have, and far worse
  than a tightly coupled machine.
- **"93 stages" is the coarsest possible cut, and it is not a line.** Every layer past the
  first block also needs the snapshots pushed at layers 0, 12, 24 and so on, so a late stage
  cannot start from its predecessor's output alone — 400 broadcast edges sit on top of the
  93 in series. Any staging plan has to carry the stack, which is what makes the payload
  150.3 KB instead of 28 KB.
- **Batching is fragmented.** With N concurrent requests spread over 93 stages, each stage
  only ever batches N/93. Weights are read per batch, not per token, so a smaller batch
  means more reading per token produced. This is the real cost of the shape and it does not
  go away.
- **The stages are unbalanced by 1.49×** and layer 0 is the worst offender.
- **Nothing here has been run on a GPU.** Every number is a 7950X3D with 47.9 GB/s of DRAM
  and two NVMe drives. The residency argument is arithmetic over measured byte counts and
  vendor bandwidth figures; it is a hypothesis about GPUs, not a finding.
- **Whether the engine can be split this way is untested.** It is currently one process
  walking all 93 layers.
- **Expert identity is data-dependent, and cannot be precomputed.** Layer L's router reads
  layer L's input, so the 16 experts become known microseconds before they are needed. Six
  ways to predict them in advance have been tested and all six fail across prompts. The map
  itself is not the problem: it is exactly deterministic, and it is smooth — cosine
  similarity between two router inputs correlates with expert overlap in every layer of two
  unrelated prompts (r = 0.22–0.68). The problem is coverage. A state library built from one
  prompt scores 14.2% on an unseen prompt against 31.1% for simply reusing the previous
  position, because the nearest stored neighbor is 0.661 cosine away instead of 0.807.
  Closing that gap needs ~10^14 states per layer, **about a zettabyte — some 700 million
  times the size of the model.** So the only structure worth relying on is temporal reuse of
  40–53%, which the existing cache already captures almost perfectly, and one layer of
  router lead time for prefetch.
- **Layer 12 is a standing anomaly.** Pinning its 250 most-used experts transfers to unseen
  prompts at 75.2% against a 27.9% baseline; no other layer comes close and nothing
  explains it yet.

## Why it is worth writing down anyway

Not because it beats anything. Frontier labs build for fleets of accelerators because
that is the reality they face, and it is a sound answer to it. This is a different
reality: no datacenter, no fabric, a monthly bill instead of a capital purchase.

What the measurements support is narrow and worth stating exactly:

> **The binding constraint on running a very large model is residency, not speed. One
> machine cannot hold 1,555 GB, and no amount of tuning changes that. But the model never
> needs more than 1.45 GB resident at once per layer, and what has to cross between layers
> is 148 KB against 2.24 GB read locally. Those two measured facts are what make many small
> machines a real option where one large one is not.**

Everything else in this document is arithmetic on top of those two numbers.

---

**Evidence** — `work-in-progress/CONTEXT-kimi-k3-benchmark.md` (the measurement campaign,
including the predictions that turned out wrong), `work-in-progress/k3-flow/cost-notes.md`
(per-layer costs and the routing-prediction dead ends), `work-in-progress/CONTEXT-k3-observation.md`
(what was observed in 24 traced forward passes).
