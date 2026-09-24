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

- **Quantization already moves it a lot.** Same model, same output: bf16 108.81 GB,
  int8 54.47 GB, MXFP4 28.94 GB of trunk, read once at startup. MXFP4 does not bind on
  upstream `a2ad8e5` — that binder lives on the archived `mxfp4-trunk` branch.
- **Routing breadth, not sequence length, drives expert reads.** At an identical 12
  positions, repetition read 115 GB and random tokens read 186 GB.
- **More positions means more agreement between them**: in-layer expert diversity falls
  77.2% → 58.8% → 45.2% → 14.3% as the prompt grows from 5 to 225 positions. Long context
  saturates, which is what makes it affordable at all.
- **Adjacent-step expert reuse is 42–65%**, and that is the figure the cache can exploit.
  The all-step intersection is 0.5% and is misleading.
- **Each token needs ~1472 experts; the cache held 1708 slots** — about one token's worth.
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


