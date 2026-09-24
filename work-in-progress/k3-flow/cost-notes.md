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

| what | bytes per token | measured |
|---|---|---|
| one embedding row | 14 KB | negligible |
| one trunk layer | 605 MB | 93 of them per token |
| whole trunk | **54.47 GB** (int8) | read 5.5 s at ~9,856 MB/s |
| 16 experts, one layer | 281 MB | 17.56 MB each |
| experts, one decode step | **12–25 GB** | after the cache warms |
| experts, prefill | 99.7 GB (5 tok) → 831 GB (225 tok) | |

I/O share of wall clock ran **22% to 45%** across the variation set.

**The trunk is re-read in full on every single token.** That is the single largest fixed
cost in the system and the one structural fact most worth thinking about later.

## What is already known that bears on reducing it

Measured during this investigation, not assumed:

- **Quantization already moves it a lot.** Same model, same output: bf16 108.81 GB,
  int8 54.47 GB, MXFP4 28.94 GB of trunk per token. MXFP4 does not bind on upstream
  `a2ad8e5` — that binder lives on the archived `mxfp4-trunk` branch.
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

How to reduce the reading cost. Nothing decided, nothing attempted. The obvious tension:
the trunk is re-read per token because it does not fit in RAM alongside everything else,
and the expert pool is 1.45 TB against 124 GiB of memory.

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

