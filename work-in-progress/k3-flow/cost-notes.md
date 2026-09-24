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
