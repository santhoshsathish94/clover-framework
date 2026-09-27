# The K3 data problem

## Where this comes from

`CONTEXT-k3-equation-reduction.md` records seven cycles aimed at making computation
disappear. The last of them measured the binding constraint and found it is not
computation:

```
threads   total     read_bytes   effective rate
   16     53.74 s   144.72 GB    2.69 GB/s
    8     56.84 s   157.34 GB    2.77 GB/s
    4     75.08 s   118.97 GB    1.58 GB/s
```

Halving threads from 16 to 8 costs 10%, where compute-bound work would double. And
144.72 GB at 2.69 GB/s is 53.8 s, the entire runtime. **The arithmetic is nearly free and
the bytes are expensive.**

One piece is therefore settled and one is not.

| | status |
|---|---|
| the computation | solved. `k3-model-equation.md` reproduces the engine bit-for-bit, and `eq.c` evaluates it in 50 s |
| the data movement | not solved, and not previously examined |

This document is the second problem. The governing question is the human's:

> **Why does it need so much data in the first place, and can the data be answered by an
> equation the way the computation was?**

Same standard as before. Nothing is assumed, each step is measured, and steps that
establish nothing are recorded too.

---

## Step 1 — how much data is actually needed

Before asking how to fetch less, establish how much of what is fetched is genuinely
distinct. Measured from the engine's own `router.pick` records, so the expert choices are
the model's rather than a reconstruction.

```
bytes per expert at layer 1  : 17,547,264   (17.55 MB)
distinct sizes across 896    : 1, so every expert is the same size

prefill, 92 MoE layers, 5 positions
   total expert draws            : 7,360
   distinct (layer, expert) pairs: 5,683
   unique per layer              : min 50, median 62, max 74, of 896

   bytes if every draw were fetched :  129.15 GB
   bytes of DISTINCT data needed    :   99.72 GB
   redundancy removable by caching  :   29.43 GB   (22.8%)
```

**The engine already achieves this exactly.** Its log reports **99.72 GB** read for the
prefill step, which is the distinct figure to the last decimal. There is no waste at the
selection level to remove. My own `eq.c` reads 118 to 157 GB because it has no expert
cache and re-fetches; that is a deficiency of my implementation, not of the model.

```
one generated token
   1,472 draws, 1,472 distinct (layer, expert)   25.83 GB
   engine reported for its decode step            24.95 GB
   already touched during prefill : 797 of 1,472  (54.1%)

the pool, for scale
   experts in the model            : 82,432   (1.45 TB)
   touched by this 5-token prompt  :  5,683   (6.89%)
```

Within a single token there is **no reuse at all** to exploit: all 16 experts at each of
92 layers are distinct, so 1,472 distinct loads is the floor. Across tokens there is
reuse, and the engine already takes it, which is why its decode step read 24.95 GB rather
than 25.83 GB.

### What Step 1 establishes

**Selection is already optimal.** The question "can we not select only the needed" has the
answer "that is what it already does". Top-16 of 896 *is* the selection, and the engine
fetches the selected set exactly once.

So the cost is not waste. It is the architecture: one token requires 16 experts at each of
92 layers, and each expert is 17.55 MB.

### What Step 1 therefore reframes

The question moves from *which* bytes to *how many bytes per expert*. 17.55 MB is already
4-bit MXFP4 with a shared group exponent, so it has been compressed once. Whether it can
be compressed again is a different question, and it is now the interesting one, because
Cycle G established that decompression work is close to free on this machine.

That trade did not exist before Cycle G. Spending compute to avoid I/O is only sensible
once you know compute is not the constraint.

---

## Not yet examined

- whether the 4-bit codes are compressible, i.e. their entropy
- whether the 17.55 MB per expert can be reduced without changing the result
- whether expert choice is predictable early enough to prefetch
- whether reuse across many tokens changes the picture, since all of this is 5 tokens
