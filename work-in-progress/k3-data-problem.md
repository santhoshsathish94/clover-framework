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

## Step 2 — what the equation needs, as distinct from what the model fetches

**The correction that produced this step came from the human.** Step 1 measured what the
*model* loads. The model loads 16 experts per layer because its router selects 16. But the
equation's output is a single token, and whether that token requires all 16 contributions
is a different question, which Step 1 never asked.

The routed output is a weighted mixture:

$$\mathrm{accL} \;=\; \sum_{j=0}^{15} \pi_j\, e_j$$

### The weights first

```
router weights, 460 draws of 16 experts

rank  1  mean 0.16460      rank  9  mean 0.05005
rank  2  mean 0.10786      rank 12  mean 0.04114
rank  4  mean 0.07686      rank 16  mean 0.03036

cumulative mass : top 1 = 16.5%,  top 4 = 43.7%,  top 8 = 67.9%
experts needed to reach 90% of the mass : median 13 of 16
the smallest of the 16 exceeds 0.02 in 83.9% of draws
```

Uniform would be 0.0625 each. The distribution is only mildly skewed around uniform, with
a rank-1 to rank-16 ratio of 5.4. Read as weights, every expert appears to matter.

**That reading would have been wrong**, and it is exactly the kind of inference the
previous document records failing repeatedly. What matters is not the mass removed but
whether the answer changes.

### The measurement

Load only the top $k$ of the 16 selected experts, renormalizing over those kept.

```
k    expert loads   requested   read_bytes   total     token   margin
16       7,360      129.15 GB    98.60 GB   48.70 s   17374   3.678
12       5,520       96.86 GB          -    33.78 s   17374   3.150
 8       3,680       64.57 GB    39.16 GB   21.49 s   17374   3.361
 4       1,840       32.29 GB    13.47 GB   12.17 s   17374   1.811
 2         920       16.14 GB          -     7.91 s     261   0.319
 1         460        8.07 GB          -     7.02 s     276   0.411
```

**The model needs 16 experts. The emitted token needs 4.**

At $k = 4$: **7.3x less device I/O**, 98.60 GB down to 13.47 GB, and **4.0x faster**,
48.70 s down to 12.17 s, with the same token.

The reads fall faster than the requests, 2.5x and 2.9x for successive halvings, because
the smaller working set also begins to fit in page cache. Part of the gain is therefore
fewer bytes requested and part is a better hit rate, and the second part depends on
having 124 GB of RAM.

Below $k = 4$ it breaks. At $k = 2$ the token changes to 261 and at $k = 1$ to 276.

### What this is, stated precisely

This is **not** a reduction in the sense the previous document required. It is an
approximation. The logits move, 18.113 to 16.598, and the margin halves, 3.678 to 1.811.
Under the criterion fixed for the equation work, bit-exactness, it fails immediately.

Under the observable that actually matters to a user of the model, *the emitted token*, it
holds, and it is the first change in this entire body of work that makes the model
materially faster without changing the answer.

Both statements are true, and which one governs depends on the observable, exactly as
`k3-redundancy.md` argued from a different direction.

### What this does not establish

- **One prompt, one token.** Five tokens of `The capital of France is`. The margin at
  $k = 4$ is 1.811 against 3.678, so the headroom is halved and a different prompt could
  cross it.
- **Prefill only.** The 17374 above is the prefill token. Whether $k = 4$ also preserves
  the decode token 20829 is untested.
- **The threshold is not established.** $k=4$ works and $k=2$ does not, on this prompt.
  Where the boundary sits in general is unknown.
- **Part of the speedup is machine-specific**, since it depends on the working set
  fitting in RAM.

---

## Not yet examined

- whether the 4-bit codes are compressible, i.e. their entropy
- whether the 17.55 MB per expert can be reduced without changing the result
- whether expert choice is predictable early enough to prefetch
- whether reuse across many tokens changes the picture, since all of this is 5 tokens
