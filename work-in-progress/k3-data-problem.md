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

### What "margin" means, and a correction to how it was used above

`margin` is the raw logit gap, `logits[top1] - logits[top2]`. It is **not scale-invariant**:
Step 28 of the equation work showed that multiplying every logit by 0.783 moved the margin
from 3.678 to 2.880 while changing nothing about the decision.

So the margin column above could have been reporting a shrinking logit vector rather than
a weakening decision. That was tested rather than assumed.

```
k    raw margin   logit std   scale-free   p(top1)   p(top2)/p(top1)
16     3.678        2.568      1.432 sd     0.792        0.025
 8     3.361        2.527      1.330 sd     0.790        0.035
 4     1.811        2.547      0.711 sd     0.539        0.163
 2     0.319        2.668      0.119 sd     0.152        0.727
```

**The concern was unfounded**: the logit spread is constant at 2.53 to 2.67 across all
$k$, so the vector is not contracting and the raw margin was a fair measure.

**But the softmax probability changes the reading of the table**, and the earlier summary
was too generous:

- **$k = 8$ costs no confidence at all.** 0.790 against 0.792. Identical certainty, for
  2.5x less I/O and 2.3x less time.
- **$k = 4$ keeps the token but nearly halves the confidence**, 0.539, with the runner-up
  rising from 2.5% to 16.3% of the winner's probability.

The defensible statement is therefore **$k = 8$ is nearly free, and $k = 4$ preserves the
answer while making the model substantially less sure of it.** For greedy decoding the
distinction does not matter. For sampling, or for any use of the probabilities, it matters
a great deal, and the raw margin hid it.

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

## An aside that turned out not to be an aside — looking at the numbers

The human asked to plot the model's numbers as points in a circle, with no expectation,
just to see. A C program maps each value to a point: angle from its index, radius from
its magnitude, positives warm and negatives cool. Rendered at `/root/k3raw/circle.png`.

Three of the four panels show only geometry. The 7,168 residual components form a tight
isotropic blob; sorting collapses it to a smooth curve; the sorted logits trace a clean
closed cardioid. Those are properties of plotting a distribution in polar coordinates.

**The fourth panel is not geometry.** The logits, plotted by vocabulary index, are visibly
asymmetric: the negative cloud sits offset and the positive points form a wedge pointing
at low token IDs. Measured rather than eyeballed:

```
index bucket           mean    frac>0     max    n>10
     0- 10240       -0.2744    0.3699  15.0213     74
 20480- 30720       -1.2306    0.2901  13.7461     10
 40960- 51200       -1.7756    0.2242  10.9803      1
 81920- 92160       -2.3549    0.1687  10.6642      1
122880-133120       -2.6423    0.1313   8.5168      0
153600-163840       -2.8006    0.1116   9.3550      0

positive logits          : 32,539 of 163,840, median index 56,581 against 81,920 if uniform
top 100 logits by index  : 78 in the first 10% of the vocabulary, 36 in the first 1%
first half vs second half: mean -1.4853 against -2.6042, frac>0 0.2587 against 0.1385
```

Monotone across all sixteen buckets. **Control:** the residual has no index structure at
all, `frac>0` of 0.4947 against 0.5033 for its two halves, so this belongs to the
vocabulary axis and is not an artifact of the plotting.

Low token IDs in a tiktoken vocabulary are the most frequent tokens, so this is a
frequency prior, measured on this model rather than assumed.

**Why it is recorded here.** Seven cycles of hypothesis-driven measurement never examined
the vocabulary dimension, because no hypothesis pointed at it. One picture drawn with no
expectation showed it immediately, and the measurement then confirmed it. That is a fact
about the method, not about the model: **choosing what to measure was the binding
constraint on this work far more often than any difficulty in measuring it.**

Whether it is useful is a separate and untested question. The tail is 1.68% of runtime, so
a vocabulary shortlist would save little here even if it were exact, and it would not be
exact without a bound of the kind Step 27 already ruled out.

---

## Step 3 — are the weights generated, or do they have to be stored?

The previous section ended on the method point that choosing what to measure was the
binding constraint. The human applied it directly: plot each **row** of a weight matrix as
a point, and see whether the rows lie on something an equation could produce. If they did,
the row would not need to be read at all. That attacks the I/O problem at its root rather
than trimming it.

Each row is projected onto two fixed random unit directions and drawn as one point.
Rendered at `/root/k3raw/rows.png`, four panels:

```
expert 1/0 w1 rows, random projection   n=  3,072   x -0.0887..0.0839  y -0.1016..0.0812
W_gate rows, layer 0                    n= 33,792   x -0.1464..0.1222  y -0.1219..0.1416
lm_head rows, random projection         n=163,840   x -0.1362..0.1285  y -0.1361..0.1254
lm_head rows, norm against row mean     n=163,840   x  0.7103..3.5805  y -0.0015..0.0014
```

**No curve, no lattice, no manifold, in any of the four.** Three panels are isotropic
clouds centered on the origin. The answer to the question as asked is no.

### The one panel that is not a cloud, and what it turned out to be

The fourth panel is a **cone**: row norm on x, row mean on y, with the spread of the mean
widening as the norm grows. That is a relation, and it is exactly testable. If a row's
components behave as independent draws, then $\operatorname{std}(\text{mean}) =
\text{norm}/n$ with $n = 7168$. Measured in ten equal-count norm buckets:

```
norm bucket        n        std(mean)      norm/n     ratio
0.71-1.98       16384       2.299e-04    2.467e-04    0.932
1.98-2.12       16384       2.743e-04    2.879e-04    0.953
2.12-2.20       16384       2.942e-04    3.017e-04    0.975
2.20-2.25       16384       3.037e-04    3.107e-04    0.977
2.25-2.31       16384       3.088e-04    3.181e-04    0.971
2.31-2.35       16384       3.143e-04    3.250e-04    0.967
2.35-2.40       16384       3.253e-04    3.319e-04    0.980
2.40-2.46       16384       3.376e-04    3.394e-04    0.994
2.46-2.54       16384       3.408e-04    3.488e-04    0.977
2.54-3.58       16384       3.607e-04    3.670e-04    0.983

per-row kurtosis : mean 3.6311  median 3.1288  min 2.2093  max 186.8092   (3.0 = Gaussian)
rows within 2.8..3.2 : 63.18%
```

The cone is the independence relation, to within 2 to 7 percent in every bucket, and most
rows are close to Gaussian. So the visible structure is a statement that the rows have
**no** structure beyond their scale.

One quantity in the same run must **not** be read as a finding: $\text{norm} /
(\text{std}\sqrt{n}) = 1.000069 \pm 0.000098$. That is algebra, not evidence. For any
vector, $\text{norm}^2 = n(\text{std}^2 + \text{mean}^2)$, and the means here are $O(10^{-4})$,
so the identity holds by construction and would hold for arbitrary data.

### A trap, recorded because the script walked into it

The measurement script ended by printing its own conclusion:

> if rows were iid Gaussian given the norm, the only row-level parameter is the norm
> itself: 163,840 numbers instead of 163,840 x 7,168 — 0.66 MB instead of 2.35 GB

**This is wrong**, and it is the same error `k3-redundancy.md` named in Cycle F. A row's
*distribution* being Gaussian does not make the row interchangeable with another sample
from that distribution. Substituting one would destroy the model. What is compressible is
the *description of the ensemble*; what has to be stored is *which member*. A 3,500x
compression figure appeared in output that had been produced honestly, and it was an
artifact of confusing those two things.

### The prediction, and the test that settles it

Stated correctly, the finding makes a hard prediction rather than offering a saving.
Values that are statistically featureless carry **maximum entropy for their variance**, so
the stored bytes should be close to incompressible. That is decidable, and it is the first
item on the "not yet examined" list below.

Expert 1/0 `w1`, 11,010,048 four-bit E2M1 codes:

```
code   0      1      2      3      4      5      6      7
p    .05768 .10991 .09553 .07629 .07658 .05270 .02591 .00550
code   8      9     10     11     12     13     14     15
p    .05765 .10989 .09552 .07620 .07652 .05265 .02593 .00554

entropy                      3.7595 bits per code   (4.0 = incompressible)
entropy floor for the tensor 5.17 MB of 5.51 MB     ratio 0.9399
zlib level 9                 5,505,024 -> 5,219,359   ratio 0.9481   0.1 s
lzma preset 6                5,505,024 -> 5,239,072   ratio 0.9517   0.7 s
scale bytes                    344,064 ->    40,005   ratio 0.1163
distinct scale byte values   11 of 256
```

**The prediction held.** 3.7595 bits of 4. Both general-purpose compressors land *above*
the entropy floor, so neither finds structure the histogram did not already account for.

The code histogram is symmetric between each code $c$ and $c+8$ to four decimal places, and
$p(\text{sign}) = 0.4999$, so the sign bit is a full incompressible bit by construction and
the three magnitude bits carry 2.76 of 3. The scales do compress, 8.6x, but they are only
5.9% of the bytes. Combined best case for the tensor pair is
$5{,}259{,}364 / 5{,}849{,}088 = 0.899$, a **10.1% saving**, and only if decompression can
outrun 2.69 GB/s.

### What Step 3 settles

The weights cannot be generated, and they cannot meaningfully be compressed. Those are the
two ways to avoid reading bytes without reading fewer of them, and both are now closed by
measurement rather than by argument. Applied to the 144.72 GB of Cycle G, the entire
compression avenue is worth about 5 s of 53.8 s, against the 2.5x already obtained in
Step 2 by reading fewer experts.

**The remaining lever on the data problem is selection, not representation.**

### What Step 3 does not establish

- **One expert, one tensor.** Expert 1/0 `w1`. Whether all 92 x 896 experts share the
  histogram is untested, though the architecture gives no reason for them to differ.
- **Two compressors.** zlib and lzma. A codec built for this data could beat the order-0
  entropy figure if higher-order structure exists; nothing here rules that out, it only
  shows the two standard tools find none.
- **Row projection is two directions.** Structure orthogonal to both random directions
  would be invisible in the plot. The kurtosis and bucket statistics are computed on the
  full 7,168 dimensions and do not share that limitation.

---

## Not yet examined

- whether the 17.55 MB per expert can be reduced without changing the result
- whether expert choice is predictable early enough to prefetch
- whether reuse across many tokens changes the picture, since all of this is 5 tokens
