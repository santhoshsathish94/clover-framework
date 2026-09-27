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

> **Corrected in Step 4 below.** The conclusion holds, the rate does not. 2.69 GB/s is
> what this access pattern gets from the device, not what the device gives: a sequential
> pass over the whole 1,446 GB model runs at 8.56 GB/s on the same disk.

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

- **One expert, one tensor.** Expert 1/0 `w1`. ~~Whether all 92 x 896 experts share the
  histogram is untested, though the architecture gives no reason for them to differ.~~
  **Struck 2026-09-27.** "The architecture gives no reason for them to differ" is an
  assumption, written to excuse an n=1 result in a document whose subject is that
  assumptions get measured. Step 4 measures all 247,296 of them. The conclusion survives;
  the reasoning offered for it here was not evidence.
- **Two compressors.** zlib and lzma. A codec built for this data could beat the order-0
  entropy figure if higher-order structure exists; nothing here rules that out, it only
  shows the two standard tools find none.
- **Row projection is two directions.** Structure orthogonal to both random directions
  would be invisible in the plot. The kurtosis and bucket statistics are computed on the
  full 7,168 dimensions and do not share that limitation.

---

## Step 4 - reading the whole model instead of a sample of it

Every number in this document up to here came from one prompt of five tokens, and Step 3
came from a single tensor. The objection is that this describes one run, not the system,
and that a model whose experts and layers are all adjustable during training may well be
**uneven** - so a divergence between its parts is not noise to be averaged away, it is the
only place where something correctable could show itself.

That is an argument for reading all of it. The model is 1.56 TB in 96 files, 93 layers,
896 experts, 494,592 expert tensors. So all of it was read, twice.

### The first result arrived before any analysis

```
pass 1  494,592 tensors  1,446.46 GB  169.0 s   8.56 GB/s   (code histograms)
pass 2  247,296 tensors  1,361.37 GB  182.9 s   7.44 GB/s   (zero row/column census)
```

The machine has 124 GB of RAM, so at least 1,322 GB of pass 1 came off the disk, giving a
floor of 7.8 GB/s however the page cache behaved.

**Cycle G recorded that the model was I/O bound at 2.69 GB/s, and that 144.72 GB at that
rate is 53.8 s, which is the whole runtime.** The arithmetic was right and the conclusion
was wrong. 2.69 GB/s is not the disk. It is what *this access pattern* gets from the disk:
scattered per-expert reads in routing order, against the sweep's sequential pass in offset
order. The device has roughly 3x more to give, and the gap is in how the reads are issued,
not in the hardware.

This was sitting in plain view the whole time and was never checked, because "I/O bound"
felt like an endpoint. It is not an endpoint until the device rate is measured separately
from the program's rate.

### The Step 3 entropy result generalizes

All 247,296 packed tensors, 2.72 x 10^12 codes:

```
entropy   mean 3.75383   std 0.004535
          min  3.39684  (layer 12, expert 821, w2)
          max  3.81131  (layer 83, expert 263, w3)
          q0.001 3.74793   q0.5 3.75331   q0.999 3.78253

the Step 3 sample, layer 1 expert 0 w1 : 3.75946
tensors within 0.01 bits of it         : 208,785 of 247,296  (84.43%)

role      n        entropy            zero frac   sign+
w1    82,432   3.75433 +- 0.00517      0.11506   0.499995
w2    82,432   3.75374 +- 0.00418      0.11738   0.500000
w3    82,432   3.75341 +- 0.00412      0.11499   0.499999
```

The weights are incompressible everywhere, not just in the sample. The sign bit is a full
bit to six decimal places in all three roles. Across layers the mean entropy varies by
0.015 bits, between 3.74954 at layer 71 and 3.76476 at layer 83.

### The divergence

The distribution is a tight bulk with something else attached. The 0.1st percentile is
3.74793 and the minimum is 3.39684, which is 79 standard deviations below the mean.

```
beyond   3 sigma :   65 low    3,385 high
beyond   5 sigma :   38 low      680 high
beyond  10 sigma :   13 low        8 high
beyond  20 sigma :    5 low        0 high
beyond  50 sigma :    1 low        0 high
```

**All 38 of the low outliers are `w2`.** None is `w1`, none is `w3`, out of 82,432 of each.
They sit in layers 1 (14), 24 (11), 12 (6), 4 (4), and one each in 3, 5, 6.

The mechanism is excess zeros. E2M1 encodes zero twice, as code 0 and code 8, and in the
extreme tensor both sit at 4.01x the model mean, 46.4% zeros against a typical 11.5%.
The other codes are depressed but not uniformly: the ratio to the model mean runs 0.562,
0.549, 0.563, 0.600, 0.696, 0.878, **1.065**, so the largest magnitude is enriched. The
tensor is not scaled down, it is hollowed out.

### What the zeros are

`w2` is `down_proj`, shape (3584 out, 3072 in), so its columns are the intermediate
neurons. Counting fully-zero rows and columns in every packed tensor - which needs no
decoding, since codes 0 and 8 are zero at any scale:

```
role       n      tensors w/ dead cols   dead cols   dead rows
w1    82,432                        0           0           0
w3    82,432                        0           0           0
w2    82,432                    6,524     162,022           0
```

Zero, not few. A column is either normal or entirely zero, with nothing in between: in the
worst tensor the surviving columns have median zero fraction 0.1104 against a control's
0.1141.

This is not quantization luck. A dead column is zero in all 3,584 rows, and each row has
its own block scale set by 31 other columns; at the observed 11.5% zero rate the
probability is 10^-3366.

### Why only w2 - and the correction it forces

The clean w1/w3 census reads as an asymmetry, and it is an artifact of how the census
asks the question. Comparing the decoded magnitudes of the same neurons in the same expert:

```
                 dead   w1 row mean|w|  dead / live      w3 dead / live
L12 E821        1,071   1.32e-04 vs 2.00e-02    0.007            0.007
L4  E478          921   3.65e-05 vs 2.07e-02    0.002            0.002
L24 E17            54   3.34e-04 vs 1.96e-02    0.017            0.017
```

The neuron is dead in **all three** matrices, 140x to 500x below its live neighbors in
`w1` and `w3` too. MXFP4 shares one scale per 32 consecutive elements along the last axis.
In `w2` the last axis is the neuron index, so a dead column sits among 31 live neighbors
whose magnitudes set the scale, and it rounds to exact zero. In `w1` and `w3` the neuron
index is the row, so a dead row's blocks contain only its own tiny values, the scale
adapts down, and the noise survives at full 4-bit resolution.

So the model spends four bits per element storing the numerical noise of neurons that
cannot affect its output, and the zero-census could not see it.

### Where the dead neurons are

```
experts with at least one dead neuron : 6,524 of 82,432   (7.91%)
total dead neurons                    : 162,022 of 253,231,104   (0.064%)
among affected experts                : median 6, q90 70, q99 232, max 1,071

layer      dead   experts   % of layer        worst single experts
24       53,877       871       1.957%        L12 E821  1,071 of 3,072  34.86%
 4       30,378       802       1.104%        L4  E478    921           29.98%
 1       21,521       659       0.782%        L1  E275    624           20.31%
 3       14,160       678       0.514%        L1  E655    544           17.71%
21        8,209       221       0.298%        L12 E846    529           17.22%
 2        7,217       371       0.262%        L4  E517    463           15.07%
12        5,387        99       0.196%        L24 E628    437           14.23%

layers 52-92 : 0 dead neurons across 36,736 experts
```

The hard edge at layer 51 is the most striking thing in the census and this document
offers no explanation for it. It is recorded as an observation.

### A second correction, caught by its own control

The first pass at "are the same neurons dead across experts" compared the observed maximum
sharing count against the Poisson **mean**, and layer 1's 18 against 7.0 looked like 2x
clustering. The right comparison is the expected **maximum** of 3,072 draws. Against a
permutation null of 2,000 size-preserving shuffles:

```
layer  experts  obs max  null max  null sd      z
 1         659       18     17.57     1.19   0.36
 4         802       22     22.48     1.37  -0.35
12          99        8      7.36     0.67   0.95
21         221        9      9.85     0.86  -0.99
24         871       36     33.79     1.70   1.30
```

**Indistinguishable from random**, every layer, and the same for contiguity at layer 24,
0.0212 adjacent pairs against 0.0201 expected. Which neurons die is random; how many die,
and in which layer, is not. This is instrument error #5's shape a third time - the
instrument reporting a property of itself - and it was caught only by asking what the
control was actually a control for.

### What Step 4 settles, and what it is worth

The dead neurons are the first thing in this entire body of work that is **exact by
construction rather than by measurement**. The `w2` column is exactly zero, so the
neuron's contribution to the sum is exactly zero, so removing it cannot change any bit.
Every earlier candidate had to be checked against output and most of them failed.

It is also the smallest. 162,022 neurons at 5,376 wasted bytes each - a `w1` row, a `w3`
row and a `w2` column - is **871 MB of 1,361 GB, or 0.064%**. As an answer to the data
problem it is nothing.

As a statement about the model it is not nothing. Layer 24 has 1.96% of its expert
capacity in neurons that cannot affect the output, spread across 871 of its 896 experts,
and layers 52 to 92 have none at all. Whether correcting that would produce a better model
is exactly the kind of question that cannot be answered by looking at a trained artifact,
and nothing here answers it.

### What Step 4 does not establish

- **No cause for the layer-51 edge.** The profile is measured; the reason is not.
- **Nothing about training.** These are properties of a finished, quantized artifact.
  That a neuron is collapsed in the deployed model does not establish when or why it
  collapsed, and this work has no access to the original weights or to the run that
  produced them.
- **Whether it can be fixed is untested and untestable here.** Retraining is not available,
  so "a divergence we could correct" remains a possibility the census supports and does not
  demonstrate.
- **The 2.69 vs 8.56 GB/s gap is measured, not exploited.** Two different access patterns
  were measured on the same device. That the model could be made to read at the higher rate
  is a hypothesis, not a result.

### One thing it does change about everything above it

Every previous finding in this document is conditional on one prompt of five tokens. Step 4
is the first that is not: it is a complete census of the artifact, with no sampling, so
there is no prompt it could fail to generalize to. The objection that produced it - measure
how the system works, not how it works for five tokens - was the correct objection, and the
two corrections it forced were both to claims already committed.

---

## Step 5 - acting on the gap

Step 4 measured a device that gives 8.56 GB/s and a model that gets 2.69 GB/s from it. That
is a hypothesis about the access pattern, not a result, so it was implemented.

### What the pattern actually was

`eq.c` reads every expert weight through mmap demand paging, from inside the arithmetic:

```c
for (int j = 0; j < TOPK; j++) {                        /* 16 experts, serial */
    X(eg, zl, file_ptr(p1->file_id) + p1->off, ...);    /* faults 5.5 MB */
    X(eu, zl, file_ptr(p3->file_id) + p3->off, ...);    /* then 5.5 MB more */
    situ(eg, eg, eu, I_);
    X(ed, eg, file_ptr(p2->file_id) + p2->off, ...);    /* then 5.5 MB more */
}
```

Every byte arrives as a 4 KB fault with the kernel's default readahead, the faults happen
between multiplies, and the sixteen experts are a serial dependent chain. There is
essentially one read stream in flight. The sweep reached 8.56 GB/s by having sixteen
threads issue 5.5 MB reads with no arithmetic in between.

### The change

The router needs only `x2b[t]`, which step (5) of the layer already computes for every
position before any expert runs. So every byte a layer will read is knowable before its
first multiply. The MoE block was split into three phases: route all positions, prefetch
the union of their experts sorted by (file, offset) with all threads faulting at once,
then run the original arithmetic untouched. Behind `K3_PREFETCH` so the control runs on
the same binary. Source: `/root/k3raw/eqp.c`; `eq.c` was left alone.

### Result

Cold page cache before every run, `drop_caches` between each:

```
configuration                      runs   wall time (s)          mean     vs off
prefetch off                          4   55.24 52.92 53.10 52.80 53.52         -
madvise WILLNEED only, async          2   49.51 50.02             49.77     1.08x
sorted + parallel touch, experts      5   42.47 41.31 42.27 42.11 42.87  42.21  1.27x
  the same, plus trunk slots          2   44.83 42.37             43.60     1.23x
```

**1.27x, and every one of the fifteen runs is bit-identical**: 163,840/163,840 logits and
7,168/7,168 final-norm floats matching the pre-change baseline, maximum absolute
difference exactly 0, token 17374 throughout. `read_bytes` is unchanged at 158.2 GB, so
the same bytes move; only the manner changed.

### Three things the experiment settled that argument would not have

**The device claim is confirmed from inside the running model.** The prefetch phase pulls
99.72 GB at 7.46 to 7.93 GB/s, and 154.19 GB at 8.42 GB/s in the trunk variant. That is
the sweep's rate, reached by the model itself, on the data the model actually needs.

**Asynchronous readahead alone is not enough, and is worse than doing the work.**
`MADV_WILLNEED` returns in 0.96 s because it only queues the request; the kernel's
readahead then fails to keep ahead of the compute and 49.77 s is all it buys. Explicitly
driving sixteen fault streams and waiting costs 12.8 s of visible barrier and wins 7.5 s
more. The version that blocks is faster than the version that does not.

**Prefetching the trunk is worth nothing**, 43.60 s against 42.21 s. Its 58 GB was already
overlapping with compute, and hoisting it into the barrier only relocated the cost. Added,
measured, removed.

### What this does to Cycle G

Cycle G's outcome was "the model as implemented is I/O bound, not compute bound", and it
re-priced the preceding six cycles on the grounds that operations are not the binding
resource. Both halves need revising:

- **The qualifier "as implemented" was carrying the whole claim** and was not noticed.
  A different implementation of the same equation, producing the same bits, is 1.27x
  faster without reading fewer bytes.
- **The run is no longer I/O dominated.** 12.8 s of 42.87 s is the I/O barrier, 30%. The
  other 70% is arithmetic plus trunk transfer that already overlaps it. Operations are
  back to being the majority cost, so the re-pricing Cycle G applied to cycles A through F
  is itself withdrawn. Those cycles still failed, on exactness, but not for the reason
  Cycle G retrospectively gave.

### What Step 5 does not establish

- **One prompt, prefill only.** Five tokens, the decode path untouched and unmeasured.
- **42.21 s is not a floor.** The barrier is pure serialized waiting. Overlapping it with
  arithmetic would need the next layer's routing, which depends on this layer's output, so
  cross-layer lookahead is not available; within-layer overlap was not attempted.
- **The 70% is not cleanly attributed.** It is compute plus whatever trunk I/O overlaps
  it, and this run carries no section timers to separate them.
- **Machine-specific in part.** One disk, 16 threads, 124 GB of RAM.
- **Nothing here is a change to the model.** The equation, the weights and every emitted
  bit are identical. This is a change to how bytes are fetched, and it is the only result
  in this document that made the model materially faster while changing nothing at all.

---

## Step 6 - what the hardware actually gives, measured rather than inferred

Two claims were put to this work: the array is a mirror and should do about 20 GB/s, and
RAM at about 47 GB/s could hold the trunk, which the code is not doing. Both are
measurements. Neither was argued.

The machine: RAID1 mirror of two KIOXIA KCD8XRUG1T92 NVMe drives, Ryzen 9 7950X3D,
4 x 32 GB DDR5 running at 3600 MT/s. Both array members carry `max_sectors_kb=128`.

### The disk

Raw `/dev/md2`, O_DIRECT, so the page cache is not involved at all:

```
threads (1 MB blocks)          block size (16 threads)
 1    3.94 GB/s                 128 KB   10.37 GB/s
 4   11.98                      256 KB   11.23
 8   13.45                      512 KB   12.48
16   13.36                        1 MB   13.35
32   13.32                        4 MB   14.08
64   13.34                       16 MB   14.04
```

Per member, run concurrently and confirmed overlapping by timestamps, `nvme0n1` gives
6.87 GB/s and `nvme1n1` 6.86 GB/s, for 13.73 GB/s together. That is two drives at the
KIOXIA CD8's rated ~7 GB/s. **The ceiling is about 14 GB/s, and md does read both
mirrors. It is not 20 GB/s**, and no thread count or block size reaches it.

The number that matters more, same file and same bytes, 16 threads:

```
            4 MB      1 MB
O_DIRECT   14.44     13.69  GB/s
buffered    8.79      9.35  GB/s
```

**The page cache costs 1.6x.** Step 4's sweep reached 8.56 GB/s and was read as evidence
of headroom in the model. It was not: 8.56 is the buffered ceiling. Every byte the sweep
and the prefetch move goes through the page cache, and that path tops out near 9 GB/s no
matter how it is driven. The remaining 1.6x is reachable only by bypassing the cache.

### The RAM, and the trunk

```
threads    read        copy (both directions)
 1      39.59 GB/s     26.02 GB/s
 4      47.80          31.28
 8      46.47          30.03
16      45.45          29.57
32      45.33          28.91
```

47.80 GB/s. The 47 GB/s figure is right.

Is the trunk in it? `mincore` immediately after a run says the 54.47 GB trunk is **91.07%
resident**, 49.61 GB. So it does end up in RAM - but nothing puts it there. It is mmapped
and arrives one demand fault at a time while the arithmetic waits, which is why
`read_bytes` is 158.20 GB: 99.72 GB of experts plus the 54.47 GB trunk, fetched once,
scattered, on the buffered path.

Loading it instead - one O_DIRECT pass into anonymous memory, 4 MB chunks, 16 threads,
behind `K3_TRUNKRAM`:

```
                     inner wall (s)        process total (s)
trunk via mmap    42.77 41.99  -> 42.38   45.33 44.55  -> 44.94
trunk into RAM    36.06 34.21  -> 35.14   43.64 41.78  -> 42.71

the load itself   54.47 GB in 3.73 s = 14.59 GB/s, O_DIRECT
```

Bit-identical, 171,008/171,008, token 17374.

The load runs at **14.59 GB/s, the full device rate, inside the real program** - which is
the disk answer demonstrated rather than benchmarked. It removes 7.2 s from the run for
3.7 s of up-front cost, so the honest figure is the process total: **44.94 s to 42.71 s**,
1.05x. Modest, because the trunk was only ever read once anyway; what changed is that it
now arrives at 14.59 GB/s in one pass instead of trickling in under the compute.

A side effect worth recording: the expert prefetch got faster too, 7.43-7.71 GB/s before
and 7.99-8.41 GB/s after, because the trunk is no longer competing for page cache.

### What this says about Step 5

Step 5's prefetch reached 8.12 GB/s and that looked close to the 8.56 GB/s sweep, so it
read as nearly done. Against the buffered ceiling of ~8.8 GB/s it is nearly done. Against
the device it is not: the experts are 99.72 GB, which is 12.5 s at 8 GB/s and 6.9 s at
14.5 GB/s. **About 5.6 s is still sitting behind the page cache**, and reaching it means
reading experts with O_DIRECT into an arena rather than faulting them through mmap.

### What Step 6 does not establish

- **The 20 GB/s figure is not reproduced here.** Six thread counts and six block sizes on
  the raw array, plus both members individually, all cap near 14 GB/s. If 20 GB/s was seen
  on this box it was under conditions this test did not reproduce, and I cannot say which.
- **O_DIRECT for the experts is costed, not built.** The 5.6 s is arithmetic from measured
  rates, not a measured run.
- **One prompt, prefill only**, as everywhere above.
- **The trunk gain is small and partly bookkeeping.** 7.2 s comes out of the timed region
  and 3.7 s goes back in before it; only the 2.2 s difference is real.

---

## Step 7 - accounting by the equation instead of by the layer loop

Everything above attributes cost to the 93-layer loop, which is the implementation's shape.
The equation has a different shape: ten primitive operators in
`k3-model-equation.md` section 2, composed by section 5. Instrumenting those instead - plus
the router dot product, which section 4.2 is explicit is *not* one of the ten - gives a
different and much sharper picture.

Two runs, cold cache, identical binary. **A** has the trunk in RAM and experts prefetched,
so nothing stalls. **B** has neither, so operator time includes the fault. The difference
is the stall, per operator.

```
operator                  A compute   B total     stall     calls
X   mxfp4 expert proj       14.625     39.067    24.442    22,080
Q   int8 projection          5.236     11.793     6.557     5,795
SiTU + sigma                  0.496      0.467         -     7,825
B   bf16 lm_head              0.409      0.438         -         1
router dot product            0.159      0.169         -       460
D, AR, C, N, L, SA, top-k     0.221      0.232         -     5,596

A:  compute 21.14 + transfer 12.40 + 0.29 = 33.83 s      37,552 major faults
B:  compute 21.14 + stall    31.00 + 0.69 = 52.86 s     520,553 major faults
```

**Two operators are the whole model.** $\mathbb{X}$ is 43% of wall and $\mathbb{Q}$ is 15%.
The other eight together are 1.3%. $\Delta$, the KDA delta-rule that took three cycles of
earlier work to get right, is 0.18%. SA is 0.00%.

Prefetching is what converts 31.0 s of scattered stall into 12.4 s of streamed transfer -
the same bytes, one seventh the major faults.

### The two dominant operators are limited by different things

This is not visible from the layer view and would have been guessed wrong:

- $\mathbb{Q}$ moves 269.13 GB at **51.40 GB/s**, against a measured RAM ceiling of 47.80
  (Step 6). It is at the memory wall. No kernel change helps; only touching fewer bytes.
- $\mathbb{X}$ moves 129.15 GB at **8.83 GB/s** from data already resident. It is nowhere
  near the memory wall - it is compute-bound. 243 GMAC in 14.6 s on 16 threads is
  0.26 MAC/cycle/thread, because the kernel decodes an E2M1 code and an E8M0 scale per
  element into scalar double lanes.

### Does the equation use the full data loaded?

Yes. Every byte loaded is dereferenced. And then dereferenced again.

```
                                    distinct    dereferenced    ratio
trunk   (Q, router, C, N, AR)       54.47 GB       281.21 GB     5.16x
experts (X)                         99.72 GB       129.15 GB     1.30x
lm_head (B)                          2.35 GB         2.35 GB     1.00x
total                              156.54 GB       412.70 GB     2.64x
```

$269.13 / 54.47 = 4.94$, and $T = 5$. Section 5 composes the model with **positions
outermost**, so every weight in the trunk is streamed from RAM once per position. Nothing
is wasted in the sense of being loaded and ignored; it is wasted by being re-read. The
arithmetic intensity is one MAC per weight byte, the worst case for a mat-vec, which is
exactly why $\mathbb{Q}$ sits pinned at the RAM ceiling.

The only genuinely unused bytes found anywhere are $A^{\log}$, stored at width 128 with 96
read (section 6 of the equation document records this), and the 162,022 dead neurons of
Step 4 at 0.064%. Both are rounding errors against a 2.64x re-read factor.

### The complexity, measured rather than asserted

`NPOS` was made a compile-time parameter and the model rebuilt for $T = 1 \dots 5$:

```
T    wall(s)    Q GB      Q s     X GB      X s    distinct prefetched
1      8.38     53.83    1.205    25.83    3.480        25.83 GB
2     14.45    107.65    2.284    51.66    5.865        48.36
3     20.98    161.48    3.306    77.49    9.087        67.98
4     27.71    215.30    4.307   103.32   11.670        85.67
5     33.94    269.13    5.237   129.15   14.662        99.72
```

$\mathbb{Q}$ bytes are $T \times 53.826$ to five significant figures. $\mathbb{X}$ bytes
are $T \times 25.83$ exactly. Wall increments are 6.07, 6.53, 6.73, 6.23, fitting
$2.0 + 6.4T$.

**The implementation is already $O(T)$ in time.** It is not quadratic, and a claim that it
is would not survive this table. What is true, and is the real defect, is that **the
constant is the entire model**: one position costs a full pass over the trunk. Distinct
expert bytes do saturate as positions share experts - 99.72 GB against the 129.15 GB that
independent selection would need, 77% - but the trunk factor is exactly $T$ with no
saturation whatsoever.

Space is $O(W)$, not $O(1)$: 54.47 GB of trunk held resident plus a layer's experts
prefetched.

### What is removable, and what is not

The $T\times$ trunk factor is pure redundancy and removing it is bit-exact by construction,
because each output element $y_{t,o}$ is an independent reduction over $i$ - batching
positions changes which loop is outermost and no reduction order at all.

The expert factor is subtler. Section 4.2 fixes the accumulation as "float32,
expert-major - one expert fully summed in before the next starts", and $\pi_j$ is indexed
by the position's own rank $j$. Iterating distinct experts outermost would therefore
accumulate each position's sum in the wrong order. It stays bit-exact only if each
expert's contribution is written to a per-position slot indexed by that position's $j$, and
the 16 slots summed afterwards in $j$ order. That is 5 x 16 x 3584 floats, 1.1 MB.

### What Step 7 does not establish

- **The predicted gains are arithmetic, not runs.** $\mathbb{Q}$ at 53.83 GB instead of
  269.13 GB is ~1.05 s instead of 5.24 s; $\mathbb{X}$ decoding 99.72 GB instead of 129.15
  is ~11.3 s instead of 14.66. Neither has been built.
- **$O(1)$ space and one-pass streaming are in tension.** Holding nothing means re-reading
  154 GB per forward pass, which is 10.6 s at the Step 6 O_DIRECT rate. Whether that beats
  holding the trunk depends on how many tokens amortize it, and that is untested.
- **Still one prompt.** $T \le 5$, and only $T = 5$ emits the verified token; the shorter
  runs emit 220, 318, 276, 387 and were never expected to match.

---

## Step 8 - removing the redundancy Step 7 measured

Step 7 found the defect: section 5 composes the model with positions outermost, so every
trunk weight is streamed from RAM once per position, 5.16x. The fix is to invert the loop
nest so the **weight** is outermost and all positions are applied to it while it is there.

### Why this is bit-exact by construction

Each output element $y_{t,o}$ is an independent reduction over $i$. Moving the position
loop inside the weight loop changes which loop is outermost and **no reduction order at
all** - the same 16 lanes, the same block order, the same final tree. `Qm` and `Xm` are
therefore bit-identical to calling `Q` or `X` once per position, not approximately so.

Three things could not be moved, and the equation says why:

- **The conv history and the delta-rule state carry across positions.** $\mathcal{C}$
  reads `convbuf` and $\Delta$ carries $S$, so that part stays strictly in position order.
  Only the projections feeding them moved.
- **Section 4.2 fixes the routed sum as float32, expert-major, in rank order.** Iterating
  distinct experts outermost would accumulate each position's sum in the wrong order. Each
  expert's output is therefore written into that position's rank slot $j$ and the 16 slots
  summed afterwards in $j$ order. 5 x 16 x 3584 floats, 1.15 MB.
- **A position selects an expert at most once**, so the rank slot is unambiguous.

### What it did, block by block

All measured cold-cache with trunk in RAM and experts prefetched:

```
                          Q GB    Q s    Q calls     X GB     X s   inner wall
baseline                269.13  5.237      5,795   129.15  14.625      33.83
+ MoE batched           201.55  3.870      3,955    99.72  12.713      30.78
+ KDA batched            79.05  1.752      1,747    99.72       -      30.1
+ MLA batched            56.73  1.516      1,171    99.72       -      30.2
+ threads pinned         56.73  1.341      1,171    99.72  11.90       27.51
```

The redundancy is gone. $\mathbb{Q}$ dereferences **56.73 GB against 53.83 distinct, 1.05x**,
down from 5.00x. $\mathbb{X}$ dereferences **99.72 GB, exactly the distinct bytes, 1.00x**,
down from 1.30x. Every intermediate build emitted 17374 with 171,008/171,008 identical.

### A detour that was noise, and the real cause

After the KDA change $\mathbb{X}$ appeared to regress, 12.713 to 15.550 s, on identical
bytes and identical call count, from a change in a different block. Repeating it gave
16.575, 13.300, 12.960 - so the regression was variance, not the edit.

The variance was anti-correlated with prefetch time, which looked like I/O but is not. This
is a 7950X3D: two CCDs, one with 3D V-cache and one without, and nothing was pinned.

```
unpinned   wall 28.21  28.70      X 12.724  13.117
pinned     wall 27.52  27.50      X 11.963  11.832
```

`OMP_PROC_BIND=close OMP_PLACES=cores` removes the spread and is faster. **Every timing in
this document before Step 8 was taken unpinned** and therefore carries roughly a second of
placement noise. None of the conclusions turn on a second, but the figures should be read
with that in mind.

### End to end, against the untouched original

```
                                        inner wall        process total
eq.c, as it was                       53.30  52.82 s      55.90  55.41 s
eqp.c batched + prefetch + trunk RAM  28.01  27.85 s      35.78  35.58 s
```

**1.56x end to end, 1.90x on the timed region, with 171,008/171,008 bits identical and
token 17374.** The process total includes the 3.74 s O_DIRECT trunk load.

### On O(n) time and O(1) space

- **Time.** It was already $O(T)$; Step 7 showed that. What Step 8 removed is the constant:
  a forward pass now touches each trunk weight once rather than $T$ times. The per-position
  marginal cost drops from a full pass over the model to arithmetic plus the experts that
  position uniquely needs.
- **Space is still $O(W)$, not $O(1)$, and this was not achieved.** 54.47 GB of trunk is
  held resident, plus a layer's experts. The batching added about 2 MB of per-position
  buffers and nothing else.
- **$O(1)$ space and one-pass streaming are in tension on this machine.** Holding nothing
  means re-reading 154 GB every forward pass, 10.6 s at the Step 6 O_DIRECT rate, against
  a 3.74 s one-off load that then serves every token. $O(1)$ space is the right design when
  the model does not fit; here it does fit, and paying 10.6 s per pass to avoid holding
  54 GB would be a straight loss. That is a measured trade, not a principle.

### What is left, and what it is worth

$\mathbb{X}$ is now 43% of wall and the prefetch 46%; everything else is 11%.

- **$\mathbb{X}$ is compute-bound**, 99.72 GB at 8.43 GB/s against a 47.80 GB/s RAM ceiling.
  Its four 4-lane double accumulators map exactly onto four AVX2 256-bit double registers,
  one lane per register lane, so an SIMD kernel would be **bit-exact by the same argument as
  `Qm`** - each lane keeps its own order. Not attempted.
- **The prefetch is still on the buffered path** at ~8 GB/s against 14.5 available with
  O_DIRECT, the 5.6 s costed in Step 6 and still not built.

### What Step 8 does not establish

- **One prompt, prefill only.** $T = 5$. The decode path is untouched and unmeasured, and
  at $T = 1$ batching has nothing to batch, so none of this helps decode.
- **The gain shrinks with smaller $T$ and grows with larger $T$.** It was measured at
  $T = 5$ only.
- **Layer 0's dense MLP was not batched**, which is where the residual 1.05x on
  $\mathbb{Q}$ lives.
- **Nothing about the model changed.** Same weights, same equation, same bits.

---

## Step 9 - can dead neurons reduce the fetch?

Step 4 found 162,022 dead neurons. The question put to this step was whether they, or
neurons like them, can be used to fetch fewer bytes. Three things had to be measured, and
each closed a different door.

### First, what the census actually says

The recollection driving the question was that most experts have more dead neurons than
useful ones. Re-reading `dead.tsv`:

```
experts with ZERO dead neurons : 75,908 of 82,432   92.09%
median expert                  : 0 dead
90th percentile                : 0 dead
worst single expert            : 1,071 of 3,072 = 34.86%
experts more than half dead    : 0
total dead                     : 162,022 of 253,231,104 = 0.064%
```

No expert is half dead, and 92% have none. The premise does not hold, and it is recorded
here because everything downstream would have inherited it.

### The bit-exact saving, and an error in measuring it

An analysis script printed the saving as 0.87% by dividing 871 MB, which is the whole
model, by 99.72 GB, which is what **one run** fetches. Those are not comparable: the run
touches 5,683 expert-loads of 82,432. Dumping the actual routing and joining it to the
census:

```
distinct expert loads this run : 5,683        (matches Step 1 exactly)
neuron slots fetched           : 17,458,176
of those, dead                 : 10,776 = 0.0617%
fetched experts with any dead  : 433    = 7.62%

bit-exact saving : 10,776 x 5,376 B = 57.9 MB of 99.72 GB = 0.058%
                 = 0.004 s of the 7.043 s prefetch
```

**Four milliseconds.** The dead fraction among fetched experts, 0.0617%, matches the global
0.0640%, so routing is neither drawn to nor repelled by damaged experts.

### Second, is there a population of nearly-dead neurons?

Exactly-zero is the wrong test - it only catches what quantized all the way down. The
equation gives a bound that uses weights alone. Since
$|b_1\tanh(g/b_1)\sigma(g)| \le \min(|g|, b_1)$ and $|g_j| \le \|w_1[j]\|\,\|\zeta\|$,
neuron $j$'s largest possible contribution is

$$s_j \cdot \|\zeta\|^2, \qquad s_j = \|w_1[j]\|\ \|w_3[j]\|\ \|w_2[:,j]\|$$

All three norms were computed for every routed expert, 99.72 GB decoded in 12.8 s -
exactly the run's own fetch, which checks the index. Sanity: **exactly 10,776 neurons have
$s_j = 0$**, matching the census to the unit, all through a zero $w_2$ column.

```
top  1% of neurons carry   1.49% of the total bound   (uniform would be  1%)
top 10%                   11.95%                      (10%)
top 25%                   28.12%                      (25%)
top 50%                   54.05%                      (50%)
top 90%                   92.02%                      (90%)

q0.01 = 2.49    median = 3.48    q0.99 = 4.61
```

**The bound is essentially uniform.** 98% of all 17.5 million neurons sit within 1.85x of
each other. There is no nearly-dead population. Static pruning is closed: dropping 6.6% of
neurons would forfeit 5% of the bound.

### Third, the realized activation is concentrated - and it still does not help

The bound is a worst case over all $\zeta$. The activation actually realized during the run
is a different matter, and it is strongly concentrated:

```
top 10% of neurons carry 46.64% of sum|h|      53.31% of neurons below 1e-2 x max
top 25%                  70.90%                93.87% below 1e-1 x max
top 50%                  90.51%
```

So most neurons contribute almost nothing **for a given input**. To exploit that in the
*fetch*, the selection must be made before the bytes are read, and the equation supplies a
way: $\mathrm{SiTU}$ factorizes, the second factor is bounded by $b_2 = 25$, so a small
first factor bounds $h_j$ regardless of $u$ - and the first factor needs only $w_1$.

Ranking by that $w_1$-only gate factor:

```
keep top  10% by gate ->  36.67% of sum|h| ; frees 71.85% of w3 blocks
keep top  25%         ->  58.84%           ; frees 40.66%
keep top  50%         ->  82.54%           ; frees 11.60%
keep top  75%         ->  95.72%           ; frees  1.39%
keep top  90%         ->  99.32%           ; frees  0.09%
```

The predictor works reasonably - 82.54% of the realized magnitude from half the neurons,
against 90.51% for an oracle ranking by $|h|$ itself. **The layout is what kills it.**

### Why the layout closes it

- A $w_3$ row is $3584/2 = 1792$ bytes and O_DIRECT reads 4096-byte blocks, so one block
  spans 2.29 rows. A skipped row only saves anything if every row sharing its block is
  also skipped, and a gate-ranked selection is scattered. At keep-50% that frees **11.60%**
  of $w_3$'s blocks, not 50%.
- $w_3$ packed is 5,505,024 of the expert's 17,547,264 bytes, **31.4%**. So keep-50% frees
  $0.116 \times 0.314 = 3.64\%$ of expert bytes: 3.63 GB, **0.26 s of the 7.043 s
  prefetch**, in exchange for losing 17.5% of the activation magnitude.
- **$w_2$ columns cannot be selectively fetched at all.** The tensor is row-major and a
  column is maximally strided, so neuron $j$'s column is 3584 nibbles scattered across the
  whole 5.5 MB. There is no block to skip.

### What Step 9 settles

Three doors, all closed by measurement:

| route | result |
|---|---|
| skip exactly-dead neurons | bit-exact, 0.058% of the fetch, 0.004 s |
| static pruning by weight norms | bound is uniform; no population to prune |
| dynamic pruning by the $w_1$ gate | activation is concentrated, but blocks do not free and $w_2$ columns are unfetchable |

The activation sparsity is real and large - half the neurons carry 9.5% of the magnitude -
and **none of it is reachable through the fetch**, because the unit of I/O is a 4096-byte
block and the unit of sparsity is a scattered row. It would pay in a compute-bound
implementation. This one is not, in that part.

### What Step 9 does not establish

- **Nothing was implemented.** No pruned run was executed, so no token or margin was
  measured; the trade is costed from the retention and block figures, not observed.
- **One prompt, prefill only.** Activation sparsity is input-dependent by definition, and
  5 positions of one prompt is the weakest possible basis for it.
- **A layout change was not considered.** Storing $w_2$ transposed, or $w_3$ padded to
  4096-byte rows, would change the block arithmetic completely. That is a different
  artifact, not this one.
- **The gate factor is one predictor.** A better one exists in principle; the oracle at
  keep-50% is 90.51% against the gate's 82.54%, so the headroom from a perfect predictor
  is 8 points and does not change the block result at all.

---

## Step 10 - could a different disk layout make the sparsity fetchable?

Step 9 closed the fetch route on layout: 1792-byte rows against 4096-byte blocks, and
`w2` columns unfetchable. The obvious follow-up is to change the layout, and the obvious
resource is the second disk.

### There is no second disk

```
nvme0n1p4 ──┐
            ├── md2 (RAID1) ── ext4 ── /     1.8T, 1.7T used, 24G free
nvme1n1p4 ──┘
```

Both drives are fully partitioned and **every partition is a RAID1 member**: p1 is the EFI
pair, p2 is swap, p3 is `/boot`, p4 is the root filesystem holding the 1.5 TB model. Using
"one disk" means failing a member out of the array that carries the only copy of that data.
That is destructive, not quickly reversible, and it was not done.

It is also unnecessary, because there is a cheaper question that decides the whole idea.

### The question that decides it

A static file layout can only help if the set of neurons to skip is **the same across
inputs**. If it changes per token, no ordering of bytes on disk can make the skipped set
contiguous, because the skipped set is not a property of the file.

Measured on the experts that more than one position routes to:

```
experts used by >1 position : 1,274
position pairs compared     : 2,174
bottom-half gate sets share :  56.90%
two independent halves would: 50.00%
```

Two independent random half-sets of 3072 overlap 50% by construction, so **56.90% is 6.9
points above chance**. Which neurons an expert switches off is very nearly a property of
the input, not of the expert.

### What the best possible layout would buy

Rather than argue from that number, it was measured. Neurons were ordered by their **mean**
gate factor across the very positions being tested - in-sample, so an upper bound on any
real static ordering - and the 4096-byte blocks each position's own skip set frees were
counted:

```
arbitrary order (Step 9)  : 11.60% of w3 blocks freed at keep-50%
best static order         : 27.02%          (in-sample upper bound)
```

Reordering roughly doubles it. The payoff:

$$0.2702 \times 0.314 = 8.5\%\ \text{of expert bytes} = 8.46\ \text{GB} = \mathbf{0.60\ s}
\ \text{of the 7.043 s prefetch}$$

for a 17.5% loss of realized activation magnitude, out of sample worse than that, and
requiring the whole 1.45 TB model to be rewritten - against 24 GB of free space.

### Padding rows to the block size costs more than it saves

The other way to fix the granularity is to pad each `w3` row from 1792 bytes to 4096 so
that one skipped row frees exactly one block. The arithmetic closes it without an
experiment:

```
expert now                  3 x (5,505,024 + 344,064)          = 17,547,264 B
w3 padded to 4096 B/row     3072 x 4096 = 12,582,912  (was 5,505,024)
expert padded                                          = 24,625,152 B  = 1.403x
skip 50% of padded w3 rows  frees 6,291,456 B
bytes actually fetched      24,625,152 - 6,291,456     = 18,333,696 B  = 1.045x
```

**4.5% worse than changing nothing**, before counting the accuracy cost. Padding buys
block-alignment by inflating the very thing being skipped.

### What Step 10 settles

| route | result |
|---|---|
| use the second disk | there isn't one; both are RAID1 members of the root array |
| reorder neurons statically | skip sets are 56.90% shared against a 50% null, so 27.02% of blocks at best, 0.60 s |
| pad rows to the block size | 1.045x the bytes, worse than doing nothing |

The activation sparsity measured in Step 9 is real and large. **Across three independent
attempts it stays unreachable through the fetch**, and the reason is the same each time:
the sparsity is per-input and the disk is not.

### What Step 10 does not establish

- **The static ordering is in-sample.** It used the mean gate of the same positions it was
  scored on, with at most 5 positions per expert. A held-out ordering would do worse, and
  how much worse is unmeasured.
- **Only one grouping was tested.** Padding to 2048 rather than 4096 bytes, or grouping
  neurons in pairs, would land between the two rows of the table above and was not tried.
- **The 56.90% figure is within one prompt.** Stability across genuinely different prompts
  is a different and unmeasured question, and could plausibly be lower still.
- **Nothing was written to disk and no layout was built.** The verdicts are from
  measurement of the selection behavior plus arithmetic on sizes, not from a re-laid-out
  checkpoint.

---

## Not yet examined

- whether reuse across many tokens changes the picture, since all of the routing work is
  5 tokens
