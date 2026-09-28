# x — can the next experts be known before the router runs?

**The 7168-wide state determines the top-16 experts exactly, and the map is smooth. That is
not enough to replace the computation with a table — but it is enough to run the computation
early. Feeding layer L's state to layer L+1's own gate names 56% of its experts before that
layer is reached, on an unseen prompt, with nothing fitted.**

Question: layer L's router reads layer L's own input, so the 16 experts of 896 become known
microseconds before they are needed. If they could be known a layer earlier, a machine
holding experts on disk could prefetch them and the 35% of wall clock spent waiting on SSD
would disappear. Is there anything to prefetch *from*?

Run 2026-09-25. The lookup question is closed. A second question it opened is answered at the
bottom, and that one is positive.

## Why it would matter

Experts are 93.0% of the model — 1,446.5 GB of the 1,555.3 GB. The trunk is 108.8 GB and
fits anywhere. If the 16 experts a layer needs could be named before that layer is reached,
a machine would need only the trunk resident plus a few GB of expert buffer, and the design
stops requiring 24 GB per layer of VRAM.

The measured budget is precise: one layer of lead time at ~7 GB/s is **0.308 GB, about 17
experts.** So a predictor does not need to be right, it needs to be right within about the
number of experts actually used.

## Determinism, established first

The question is meaningless unless the same state always produces the same experts.

```
replayed run, hashed tensors            841 / 841 bit-identical
five row taps, positions 0-4, six runs
  at lengths 5/12/18/40/80/160          465 / 465 identical
worst relative difference               0.000e+00
```

Nothing about the engine is stochastic. `The capital of France is` produces ` Paris` in
every run at every length.

## Four routes built on expert indices. All four fitted the prompt.

| route | inside the fitted prompt | on unseen prompts |
|---|---:|---|
| static popular set, top-193 | 80.0% | 16.4–25.0% against 21.5% random — **dead** |
| learned cross-layer co-occurrence | 27.6% vs 20.4% frequency | 4.3–13.9% — **collapses** |
| rank position within the top-16 | 29.1% vs 28.3% | not a mechanism — **dead** |
| previous position, free, no model | 31.2% | **24.8–40.4% — the only survivor** |

The co-occurrence route is the instructive one: 27.6% against a 20.4% frequency baseline is
a genuine cross-layer signal, the first found, and it still transferred at 4.3%. **Expert
index co-occurrence is a property of one text, not of the model.**

### One exception, unexplained

Pinning the 250 most-used experts at **layer 12** transfers to unseen prompts at **75.2%**
(73.8 / 81.5 / 75.0 / 70.4 on four prompts) against a 27.9% baseline. The next best layer is
49.5%, and only 3 of 92 layers beat baseline by more than 1.5×. Layer 12 draws on a narrow
set for every prompt tested — 53 to 128 distinct experts, against 387 in the prompt it was
fitted on. Snapshot layers as a class are not the cause: 30.4% against 28.3% for the rest.
Recorded as an observation. No mechanism.

## The instrument was wrong

Two corrections, both from the model's creator, both of which reversed the result:

- Look at the **7168 fractional set**, not the expert indices. Experts are selected by
  weights that must sum to 1.0, so the selection itself deviates; the state does not.
- Look at **one layer**, not all 92.

So `K3_TRACE_RAW` was added on `clover/observe` — a binary sink at `norm.pre_mlp`, the exact
vector the router reads. Record layout is `int32 layer, int32 pos, int32 cols`, then `cols`
float32.

```c
K3_TRACE_RAW("norm.pre_mlp", hin, T, E);   /* k3_ops.c, after the pre-MLP norm */
```

Gate passed before use: `make test` after the change reports **"VERDICT: ENGINE MATCHES THE
REFERENCE EXACTLY"** and all weightless tests pass. The tap observes; it does not perturb.

Two prompts traced: `v6-long-context` prose, 225 positions, 600,212,700 bytes, 454.0 s; and
a 793-byte C source file, 268 positions, 714,920,016 bytes.

### A pooled table that was an artifact, not a finding

Binning similarity against overlap across 12 layers at once produced
`2.4 / 5.5 / 20.1 / 17.3 / 36.8 / 3.8 / 20.7 %` — the 0.90–0.95 bin collapsing between
neighbors at 36.8% and 20.7%. That is not signal. The layers have cosine means from 0.304 to
0.975, so a shared bin mixes typical pairs from one layer with extreme pairs from another.
Pooling a binned statistic across units whose predictor distributions differ is invalid.

## Per layer, the map is smooth

Cosine similarity between two positions' router inputs, against how many of their 16 experts
agree. **Pairs within ±6 positions are excluded**, so this is not temporal adjacency.

```
layer   cos mean   lowQ ovl  highQ ovl    ratio  pearson
  1       0.923       2.2%       5.3%     2.40x    0.218
 12       0.975      12.9%      28.2%     2.19x    0.392
 24       0.304       2.8%      11.1%     4.02x    0.577
 36       0.330       3.1%      12.9%     4.12x    0.599
 41       0.696       4.6%      11.2%     2.44x    0.446
 48       0.697       8.8%      20.3%     2.32x    0.455
 54       0.556      20.6%      42.6%     2.07x    0.564
 60       0.548      11.6%      16.6%     1.44x    0.366
 72       0.628       6.6%      15.6%     2.35x    0.418
 84       0.793      25.5%      44.7%     1.76x    0.493
 90       0.540      23.3%      41.8%     1.79x    0.571
 92       0.404       5.1%      17.9%     3.50x    0.454
```

Positive in all twelve. The unseen code prompt reproduces it independently: r = 0.317 to
0.678, **mean 0.470**. Smoothness is a property of the model, not of one text.

Within a single prompt it is also usable — nearest neighbor on the state beats the free
baseline:

```
vote of 5 nearest states      40.4%
nearest state                 37.8%
previous position (free)      32.1%
random other position         16.0%
chance                         1.8%
```

## Across prompts it fails, in every layer

Bar set before the run: beat previous-position on an unseen prompt. Library is the prose
trace, query is the C-code trace.

```
vote of 5 nearest PROSE states     14.2%
nearest PROSE state                12.6%
previous position, same prompt     31.1%   <- the free baseline
random prose position               7.8%
chance                              1.8%
```

Loses in all 12 layers, by 4.9 to 38.5 points.

### The failure is coverage, not smoothness

```
nearest-neighbor cosine   within prompt 0.807   across prompts 0.661   gap -0.146
```

The per-layer gap predicts the per-layer failure. Layer 12, gap −0.010, scored 21.8%. Layer
92, gap −0.297, scored 3.2%. Layer 24, gap −0.278, scored 9.8%. The map is fine. The unseen
prompt simply lands where nothing has been stored.

## How much library would close the gap: about a zettabyte

Growing the library 8 → 224 barely moves coverage — layer 24 goes 0.284 → 0.371 for 28× the
states. Fitting `(1 − cos) ~ N^slope` and solving for the 0.807 that makes prediction work:

```
layer 92   slope -0.073   2.8e+08 states
layer 54         -0.036   3.84e+10
layer 90         -0.032   4.14e+14
layer 24         -0.039   3.51e+15
layer 36         -0.031   1.16e+19
median                    4.14e+14 states per layer
```

At 28,672 bytes per state that is **~12 EB per layer and ~1 ZB for the model — roughly 700
million times the size of the model itself.** The slopes put the effective dimension of the
state manifold in the tens (~27–65), which is the reason: nearest-neighbor coverage is
hopeless at that dimensionality.

> **The premise was right and survived every attempt to fault it. The 7168 → top-16 map is
> exactly deterministic and it is smooth. It still cannot be turned into a lookup, because
> the domain cannot be covered. The table that would let you skip the router is eight orders
> of magnitude larger than running the router. That is a reason, not a dead end.**

## What this means for the design

Prefetch cannot be driven by a stored map. It can be driven by the gate itself, run early.

---

# The router was never the thing to derive

The whole investigation above tried to approximate a function that is already closed form.
Read from the engine:

```
score[e]  = sigmoid( W_l[e] . x_l )     W_l = [896, 7168] I8R, 6.43 MB packed
choice[e] = score[e] + b_l[e]           b_l = 896 f32, 3.584 KB
experts   = top-16 by choice
```

One projection, an elementwise sigmoid, a per-expert bias, top-k. No recurrence and nothing
hidden. **All 92 gates together are 1.182 GB, 0.076% of the model, and they are already
resident inside the trunk.** Approximating that can only be less accurate for no saving.

The router was never the cost either: 6.4 MFLOP per layer per token, 590 MFLOP across the
model, against 206 GFLOP for the token as a whole.

**The obstacle is the dependency, not the form.** `R_l` needs `x_l`, and `x_l` does not exist
until layer `l-1` has read its experts. So the question is not what the router is, but when it
can be evaluated.

## One-layer lookahead: run the real gate on a stale state

Keep the exact gate. Feed it the previous layer's state.

```
predicted experts at L+1  =  top16( W_{L+1} . x_L + b_{L+1} )
```

Nothing is fitted. There is no theta, no training set, no library, so there is nothing that
can overfit a prompt.

**Gate passed first:** the same code driven by each layer's *own* state reproduces the traced
picks at **100.0%**, all 92 layers, all 225 positions. That validates the tensor offsets, the
I8R dequantization (`w[e][i] = int8[e][i] * scale[e]`, rows of `[f32 scale][int8 x 7168]`),
and that the `norm.pre_mlp` tap is exactly the vector the router reads.

```
GATE re-run on its OWN state (must be 100%)      100.0%
GATE of layer L+1 driven by state of L            56.9%
baseline: reuse layer L's picks for L+1            1.7%
chance                                             1.8%
```

**Consecutive layers share essentially no experts — 1.7%, at chance — while the state carries
over well enough to place 57% of the next layer's choices.** That single contrast explains
every failure above: expert indices carry nothing between layers, and the state carries
almost everything.

### The sawtooth is visible in the accuracy, as predicted

Snapshots every 12 layers **replace** the residual rather than adding to it, so the state
should jump there and drift smoothly in between. It does:

```
layer mod 12    0      3      7     11
lookahead    27.8%  55.1%  69.4%  73.1%
```

Worst at the snapshot, climbing monotonically to the layer before the next one. Predicted
from the residual structure before the run, not fitted after.

### It transfers, which nothing else did

| prefetch budget | v6 prose | unseen C code |
|---|---:|---:|
| top-16 (0.28 GB) | 56.7% | **55.4%** |
| top-24 (0.42 GB) | 65.5% | 64.1% |
| top-32 (0.56 GB) | 70.4% | 69.1% |
| top-48 (0.84 GB) | 76.0% | 74.9% |
| top-64 (1.12 GB) | 79.2% | 78.4% |
| top-128 (2.25 GB) | 85.6% | **85.3%** |

1.3 points apart at top-16 and 0.3 at top-128, on two prompts as far apart as English prose
and C source. Nothing was fitted, so nothing transferred badly.

### How fast the information decays

The useful question is not whether two or three layers back can predict the next router, but
how quickly predictive information decays with the age of the state. Driving layer L's gate
from the state of layer L-1-d, with pairs that cross a snapshot boundary excluded:

| d layers | v6 prose | unseen code | all pairs | n clean |
|---:|---:|---:|---:|---:|
| 1 | 59.3% | 58.1% | 55.6% | 18,900 |
| 2 | 48.6% | 47.0% | 44.3% | 17,100 |
| 3 | 41.9% | 40.8% | 38.2% | 15,300 |
| 4 | 36.4% | 35.8% | 34.4% | 13,500 |
| 5 | 31.3% | 31.4% | 31.4% | 11,700 |
| 6 | 26.6% | 27.1% | 28.8% | 9,900 |
| 7 | 22.7% | 23.7% | 26.6% | 8,100 |
| 9 | 17.7% | 18.8% | 25.0% | 4,500 |

**The ratio is 0.86 per layer and holds across the whole range — exponential decay, halving
every ~4.7 layers** (4.6 on prose, 4.9 on unseen code). At 32.4 vector operations per layer
that is a half-life of roughly **150 operations**. The two prompts stay within about one
point of each other at every distance, so the decay constant is a property of the model and
not of the text.

For a design this is the number that matters: lead time is buyable, but it costs accuracy at
a fixed exponential rate, and beyond about 7 layers the prediction is worth less than simply
prefetching a larger candidate set at short range.

**Unexplained.** From d >= 6 the all-pairs column *overtakes* the snapshot-free one — 25.0%
against 17.7% at d = 9. Crossing a snapshot should destroy the state. The candidate
mechanism is that a snapshot copies the residual verbatim onto the stack and AttnRes
re-injects it at every later layer, so a pre-snapshot state is still literally present
downstream; but it could equally be a selection effect in which layers survive the clean
filter at large d. Recorded as an anomaly, not a finding.

### What it costs

The gates are already in the trunk, so residency does not change. The extra work is one more
gate evaluation per layer: **590 MFLOP per token against 206 GFLOP, +0.29% compute.** At the
17-expert budget that one layer of lead time buys at ~7 GB/s, that moves **~56% of the
25.83 GB/token of expert reads off the critical path.**

Correctness is untouched. The exact router still runs and still decides; a wrong prediction
costs a cache miss, never a wrong token. This is the reason approximation is admissible here
and is not admissible for the layer output.

### What this does not establish

- **Lead time beyond one layer costs accuracy exponentially** — measured above, halving
  every ~4.7 layers. A 93-stage pipeline has a whole stage of lead and can spend it; a
  deeper split cannot.
- **Snapshot layers get little benefit** — 27.8% at `L mod 12 == 0`, one layer in twelve.
- **Two prompts, prefill positions, one generated token each.** Measured on traces, not in
  the engine: no implementation exists and no wall-clock improvement has been observed.
- The 56% figure is expert *identity* accuracy, not a measured speedup. Whether the I/O
  actually overlaps depends on the prefetch machinery, which does not exist yet.

## What this does not cover

- **Two prompts, 12 of 93 layers, one generated token each.** Prose and C source are far
  apart, which makes the negative result strong, but two points are two points.
- **The zettabyte figure is an extrapolation.** The exponents are measured over a 28× range
  of library size; the endpoint is fitted, not observed. It could be wrong by orders of
  magnitude without changing the conclusion.
- **Layer 12 is unexplained** and is the one place a cheap win may still exist.
- **No attempt was made to train a predictor on the state.** Nearest neighbor is the weakest
  possible method; it was chosen because it needs no training set and therefore cannot
  secretly fit the prompt. A learned model might do better, and would have to clear the same
  cross-prompt bar to mean anything.

**Evidence** — analysis scripts (`nn.py`, `cross.py`, `why.py`, `cov.py`, `lookahead.py`,
`ahead2.py`), both router traces, both run logs and the engine patch are in `k3routing.tgz`.
The two raw state dumps are 1.32 GB of float32 and are deliberately not in this repository;
they regenerate from the patch and the two prompt files. The gate reconstruction reads
`trunk_i8/trunk.bin` directly at `layers[L].file_off + tensor.off` and is verified by the
100.0% sanity gate.
