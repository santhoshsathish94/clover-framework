# Redundancy in K3

## What this document is

An account of what redundancy turned out to be in a model we can execute exactly,
written after measuring rather than before.

It is **not** a reduction. Nothing here makes K3 cheaper. Every candidate tested is
recorded in `CONTEXT-k3-equation-reduction.md` with its evidence, and all of them
failed. This document exists because the failures were informative in a way the
successes would not have been: they were each blocked by a *different* obstacle, and
the set of obstacles is the finding.

The reference throughout is `k3-model-equation.md`, which reproduces the engine
bit-for-bit: 79,742,816 of 79,742,816 floats on prefill, 96,587,584 of 96,587,584 with
a decode step, emitting tokens 17374 then 20829. Without an exact reference none of the
measurements below would mean anything, because every one of them is a comparison.

---

## 1. Redundancy is not a property of a computation

It is a property of a computation **and a chosen observable**. The same K3 forward pass
is almost perfectly non-redundant under one observable and substantially redundant under
another, and the three below were all measured on the same run.

| observable | what turned out to be redundant |
|---|---|
| all 163,840 logits, bit-identical | almost nothing |
| the top-1000 ordering | the final RMSNorm scalar |
| the emitted token alone | 6 of 8 snapshots individually, up to 3 of them jointly |

The final norm scalar is the cleanest illustration. Dropping it leaves the emitted token
at 17374 and the top-1000 ordering exactly intact, while **365 of 163,840 positions
reorder** from rank 4323 down. Under "the token", it is redundant. Under "the full
ranking", it is not. Both statements are true of the same operation.

This is why "intermediate values are noise" cannot be evaluated as stated. It has no
truth value until the observable is named.

---

## 2. Four kinds of redundancy, each measured

They are routinely spoken of as one thing. They behave differently.

### Representational — the same information in more numbers than necessary

The KDA state is a $128\times128$ matrix per head. Its rank is exactly $t$ after $t$
positions, confirmed by a singular spectrum with an eight-order gap between $s_t$ and the
noise floor. The state is carried in 16,384 numbers when $2 \cdot 128 \cdot t$ suffice.

Real, and provable. Not free: see section 3.

### Recomputational — the same value computed more than once

The eight snapshots are frozen when pushed, so their normalization $v \cdot \mathrm{inv}$
never changes, yet it is recomputed on every later aggregation. 4,048 of 4,974
normalizations are recomputation.

Real, and removable with no arithmetic change at all.

### Arithmetic — operations whose result is known without performing them

11.53% of decoded expert weights are exactly zero, stable to a standard deviation of
0.026% across 64 experts. Multiplying by zero and adding zero to a finite accumulator are
exact no-ops. Measured against the real denominator, this is **8.6% of the forward pass**.

Real, exact, and the largest redundancy found anywhere in this work.

### Decisional — information that does not change the answer

Six of eight snapshots can be removed individually without changing the emitted token.

Real, and the most treacherous of the four, for the reason in section 4.

---

## 3. Three obstacles, and they are not the same obstacle

Every candidate failed. What matters is that they failed differently.

### Obstacle 1 — the saving requires reassociating the arithmetic

The factored KDA state reproduces `kda.o` to 1.9e-08 and costs 15 to 77 times fewer
operations. Only **7,463 of 61,440** values are bit-identical. The backward query form of
Step 14/18 behaves the same way: correct to float32 epsilon, not to the bit.

Exactness over the reals and exactness in float32 are different properties, and a
sufficient statistic that reorders operations generally has the first and not the second.
The cost of insisting on the second was measured directly: emulating the engine's
reduction order rather than calling a matmul is **at least 12 times slower**, and that
figure is a lower bound because the BLAS it was compared against is an unoptimized
reference build running at 4.3 GFLOPS.

### Obstacle 2 — the denominator is too small

Caching the snapshot normalization is byte-identical to the engine: 163,840 of 163,840
logits, maximum absolute difference zero. It removes 81% of the normalizations, and the
aggregation is **0.04%** of measured runtime. The saving is smaller than the run-to-run
noise by roughly two orders of magnitude.

Dropping the final norm scalar saves 21,504 operations, four millionths of one percent.

A correct elimination of something that costs nothing is still nothing.

### Obstacle 3 — the redundancy is real and large but wrongly shaped

The 8.6% of exactly-zero expert arithmetic is perfectly scattered:

```
fully zero 16-element blocks : 0 of 688,128
fully zero 32-element groups : 0 of 344,064
fully zero rows / columns    : 0 / 0
zeros per row                : 346 to 497, of width 3584
random placement at the same rate predicts 9.8e-16 zero blocks; observed 0
```

No unit the kernel loads is all zero. At 88.5% density, sparse indexing costs more than
the 11.5% it saves. The arithmetic is provably unnecessary and cannot be skipped.

This obstacle is not mathematical. It is an alignment failure between where the
redundancy sits and how memory is read, and none of the reduction literature framing
anticipates it.

---

## 4. Redundancy does not compose, and ablation does not test sufficiency

Six of eight snapshots are individually removable without changing the token. Four of
those six removed together change it.

```
{7}        17374    {4,5,6,7}   276     changed
{6,7}      17374    {2,3,4,5}   276     changed
{5,6,7}    17374    {2..7}       13     changed
{2,4,6}    17374  margin 0.013
{3,5,7}    17374  margin 3.726   higher than keeping everything
```

Two things follow, and both were surprises.

**Individual removability is not redundancy.** It is a marginal statement. The
information is distributed: no single snapshot carries it, the group does.

**Survival is not invariance.** `{2,4,6}` keeps the token with a margin of **0.013**. The
perturbation simply failed to cross a boundary. And `{3,5,7}` produces a *higher* margin
than keeping everything, so removal is not even monotonically harmful.

Under the criterion these tests were meant to serve — $\phi(S_1)=\phi(S_2) \Rightarrow$
indistinguishable downstream — dropping a snapshot fails immediately, because the logit
vector changes by up to 5 nats. Argmax ablation measures robustness. It cannot measure
sufficiency, because the argmax discards exactly the evidence sufficiency is about.

---

## 5. The architecture already performed the reductions

This is the observation that reframed the search.

| place | what it already is |
|---|---|
| `f_a` / `f_b` | a rank-128 factorization of a $12288\times7168$ map |
| MLA latent | 512 dimensions plus 64 rope, in place of $96\times256$ |
| KDA head state | $128\times128$ per head, compressed from the 7168 stream |
| MXFP4 experts | 4 bits per weight with a shared group exponent |

Every structural boundary the reduction document identifies as a place to look is a
place where a compression has **already been applied**. What is left over is the part
that resisted it, which is why the remaining searches come back empty:

```
W_gate, 33792 x 7168        full rank 7168, by LU and by Gram spectrum
one expert W1, 3072 x 3584  full rank 3072
two experts stacked         already span all of R^3584
896 experts at layer 1      896 distinct, no duplicates
(g,u) pairs, 248,832        0 exact duplicates, 0 exact zeros
recurrence, 4 kinds         none, every one at or below its control
```

A useful way to state it: the model is not a computation with redundancy in it. It is a
computation from which the available redundancy was already removed at design time, and
the residue is dense by construction.

---

## 6. Where the cost actually is

Measured, not estimated, because the estimate was wrong by a large factor.

```
MoE routed experts                       75.05%
attention, KDA and MLA                   14.87%
MoE shared expert                         5.34%
tail: aggregate, norm, lm_head            1.68%
MoE latent down-projection                1.14%
MoE router                                0.56%
dense MLP, layer 0 only                   0.29%
snapshot aggregation + pre-MLP norm        0.04%
```

I had estimated experts at 45% and attention at 38% from a FLOP count. The FLOP count
misses that every expert weight is nibble-extracted, table-looked-up and scaled before
the single multiply-accumulate it does count, with no reuse to amortize it: five
positions draw 80 experts, of which 74 are distinct.

Three quarters of this model is decoding and multiplying 4-bit weights that are each used
approximately once.

---

## 7. How to measure redundancy without fooling yourself

Seven instrument errors occurred across this work. Five share one shape.

| # | the instrument | what it reported | what was true |
|---|---|---|---|
| 2 | a filtered tap loader | "site 27 is not traced" | the filter excluded it |
| 3 | a shadowed variable | snapshot norms | per-head KDA norms |
| 5 | `pgrep -f walkdec` | the walk is still running | it matched its own command line |
| 6 | float64 tolerance on float32 data | rank 94, 127, 128 | rank 1, 2, 3 |
| 7 | a FLOP model of a decoding kernel | experts are 45% | 75% |

**The instrument reported a property of itself and it was read as a property of the
system.** That is the shape. Guarding against it is not carefulness, it is a procedure:

- **Always run a control.** The gate-up alignment would have been reported as a striking
  structural finding without one. With one, the shuffled pairing collapsed to chance in
  both directions, which is what made the finding real rather than plausible.
- **Print the spectrum, not the rank.** A single integer from an unstated tolerance is
  not evidence. The gap between $10^{-2}$ and $10^{-8}$ is.
- **Prefer an impossibility check.** The W_sh1 measurement reported rank 6658 for a
  matrix with 6144 rows. A rank above the row count cannot happen, so the number was
  self-evidently the tolerance and not the data. That caught a repeat of error 6 without
  requiring me to be suspicious.
- **Judge completion by the artifact.** Not the process table, which can describe the
  watcher rather than the work.
- **Measure the denominator before quoting a percentage.** Six figures in the evidence
  file were quoted against an estimated denominator and are quantitatively unreliable
  as a result.

---

## 8. What this does not establish

- **One prompt.** Five tokens, `The capital of France is`. Every number is n=1, and the
  ablation margins are tight enough that another prompt could reorder that table.
- **Two layers, mostly.** The $(g,u)$ analysis covers layers 0 and 1. The expert analysis
  covers layer 1. The other 91 layers are assumed similar and were not checked.
- **Nothing here bounds what is possible.** Every result is "this candidate failed",
  never "no candidate can succeed". The one genuinely general argument in the whole body
  of work is narrow: since $o = S^\top q$ with $q$ unconstrained makes $S \mapsto (q
  \mapsto S^\top q)$ injective, no lossy $\phi$ can exist on the *whole* state space.
  Everything else is an absence of evidence.
- **The approximate structure was not pursued.** Two thirds of gate-up row pairs have
  $|\cos| > 0.5$ against 0.011 for shuffled pairs. That is real and large and sits
  outside the exactness criterion fixed for this work. Whether it supports an
  approximate reduction is untested, and would be a different project with a different
  success condition.
