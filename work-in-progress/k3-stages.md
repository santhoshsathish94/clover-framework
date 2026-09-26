# Kimi K3, one stage at a time

The model is not 93 things happening in a row. It is a long sequence of individual
transformations, per position, with a recurrent aggregation that ties groups of layers back
together rather than passing state strictly forward. This document walks those stages one at
a time and proves each one separately against the running engine.

Each entry has the same four parts:

1. **The equation** — what the stage is, mathematically.
2. **What this stage exactly does** — in words, including what it does *not* do.
3. **The real data** — the actual captured numbers from a real execution.
4. **What the equation gave** — my computed result against the engine's, and the verdict.

Evidence throughout comes from one traced execution of the released checkpoint on the pure-C
engine: prompt `The capital of France is`, 5 tokens, raw per-stage vectors captured to
`e1.bin`. Weights are read from the checkpoint, never from the engine's own buffers.

---

## Stage 1 — the embedding lookup

### 1. The equation

$$h^{(0)}_p \;=\; W_{\text{embed}}\big[\,t_p\,\big], \qquad
W_{\text{embed}} \in \mathrm{BF16}^{\,163840 \times 7168}$$

For each position $p$, take token id $t_p$, select that row of the embedding table, and widen
each BF16 value to float32. There is no sum, no product, and no nonlinearity.

### 2. What this stage exactly does

It turns five integers into five vectors of 7,168 floats.

- **Input**: the token ids, one per position. Nothing else. No weights are combined, no
  context is consulted, position $p$ does not see position $q$.
- **Operation**: a row selection followed by a format widening. BF16 to F32 is exact by
  construction — BF16 *is* the top 16 bits of an F32, so widening is a shift and a zero-fill,
  not a conversion that can round.
- **Output**: $h^{(0)}_p$, the residual stream at its starting value, 7,168 floats per
  position, 35,840 floats in total for this prompt.
- **What it does not do**: no positional information is added here. No normalization. This is
  the one stage in the model that performs no arithmetic at all, which is why it is the
  natural zero point — anything that fails to match here is a reading error, not a rounding
  difference.

### 3. The real data

Source: engine tap site 31 (`embed`), five records of 7,168 float32.
Table: `language_model.model.embed_tokens.weight`, BF16, `[163840, 7168]`, in
`model-00094-of-000096.safetensors`, memory-mapped directly from the checkpoint.

**The output is a pure widening.** Every captured float has its low 16 bits zero. A float32
produced by any computation essentially never does.

```
pos 0..4:  0 of 7168 floats have nonzero low 16 bits   => pure widening: True
```

**The token ids were not taken from the log.** To keep the check independent, each position's
engine vector was searched against all 163,840 rows of the table. Exactly one row matched in
every case:

```
pos 0  token id 1008    candidates sharing component 0: 117
pos 1  token id 10484   candidates sharing component 0: 299
pos 2  token id 318     candidates sharing component 0: 219
pos 3  token id 15383   candidates sharing component 0:  92
pos 4  token id 387     candidates sharing component 0:  52
```

**The recovered ids decode back to the prompt**, which is what makes the match above
non-circular — a wrong row would not spell anything:

```
1008 b'The'   10484 b' capital'   318 b' of'   15383 b' France'   387 b' is'
joined: b'The capital of France is'   (24 bytes)
```

The engine log independently records `tokenized: 24 bytes -> 5 ids`. Byte count and id count
both agree with the recovered rows.

### 4. What the equation gave

Bit-for-bit against the engine, all five positions:

| position | token id | identical floats | max ulp |
|---|---|---|---|
| 0 | 1008 | 7168 / 7168 | 0 |
| 1 | 10484 | 7168 / 7168 | 0 |
| 2 | 318 | 7168 / 7168 | 0 |
| 3 | 15383 | 7168 / 7168 | 0 |
| 4 | 387 | 7168 / 7168 | 0 |

**35,840 of 35,840 floats identical. Max ulp 0.**

The first six components of position 0, with the XOR of my float against the engine's:

| i | bf16 bits | equation | engine | xor |
|---|---|---|---|---|
| 0 | 0x3D05 | 0.032470703 | 0.032470703 | 0x00000000 |
| 1 | 0xBB21 | -0.002456665 | -0.002456665 | 0x00000000 |
| 2 | 0xBD14 | -0.036132812 | -0.036132812 | 0x00000000 |
| 3 | 0xB8AA | -8.1062317e-05 | -8.1062317e-05 | 0x00000000 |
| 4 | 0x3C14 | 0.0090332031 | 0.0090332031 | 0x00000000 |
| 5 | 0x3CBF | 0.02331543 | 0.02331543 | 0x00000000 |

**Verdict: identical, not approximate.** The equation for stage 1 is exactly the engine's
behavior, and the agreement is at the level of raw bits rather than a tolerance.

One caveat worth stating plainly, because it bounds what this proves: stage 1 matches trivially
*because* it contains no summation. It establishes that the capture is being read correctly,
that the weight file is being addressed correctly, and that the comparison method is sound. It
says nothing yet about stages that accumulate.

Reproduce with `stage1.py` on the trace host.

---

## Stage 2 — the pre-attention aggregation, which does not run here

### 1. The equation

$$h \;\leftarrow\; h \qquad \text{when } d = 0$$

Identity. Not because an aggregation computed something that happened to equal its input, but
because the aggregation is never entered.

### 2. What this stage exactly does

Nothing, at this point in the model.

The engine carries a stack of snapshots of the residual stream and, before attention, blends
the current residual with every stored snapshot. That blend is the mechanism that stops the
model being a straight chain. But it is guarded:

```c
/* aggregation before attention, only when snapshots already exist */
if (*n_blocks > 0) {
    ...
    k3_attn_res(h + t*E, src, foldA, *n_blocks + 1, E, c->rms_eps);
}
K3_TRACE_VEC("attn_res.pre_attn", h, T*E);
```

At this stage `*n_blocks == 0` — no snapshot has been pushed yet, because the first push
happens *after* this point in the same layer. The guard is false, the kernel is not called,
and the trace is emitted on an untouched buffer.

So there is no summation, no weight, no `eps` and no `expf` in this stage. None of it
executes. The residual arrives and leaves unchanged.

### 3. The real data

| evidence | value |
|---|---|
| `stack.depth` | 0 |
| guard `*n_blocks > 0` | false, kernel not entered |
| `embed == layer.in` | 7168 / 7168 floats, all 5 positions |
| `embed == attn_res.pre_attn` | 7168 / 7168 floats, all 5 positions |
| L2 across the three sites | 4.13786926, unchanged |
| FNV across the three sites | `f45927e03fd96f75`, unchanged |

The hash is byte-identical at `embed`, `layer.in` and `attn_res.pre_attn`. That is what a
pass-through looks like, and what a computation essentially never produces.

### 4. What the equation gave

Difference from the engine: **exactly zero**, for the same reason as stage 1 — nothing was
computed. 35,840 of 35,840 floats identical.

### How this was nearly recorded wrong

The first attempt at this stage worked from the captured numbers alone. It proposed candidate
operations, scored each against the data, and the winner was a softmax over RMS-normalized
projections — which, with a single source, gives weight exactly 1.0 and returns the input
untouched. Every number agreed. The conclusion drawn was that stage 2 is that formula
degenerating to the identity.

That is wrong. The kernel is not invoked at all.

Both accounts predict bit-identical output, so no amount of comparing vectors could separate
them. Only reading the code that runs could. The lesson is narrow and worth keeping: a
hypothesis that reproduces the observations to the last bit can still misdescribe the machine,
and at a control-flow boundary it usually will.

### Noted for later stages, not acted on

Reading `k3_attn_res` to establish the above also settled two things that belong further
along, recorded here so they are not rediscovered:

- the source stack is ordered **snapshots first, current residual last**, with
  `nsrc = n_blocks + 1`
- the weighted sum accumulates in **float32**, source-major — earlier reconstructions of this
  kernel accumulated in float64, which is the likely cause of the partial bit-agreement seen
  in those checks

---

## Stage 3 — the snapshot push, which does run

### 1. The equation

There is no equation on the residual. Stage 3 leaves it untouched:

$$h \;\leftarrow\; h$$

Its content is a state transition:

$$S \;\leftarrow\; S \cup \{\,r\,\}, \qquad d \;\leftarrow\; d+1, \qquad r = h^{\text{(layer entry)}}$$

Here $d$ goes $0 \to 1$ and $S$ becomes $\{\,\text{the embedding}\,\}$.

### 2. What this stage exactly does

This is the first stage in the model that does anything, and what it does is write machine
state rather than transform data. Three effects, none of them to the residual:

1. the layer-entry residual is copied into slot `n_blocks` of the snapshot stack
2. `n_blocks` increments, 0 to 1
3. `have_prefix` is cleared to 0

The guard here is `layer_idx % attn_res_block == 0`, which is true, where stage 2's guard was
false. That single snapshot is what stage 2 will have to aggregate over from the next layer
onward — stage 2 was empty precisely because stage 3 had not happened yet.

### 3. The real data

**It executed.** The engine emits `snapshot.pushed` only from inside the branch:

```
layers where snapshot.pushed was emitted: [0, 12, 24, 36, 48, 60, 72, 84]
value at stage 3: 1                       => branch TAKEN
```

Eight pushes in the whole forward pass, and the raw snapshot record exists at exactly those
same eight points, five positions each.

**The state change, confirmed from both sides:**

```
stack.depth observed before the push:      0
snapshot.pushed (n_blocks after ++):       1
stack.depth observed by the next reader:   1
```

**What was copied:**

```
pos 0..4  snapshot == embed 7168/7168  == layer.in 7168/7168  == attn_res.pre_attn 7168/7168
```

### 4. What the equation gave

The residual is not written, so the difference there is zero by construction. The state write
is exact: **35,840 of 35,840 floats identical** between the stored snapshot and its source.

### What the data cannot settle here

All three candidate sources — `embed`, `layer.in`, `attn_res.pre_attn` — are the same vector
at this point, because stage 2 did not run and nothing has diverged them. So the measurement
cannot say which one was copied. The source says it is `pref`, the buffer captured at layer
entry, which is `layer.in`. That is reported from the code, not from the measurement.

It could be discriminated at a later push, where the aggregation has already modified `h`
before the copy. That is a different stage and has not been looked at.

### A note on counting

Three stages in, there are already three different kinds:

| stage | kind | effect on the residual |
|---|---|---|
| 1 | transforms data | writes it |
| 2 | skipped by a guard | none, not executed |
| 3 | writes machine state | none, but changes what later stages see |

They are not interchangeable, and a stage count that treats them as one kind will not land.

---

## Stage 4 — the pre-attention RMSNorm, the first arithmetic in the model

### 1. The equation

$$y_i \;=\; \big(w_i \, x_i\big)\cdot \mathrm{inv},
\qquad
\mathrm{inv} \;=\; \mathrm{fl}_{32}\!\left(\frac{1}{\sqrt{\dfrac{1}{n}\displaystyle\sum_{j=0}^{n-1} x_j^{2} \;+\; \epsilon}}\right)$$

with $n = 7168$ and $\epsilon = \mathrm{fl}_{32}(10^{-5})$. The sum is accumulated in double,
`inv` is rounded to float32 exactly once, and the outer product is evaluated in float32
**left to right**: $(w_i x_i)$ first, then $\times\,\mathrm{inv}$.

### 2. What this stage exactly does

It rescales the residual by a single factor derived from the vector's own root-mean-square,
then applies a learned per-component gain (`input_layernorm.weight`, BF16).

Its input is still the raw embedding. Stage 2 was skipped and stage 3 wrote only machine
state, so nothing has touched the residual since stage 1.

It writes into a **separate buffer**, not in place. The residual survives untouched, which is
what lets the residual addition later in the layer still see the pre-norm value. Everything
downstream in the attention path reads the normalized copy; the residual path does not.

The whole kernel is four lines:

```c
double ss = 0.0;
for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
for (int i = 0; i < n; i++) y[i] = w[i] * x[i] * inv;
```

### 3. The real data

```
pos  ss (double, sequential)  inv (float32)   identical   max ulp
0    4.1802332169374026       41.0588531      7168/7168   0
1    3.6784657435065022       43.7194748      7168/7168   0
2    2.4703597339373164       53.1016273      7168/7168   0
3    3.4671155571629306       45.0060883      7168/7168   0
4    3.3257877640092          45.9326324      7168/7168   0
```

`eps` as the engine actually holds it is `float32(1e-5)` widened to double,
`9.9999997473787516e-06` — not `1e-5`.

Two order-sensitive choices were **tested rather than asserted**:

| choice | effect |
|---|---|
| sequential double sum vs numpy pairwise sum | no effect, output bit-identical either way |
| `(w·x)·inv` vs `w·(x·inv)` | ~74% identical, so roughly a quarter of every vector wrong |

At position 3 the sequential and pairwise sums differ by one ulp in double
(`...9306` against `...9311`) and the float32 output is still identical, because that
difference vanishes when `inv` is rounded. The double accumulator has headroom at this width,
so the sequential order is not claimed to be required here. The **multiply association is**.

### 4. What the equation gave

**35,840 of 35,840 floats identical. Max ulp 0.**

This is the first stage that could have been wrong and was not. Stages 1 to 3 matched because
they compute nothing — a lookup, a skipped branch, a state write. Stage 4 matches because the
arithmetic is right, association order included.

---

## Stage 5 — the KDA projections

### 1. The equation

Every projection goes through one int8 kernel. For output row $o$, with int8 weights
$w_{o,i}$ and a per-row float32 scale $s_o$:

$$y_o \;=\; s_o \cdot \Big[\big((A_0{+}A_4)+(A_2{+}A_6)\big) + \big((A_1{+}A_5)+(A_3{+}A_7)\big)\Big]$$

where $A_j$ is a float32 lane accumulated with single-rounded fused multiply-add over the
inputs at $i \equiv j \pmod 8$, taken in blocks of 16. The eight lanes are reduced by that
fixed tree, not by a running total.

Stage 5 applies this six times. The observable one chains it twice:

$$z \;=\; \mathrm{mm}\big(W_{f_b},\; \mathrm{mm}(W_{f_a},\, x)\big)$$

### 2. What this stage exactly does

It is step 1 of the KDA attention block: six int8 matmuls per position, all reading the
stage 4 output.

| projection | shape | becomes |
|---|---|---|
| `q` | `[12288, 7168]` | query |
| `k` | `[12288, 7168]` | key |
| `v` | `[12288, 7168]` | value |
| `b` | `[96, 7168]` | one scalar per head, later beta |
| `f_a` then `f_b` | `[128, 7168]` then `[12288, 128]` | `z`, the decay-gate input |

The `f_a`/`f_b` pair is the one to notice. It is computed **once per position and feeds every
head**, `E -> 128 -> 12288`. It is a shared low-rank path, not a per-head projection.

### 3. The real data

Only one of the six is directly observable:

```
site 19 kda.q_conv   dim 12288    after ShortConv, not the raw projection
site 20 kda.k_conv   dim 12288    after ShortConv
site 21 kda.v_conv   dim 12288    after ShortConv
site 24 kda.beta     dim 96       after the sigmoid
site 28 kda.z        dim 12288    the raw f_b(f_a(x)) output
```

There is no tap on raw `q`, `k`, `v` or `b`. Their first appearance is already fused with the
next stage.

Reproducing the engine's AVX2 path exactly:

```
pos  max |z|       identical      max ulp
0    5.61936617    12288/12288    0
1    8.0720396     12288/12288    0
2    13.0760984    12288/12288    0
3    7.28017998    12288/12288    0
4    8.24403381    12288/12288    0

total identical: 61440 / 61440
```

The reduction tree is not cosmetic. Same weights, same inputs, plain dot product instead:

```
pos 0..4  identical 438, 501, 543, 478, 630 of 12288   (~4%)
```

Roughly 96% of every vector wrong, purely from summing in a different order.

### 4. What the equation gave

**61,440 of 61,440 floats identical. Max ulp 0.**

### What this does not cover

Three limits, stated rather than glossed:

- **Four of the six projections are not individually verified**, because nothing observes
  them. What is verified is the kernel all six call, and `z` exercises it at two different
  shapes, `in=7168` and `in=128`.
- **`f_a`'s 128-dim intermediate is also unobserved.** `z` matching in all 61,440 floats is
  very strong evidence that it is exact, but that is inference, not measurement.
- Both shapes divide by 16 exactly, so the kernel's scalar tail loop never executes here and
  remains untested.

*Both of the first two were later closed. Stage 6 consumes `q`, `k` and `v` and matches bit
for bit; stage 8 consumes `b` and matches bit for bit. Of the six projections only `f_a`
remains confirmed by inference rather than measurement. See those entries.*

---

## Stage 6 — ShortConv with fused SiLU

### 1. The equation

For channel $c$ at position $t$, with taps $w_{c,0..3}$ and history $b_{0..2}$ ordered
oldest first:

$$a \;=\; \Big(\big((w_{c,3}\,x_t) \,+\, w_{c,0} b_0\big) \,+\, w_{c,1} b_1\Big) \,+\, w_{c,2} b_2$$

$$y \;=\; a \cdot \sigma(a), \qquad \sigma(a)=\frac{1}{1+\exp(-a)}$$

Every operation is float32, evaluated strictly left to right, with `exp` from glibc. Note
that the **current input is the first term**, followed by the history from oldest to newest.
The parenthesization is not decoration; it is the difference between 100% and 80%.

### 2. What this stage exactly does

A depthwise causal convolution of width 4 along the position axis, with SiLU fused into the
same kernel, applied independently to `q`, `k` and `v`.

Each of the 12,288 channels has its own four taps and its own history buffer. Nothing is
mixed across channels.

This is the first stage that **mixes across positions**. Stages 1 to 5 treated every position
independently; here position $t$ sees positions $t-1$, $t-2$ and $t-3$. The history at the
start of a fresh call comes from the carried recurrent state.

The conv weights are **F32**, the only F32 weights encountered so far.

### 3. The real data

Reproducing the kernel exactly — newest tap first, plain float32 adds, glibc `expf`:

```
q  [12288, 12288, 12288, 12288, 12288]
k  [12288, 12288, 12288, 12288, 12288]
v  [12288, 12288, 12288, 12288, 12288]
total identical: 184320 / 184320
```

Three variants were run against the same data to find which choices are load-bearing:

| variant | identical of 184320 | reading |
|---|---|---|
| **A** exact | **184320** | — |
| B FMA-contracted adds | 140283 | the compiler did not contract; plain multiply then add |
| C numpy `exp` instead of libm | 159267 | glibc `expf` is required |
| D oldest tap first, current last | 147438 | the accumulation order is load-bearing |

Variant D fails in a pattern that confirms the mechanism rather than merely scoring badly:

```
q  [12288, 12288, 8755, 7822, 7645]
```

Positions 0 and 1 stay exact, then it collapses. At $t=0$ there is one nonzero term; at
$t=1$ there are two, and the addition of two terms commutes. From $t=2$ there are three or
more and the order starts to matter. The failure appears exactly where the arithmetic says it
must, and nowhere earlier.

Variant B fails the same way — position 0 exact, because a single multiply has no add to
contract, and everything after it degraded.

### 4. What the equation gave

**184,320 of 184,320 floats identical. Max ulp 0.**

### What this settles, and what it does not

**It closes a stage 5 gap.** The raw `q`, `k` and `v` projections have no tap and were
recorded there as unverified. Stage 6 consumes them and 184,320 floats come out bit-identical.
An error in the raw projections surviving the convolution unchanged across that many values is
not credible, so those three projections are now confirmed indirectly.

**Confirmed by implication:** the carried conv state is zero on a fresh call. That was an
assumption; had it been wrong, nothing would have matched.

**Still not verified:** the `b` projection. Nothing has observed it yet.

**This is a composed check.** Stage 5 and stage 6 together, because no tap exists between
them. It is not an isolated measurement of stage 6, and the available taps do not allow one.

---

## Stage 7 — per-head L2 normalization of q and k

### 1. The equation

For each head block $v \in \mathbb{R}^{128}$ of `q` and of `k`:

$$v_i \;\leftarrow\; v_i \cdot \mathrm{fl}_{32}\!\left(\frac{1}{\sqrt{\displaystyle\sum_{j=0}^{127} v_j^{2} \;+\; \epsilon}}\right),
\qquad \epsilon = \mathrm{fl}_{32}(10^{-6})$$

The sum is accumulated in double, `inv` is rounded to float32 once, then one float32 multiply
per element. No division by $n$, and no learned weight.

### 2. What this stage exactly does

It scales each head of `q` and `k` to unit length, independently and in place.

This is the second normalization in the model and it is **not** the same operation as stage 4.
Conflating the two is easy and would be wrong in four separate ways:

| | stage 4 | stage 7 |
|---|---|---|
| divisor | $\sqrt{\tfrac{1}{n}\sum x^2 + \epsilon}$, root mean square | $\sqrt{\sum x^2 + \epsilon}$, true L2 |
| epsilon | `1e-5` | `1e-6` |
| learned weight | yes, `input_layernorm.weight` | none |
| scope | the whole 7168 vector | each 128-wide head, 96 per position |
| in place | no, writes a separate buffer | yes |

`v` is deliberately left alone. Only `q` and `k` are normalized, and the engine's site table
confirms it: there are `kda.q_norm` and `kda.k_norm` taps and no `kda.v_norm`.

### 3. The real data

```
q_norm  site 22  identical per position: [12288, 12288, 12288, 12288, 12288] of 12288
k_norm  site 23  identical per position: [12288, 12288, 12288, 12288, 12288] of 12288
total identical: 122880 / 122880
```

Accumulator order was tested rather than assumed, and at this width it does not matter:

```
numpy pairwise sum instead of sequential:  122880 / 122880
```

Same result as stage 4. At $n=128$ the double accumulator has ample headroom, so the
sequential order is not claimed to be required here.

**The epsilon leaves a signature, and it is a second confirmation.** The output is not exactly
unit length. If `eps` is added to the raw sum of squares, the resulting norm is
$\|v\|/\sqrt{\|v\|^2+\epsilon} \approx 1 - \epsilon/(2\|v\|^2)$, so smaller heads should land
further below 1 by a predictable amount:

```
head   L2 before      L2 after       predicted
0      0.447325409    0.999997474    0.9999975
2      0.336629768    0.999995567    0.99999559
```

The measurement matches the prediction to eight digits. That independently establishes that
`eps` is `1e-6` and that it is added to the **sum**, not to the mean — a conclusion reached
from the data alone, agreeing with the code.

### 4. What the equation gave

**122,880 of 122,880 floats identical. Max ulp 0.**

The chain now runs stages 1 to 7 from the token ids with nothing re-seeded from the engine,
and every observable output along the way is bit-identical.

---

## Stage 8 — beta, the per-head write gate

### 1. The equation

For each head $h$:

$$\beta_h \;=\; \sigma(b_h) \;=\; \frac{1}{1 + \exp(-b_h)}$$

float32 throughout, `exp` from glibc, applied in place over the `b` projection buffer. One
scalar per head per position, 96 per position.

### 2. What this stage exactly does

It turns the `b` projection into **beta**, a gate in $(0,1)$.

Beta is the write strength for the recurrent state. It controls how much of the new key and
value pair is written into each head's state at each step, so it is the quantity that decides
how fast a head forgets what it already holds.

This is the smallest stage so far by data volume, 480 values across the whole prompt, but it
is a distinct kernel step with its own effect on the recurrence.

### 3. The real data

The full chain, from the token ids, nothing re-seeded:

| tensor | site | identical |
|---|---|---|
| norm.pre_attn | 2 | 35840 / 35840 |
| kda.q_conv | 19 | 61440 / 61440 |
| kda.k_conv | 20 | 61440 / 61440 |
| kda.v_conv | 21 | 61440 / 61440 |
| kda.q_norm | 22 | 61440 / 61440 |
| kda.k_norm | 23 | 61440 / 61440 |
| kda.z | 28 | 61440 / 61440 |
| **kda.beta** | **24** | **480 / 480** |
| **TOTAL** | | **404960 / 404960** |

Stage 8 on its own:

```
glibc expf   identical 480 / 480
numpy exp    identical 448 / 480
```

The libm dependence appears again. A small sample, but consistent with stage 6.

```
raw b projection range : -2.394173 .. 8.490455
beta range             :  0.083618 .. 0.999795
```

Position 0, first six heads:

```
head   b raw            beta mine        beta engine
0      2.97894406       0.951613784      0.951613784
1      4.21393061       0.98542738       0.98542738
2      2.54413652       0.927178621      0.927178621
3      2.99129462       0.952179253      0.952179253
4      0.702482402      0.668737888      0.668737888
5      1.364434         0.796479404      0.796479404
```

The spread is wide and it matters: beta runs from 0.084 to 0.9998, so some heads overwrite
their state almost completely at each step while others barely write at all.

### 4. What the equation gave

**480 of 480 identical. Max ulp 0.**

### This closes the last open item from stage 5

Stage 5 recorded that four of the six projections had no tap and were unverified. That list
is now empty:

| projection | how it was confirmed |
|---|---|
| `q`, `k`, `v` | stage 6, ShortConv output bit-identical |
| `f_b`, giving `z` | stage 5, directly tapped |
| `b` | stage 8, beta bit-identical |
| `f_a` | inferred through `z`, still not measured |

`f_a` is the one remaining projection whose correctness rests on inference rather than a
measurement of its own output.

---

## Stage 9 — the decay chain, alpha

### 1. The equation

For head $h$ and channel $i = hD + d$:

$$a_h = \exp\big(A_{\log}[h]\big), \qquad u_i = a_h\big(z_i + \mathrm{dt\_bias}_i\big)$$

$$g_i = \lambda\,\sigma(u_i), \qquad \alpha_i = \exp(g_i), \qquad \lambda = -5$$

All float32, with `exp` and the sigmoid from glibc. Note that $a$ is indexed **per head**
while `dt_bias` is indexed **per channel**, and that `g` is written **over** `z` in place —
which is why sites 28 (`z`) and 29 (`g`) are the same buffer at two different times.

### 2. What this stage exactly does

It produces the per-channel **forget factor** for the recurrent state. Each head's state is
multiplied by $\alpha$ before the new write, so this is the stage that decides what gets
discarded and how fast.

The $-5$ is a hard clamp by construction, not a tuned range: $g = -5\,\sigma(u)$ lies in
$(-5, 0]$ whatever $u$ does, so $\alpha$ can never leave $(e^{-5}, 1]$. No value of the input
can make a head forget faster than $e^{-5}$ per step or retain more than perfectly.

### 3. The real data

| tensor | site | identical |
|---|---|---|
| kda.g | 29 | 61440 / 61440 |
| kda.alpha | 25 | 61440 / 61440 |
| **full chain, stages 1 to 9** | | **527840 / 527840** |

**The two indexings are forced by the checkpoint, not chosen.** The intent was to run a
per-channel variant of `A_log` as a control, since the engine source carries a comment
warning that indexing it per channel is a silent fatal error. The variant could not be
constructed:

```
A_log   length 128, nonzero 96   (H = 96 heads)
dt_bias length 12288             (P = 12288 channels)
entries 96..127 of A_log are exactly zero
```

There are 12,288 channels and 128 `A_log` entries, so the tensor is 96 times too short for
per-channel indexing and the attempt fails on an array bound. Exactly the first 96 entries
are nonzero, matching the head count. The shapes settle it on their own, which is a firmer
result than a variant that merely scores badly.

Ranges:

```
a = exp(A_log[h])  : 0.470926 .. 11.776435
g                  : -5.000000 .. -0.000000     lower bound lb = -5.0
alpha              : 0.006738 .. 1.000000       e^lb = 0.006738
mean alpha         : 0.786708
fraction of alpha above 0.99 : 11.72%
fraction of alpha below 0.50 : 12.50%
```

Both bounds are reached exactly. `g` touches -5.000000 and `alpha` touches 0.006738, which is
$e^{-5}$ to six digits. The gate is saturating at both ends on a five-token prompt rather
than sitting in a comfortable middle, and the distribution is genuinely split: 11.7% of
channels keep essentially everything while 12.5% discard more than half per step.

### 4. What the equation gave

**122,880 of 122,880 floats identical across `g` and `alpha`. Max ulp 0.**

### One behavior worth recording

The run emits `RuntimeWarning: overflow encountered in expf`. It is benign, and reproducing
it is the point: for strongly negative $u$, $\exp(-u)$ overflows to infinity, the sigmoid
returns 0, so $g = 0$ and $\alpha = 1$, meaning no decay. The engine takes the same path. The
bits match, so the overflow is being reproduced rather than avoided.

---

## Stage 10 — the recurrence

### 1. The equation

Per head, at each position, with $\alpha \in \mathbb{R}^{128}$ from stage 9 and $\beta$ the
scalar from stage 8:

$$S \leftarrow \operatorname{diag}(\alpha)\,S, \qquad u = S^\top k$$

$$S \leftarrow S + \beta\,k\,(v-u)^\top, \qquad o = S^\top \frac{q}{\sqrt{128}}$$

All float32, every sum accumulated **sequentially in $i$** ascending, no FMA. The output is
taken from the **already updated** state.

### 2. What this stage exactly does

It is the recurrence, and it is the first stage that carries **state across positions within
a head**: a $128 \times 128$ matrix per head, 96 of them, updated once per position.

It is a **delta rule**, not an accumulation. Four steps:

1. **decay** — row $i$ of $S$ is scaled by $\alpha_i$. Per *key channel*, not a scalar. This
   is what "channel-wise forget gate" means, and it is why stage 9 produced 12,288 alphas
   rather than 96.
2. **read** — $u = S^\top k$, what the state currently predicts for this key.
3. **write** — $S_{ij} \mathrel{+}= k_i\,\beta\,(v_j - u_j)$. The $(v-u)$ term is the
   prediction error. Plain accumulation would write $v$; writing the error is what makes this
   a delta rule.
4. **output** — $o = S^\top q$ from the already updated state, not the pre-update state.

`q` is pre-scaled by $1/\sqrt{128}$ in the caller, before the step sees it.

### 3. The real data

| tensor | site | identical |
|---|---|---|
| kda.o | 26 | 61440 / 61440 |
| **full chain, stages 1 to 10** | | **589280 / 589280** |

```
per position: [12288, 12288, 12288, 12288, 12288] of 12288
o range   : -0.020787 .. 0.078446
final |S| : mean 0.001612  max 1.109369
```

After five positions the state is sparse in magnitude: mean $|S|$ of 0.0016 against a max of
1.109, so a small number of entries carry almost all of it.

### 4. What the equation gave

**61,440 of 61,440 floats identical. Max ulp 0.**

### One detail deliberately not reproduced

The kernel carries `if (ki == 0.0f) continue` guards in the read and write steps. Those were
not reproduced, on the reasoning that skipping a term whose multiplier is zero is numerically
inert. The output came out bit-identical across all 61,440 floats, which confirms it: the
guards are a speed optimization and not part of the arithmetic.

### A note on the tooling, recorded because it cost two runs

Commands issued over ssh began hanging with no output and no prompt return, which looks
exactly like a server or network fault. It was neither. Without `-n`, ssh forwards the local
terminal's stdin to the remote command, so any remote command that ends up waiting on stdin
blocks forever.

The diagnosis worth keeping is the order: a trivial local command first, then
`ssh -n host 'echo OK'`, then the real command. Local fine plus ssh fine plus this one hanging
locates the fault in the command, not the link. Checking `uptime` and `pgrep` on the server
first showed load 0.00 and nothing running, which ruled out the remote side before anything
was killed there.

---

## Stage 11 — head-wise RMSNorm on the recurrence output

### 1. The equation

For each head block $v \in \mathbb{R}^{128}$ of $o$, in place:

$$v_i \;\leftarrow\; (w_i\,v_i)\cdot \mathrm{fl}_{32}\!\left(\frac{1}{\sqrt{\tfrac{1}{128}\displaystyle\sum_{j} v_j^{2} + \epsilon}}\right),
\qquad \epsilon = \mathrm{fl}_{32}(10^{-5})$$

The same kernel as stage 4, at a different width, with a different weight, applied in place.

### 2. What this stage exactly does

It renormalizes each head of the recurrence output and applies the learned `o_norm` gain.

This is the **third distinct normalization** in the model, and no two of them are the same
operation:

| | stage 4 | stage 7 | stage 11 |
|---|---|---|---|
| kind | RMS | true L2 | RMS |
| scope | whole 7168 vector | 128 per head | 128 per head |
| divide by $n$ | yes | no | yes |
| epsilon | 1e-5 | 1e-6 | 1e-5 |
| weight | `input_layernorm` | none | `o_norm` |
| in place | no | yes | yes |

### 3. The real data

```
o_norm: 128 values, min 0.003512 max 0.033375

head   RMS before         RMS after
0      0.00079877037      0.00238616205
1      0.00238680607      0.00751662952
2      0.000409175463     0.0010223587
3      0.000238712307     0.000686388838
4      0.000137517843     0.000487568563
5      0.000138764023     0.000486911634

o  range: -0.020787 .. 0.078446
on range: -0.055641 .. 0.079059
```

The stage **amplifies**, by roughly 2.5 to 3.5 times and by a different factor per head. The
recurrence output is very small, RMS of order 1e-4 to 2e-3, and this lifts it before the gate.

Everything already verified is unchanged: the chain through stage 10 remains 589,280 of
589,280.

### 4. What the equation gave

**Nothing yet. This stage cannot be measured on its own.**

From the source:

```c
for (h...) k3_rmsnorm(ot + h*D, ot + h*D, w->o_norm, D, c->rms_eps);   /* stage 11 */
k3_mmw(gb, xt, w->g, w->wdt, E, P);                                    /* stage 12 */
for (int i = 0; i < P; i++) ot[i] *= sigmoidf_(gb[i]);
K3_TRACE_RAW("kda.gated", ot, 1, P);                                   /* site 27 */
```

Site 26 is before the norm and site 27 is after the norm **and** the gate, with no trace call
between them. The output is computed and held; its correctness is deferred to stage 12, the
same situation as stage 5's `q`, `k` and `v`, which stage 6 resolved.

### An error this run caught in its own instrument

The script printed a list of "tap sites captured at this layer" that omitted site 27, and that
was used to argue the gated output is never traced. **The list was wrong.** It reflected the
script's own load filter, which had never requested site 27, not what the engine captured.
Read straight from the binary, layer 0 carries 21 sites and site 27 is among them, 5 records
of 12,288.

This is the same failure as stage 2 in a different costume: a claim about the machine that was
really a property of the instrument. The conclusion survives, because the absence of a trace
call between the norm and the gate is visible in the source. The evidence for it does not
survive, and has been replaced.

The general form worth keeping: **a filtered view can only show absence of what it was asked
to load.** Any claim that something is missing has to come from the unfiltered source.

---

## Stage 12 — the gate

### 1. The equation

$$\mathrm{gated}_i \;=\; \mathrm{RMSNorm}_{\text{head}}(o)_i \cdot \sigma\big((W_g\,x)_i\big)$$

with $x$ the **stage 4 output**, $W_g \in \mathbb{I}8^{\,12288 \times 7168}$ through the same
`k3_matmul_q8` kernel as stage 5, and the multiply elementwise in float32.

### 2. What this stage exactly does

It scales the normalized attention output by a sigmoid gate. The gate is a projection of the
layer input, not of anything attention produced.

**The gate never sees the attention result.** $W_g$ is `[12288, 7168]`, so its input has to be
7168 wide. The attention output is 12,288 wide, so it cannot be the source, and the only
7168-wide vector in scope is the stage 4 output. The shape forces it, the same way the shapes
forced the `A_log` indexing at stage 9 — this is not read off a variable name.

The consequence is worth stating. Stages 5 through 11 — every projection, the convolution,
both norms, the whole recurrence — produce a value that is then scaled by a factor computed
entirely **independently of all of it**. The gate depends only on the token's own normalized
representation, so nothing it does can respond to what attention actually retrieved.

### 3. The real data

```
kda.gated identical per position: [12288, 12288, 12288, 12288, 12288] of 12288
total 61440 / 61440

gate value range        : 0.002409 .. 0.997656
mean gate               : 0.449078
fraction of gates < 0.1 : 0.99%
fraction of gates > 0.9 : 0.10%
gated range             : -0.050093 .. 0.055357
```

**The gate is graded, not a switch.** Only 1.09% of the 61,440 values lie near either rail. It
spans nearly the whole open interval but sits mostly in the middle, mean 0.449, so it applies
continuous attenuation — roughly halving everything — rather than selecting heads on or off.

That is a real contrast with stage 9. `alpha` saturated at both ends on this same prompt,
touching $e^{-5}$ and 1 exactly. Two sigmoid gates in the same block, behaving quite
differently.

### 4. What the equation gave

**61,440 of 61,440 floats identical. Max ulp 0.**

### This closes stage 11

Stage 11 has no tap of its own, and its output feeds directly into this multiply. Since the
product is bit-identical across all 61,440 floats, the head-wise RMSNorm producing it must be
exact. Confirmed indirectly, the same way stage 6 confirmed stage 5's `q`, `k` and `v`.

### Where the chain stands

| stages | site | identical |
|---|---|---|
| 1 to 10 | various | 589,280 / 589,280 |
| 11 | none | confirmed through stage 12 |
| 12 | 27 | 61,440 / 61,440 |
| **total** | | **650,720 / 650,720** |

Every observable output from the token ids through the end of the gate is bit-identical, with
nothing re-seeded from the engine.

---

## Stage 13 — the output projection

### 1. The equation

$$\mathrm{attn.out} \;=\; W_o \cdot \mathrm{gated}, \qquad W_o \in \mathbb{I}8^{\,7168 \times 12288}$$

Through the same int8 kernel and the same fixed reduction tree as every other projection in
this layer.

### 2. What this stage exactly does

It folds the gated per-head result back to the residual width. The 96 heads of 128 dimensions
each collapse into a single 7,168-vector that can be added to the residual stream.

It is the same `k3_matmul_q8` as stage 5 at the transposed shape: inner dimension 12,288,
which is 768 sixteen-wide blocks with **zero scalar tail**. The kernel's tail loop has still
not been exercised anywhere in this layer.

This is the last stage of the attention block.

### 3. The real data

```
attn.out identical per position: [7168, 7168, 7168, 7168, 7168] of 7168
total 35840 / 35840

gated    range : -0.050093 .. 0.055357
attn.out range : -0.157121 .. 0.228775

attn.out  L2 per position: [0.36232, 0.430709, 0.750011, 0.585526, 0.571729]
embedding L2 per position: [2.044562, 1.917933, 1.571738, 1.862019, 1.823674]
```

The projection **expands**, inputs bounded by about 0.055 and outputs reaching 0.229, because
12,288 gated values sum into each of 7,168 outputs.

Against the residual it is about to join, attention contributes between 18% and 48% of the
embedding's magnitude. The ordering is worth noting: position 2, `' of'`, has the smallest
embedding at 1.572 and the largest attention output at 0.750. The least informative token on
its own draws the most from context.

### 4. What the equation gave

**35,840 of 35,840 floats identical. Max ulp 0.**

---

## The attention block, complete

Thirteen stages, run from the token ids with nothing re-seeded from the engine:

| # | stage | tap | result |
|---|---|---|---|
| 1 | embedding lookup | 31 | 35,840 / 35,840 |
| 2 | aggregation | — | skipped, guard false |
| 3 | snapshot push | 30 | state write exact |
| 4 | pre-attention RMSNorm | 2 | 35,840 / 35,840 |
| 5 | six projections | 28 | 61,440 / 61,440 |
| 6 | ShortConv + SiLU | 19, 20, 21 | 184,320 / 184,320 |
| 7 | per-head L2 norm | 22, 23 | 122,880 / 122,880 |
| 8 | beta | 24 | 480 / 480 |
| 9 | decay chain | 25, 29 | 122,880 / 122,880 |
| 10 | recurrence | 26 | 61,440 / 61,440 |
| 11 | head-wise RMSNorm | none | confirmed via stage 12 |
| 12 | gate | 27 | 61,440 / 61,440 |
| 13 | output projection | 3 | 35,840 / 35,840 |
| | **total** | | **686,560 / 686,560** |

Max ulp 0 at every site.

**What is still not measured**, as opposed to verified:

- `f_a`'s 128-dimensional intermediate, which is confirmed only by inference through `z`
- the int8 kernel's scalar tail loop, which no shape in this layer triggers, so that branch
  has never run

Both are gaps in coverage, not known defects, and neither is closed by anything above.

---

## Stage 14 — the residual update

### 1. The equation

$$r \;\leftarrow\; \begin{cases} r + \mathrm{attn.out} & \text{if } \texttt{have\_prefix} \\[2pt] \mathrm{attn.out} & \text{otherwise} \end{cases}$$

At this stage the second branch applies. There is no arithmetic at all — it is a copy.

### 2. What this stage exactly does

**It replaces the residual. It does not add to it.**

Stage 3 pushed the snapshot and set `have_prefix = 0`. Stage 14 reads that flag and takes the
`else` branch. The two are a pair: save the residual to the stack, then overwrite it.

So at this layer the embedding is *discarded from the residual stream*. It survives only
inside the snapshot stack, where stage 2 of later layers will blend it back in. The stream
restarts from the attention output alone. The residual is deliberately reset at block
boundaries rather than accumulated through them.

### 3. The real data

```
replacement        resid = attn.out          : [7168 x5] of 7168
control, addition  resid = embed + attn.out  : [0, 0, 0, 0, 0] of 7168
```

The addition gets **zero** floats right. This is not a near miss between two plausible forms,
it is a different operation.

Confirmed a second way, with my reconstruction out of the loop entirely:

```
engine site 4 vs engine site 3, independent of me: [7168 x5] of 7168
```

The engine's own `resid.post_attn` is bit-identical to its own `attn.out`. That is a `memcpy`,
visible in the engine's data whether or not anything of mine exists.

Magnitudes across the reset:

```
embedding L2       : [2.0446, 1.9179, 1.5717, 1.862,  1.8237]
resid.post_attn L2 : [0.3623, 0.4307, 0.75,   0.5855, 0.5717]
```

A drop to roughly 18 to 48% of the incoming magnitude.

### 4. What the equation gave

**35,840 of 35,840 floats identical.**

### A mislabeled line in my own output

The run printed a row labeled `snapshot S L2` containing 96 numbers. **That label is wrong.**
The stage 10 code assigns `S = np.zeros((H,D,D))` for the recurrence state, which shadowed the
`S` holding the stage 3 snapshot list. Those 96 values are per-head KDA state norms, not the
snapshot.

Nothing depends on it. The snapshot variable is never read again in these scripts, and stage 3
was verified directly against site 30 at the time. But the line asserts something it does not
show, so it is not used and not reproduced here.

That makes three instrument errors caught during this walk: stage 2's fitted-but-wrong
mechanism, stage 11's filter-artifact site list, and now a variable collision. All three were
mine. None were the engine's.

### Where the chain stands

| stages | identical |
|---|---|
| 1 to 13, the attention block | 686,560 / 686,560 |
| 14, the residual update | 35,840 / 35,840 |
| **total** | **722,400 / 722,400** |
