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
