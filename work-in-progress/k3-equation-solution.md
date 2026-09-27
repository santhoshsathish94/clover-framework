# The equation, solved for what actually varies

## The question

`k3-model-equation.md` states the model as one composed expression and says of its
section 1.1:

> That is the entire input. Everything else is a constant of the model.

If that sentence is taken literally, then every symbol in the equation is either a function
of the input or it is not, and the ones that are not have a value that could be determined
once and never again. This document works through the equation symbol by symbol and decides
which is which.

**The method is the equation, not measurement.** Nothing here is established by timing a
run or by reading a byte count. A symbol is run-invariant if its defining expression
contains no input symbol, transitively. That is a property of the text, and it is decidable
by inspection.

The previous documents in this series asked where time goes and how the bytes move.
This one asks a question that does not depend on either: **what does a run actually have to
compute, and what has it already been told?**

---

## The classification

Every symbol gets exactly one of four labels, assigned by what its definition depends on.

| label | means | determined |
|---|---|---|
| **E** | fixed by the equation itself | before any model exists |
| **W** | a function of the weights and nothing else | when the checkpoint is chosen |
| **R** | depends on $T$ or on some $t_p$, transitively | every run |
| **S** | carried state: **R**-valued in general, with an **E** initial value | every run, evolving |

The rules that assign them:

1. A symbol whose definition mentions only numerals and other **E** symbols is **E**.
2. A symbol whose definition mentions only **E** and **W** symbols is **W**.
3. A symbol whose definition mentions any **R** or **S** symbol is **R**.
4. The input symbols $T$ and $t_p$ are **R** by definition.
5. A symbol defined by a recurrence over positions, with an initial value that is **E**, is
   **S**.

Rule 2 is the one that does work. A quantity can be written inside the per-run part of the
expression and still be **W**, because what decides the label is the *dependency*, not the
*position in the text*. Every such symbol is a value the equation permits to be computed
once and reused for every run for the lifetime of the checkpoint.

The interesting output of this document is therefore not the **E** list, which is trivial,
nor the **R** list, which is most of the equation. It is the **W** list: the symbols that
look like work and are not.

---

## Step 2 - the input, the shapes, the scalars, the layer sets

### 1.1, the input

| symbol | label | why |
|---|---|---|
| $T$ | **R** | rule 4 |
| $t_0 \dots t_{T-1}$ | **R** | rule 4 |

Two symbols. By the equation's own sentence, this is the whole of what varies, and the
rest of this document is an attempt to find anything that contradicts it.

### 1.2, the shape constants

All 22 are **E**: $E, H, D, P, V, N_L, K_c, N_{\text{exp}}, k, \text{LAT}, I, \text{SI},
D_I, Q_N, Q_R, Q_H, V_H, \text{KVL}, \text{KVW}, \text{KVD}, Q_{\text{lora}}, \text{GRP}$.

$P = HD$ is defined from two **E** symbols, so it is **E** by rule 1 rather than by being
written down as 12288.

One honest qualification. These are **E** only in the sense that they are part of the
equation's text. They are not universal constants - they describe this architecture, and a
different architecture would change them. What makes them **E** and not **W** is that
choosing a *different checkpoint of the same architecture* does not move them. The
distinction matters at step 8, where the question is what a precompute is keyed to.

### 1.3, the scalar constants

All 8 are **E**: $\epsilon_5, \epsilon_6, \lambda, b_1, b_2, \rho, q_{\text{sc}},
m_{\text{sc}}$.

**This is where the theme of the document first shows up.** Three of these are written as
expressions rather than as literals:

$$\epsilon_5 = \mathrm{fl}_{32}(10^{-5}), \qquad
q_{\text{sc}} = 1/\sqrt{128}, \qquad m_{\text{sc}} = 1/\sqrt{192}$$

A square root and a reciprocal each. An implementation that follows the text will evaluate
them where they appear, which for $q_{\text{sc}}$ and $m_{\text{sc}}$ is inside the
attention block, and therefore per layer and per position. Nothing in the equation asks for
that. They are numerals that happen to be written as arithmetic.

This is a trivial cost and is worth nothing on its own. It is recorded because it is the
smallest clean instance of the rule that decides everything else in this document: **where
a quantity is written tells you nothing about when it has to be evaluated.**

### 1.4, the layer type sets

All 4 are **E**: $\mathbb{M}, \mathbb{K}, \mathbb{E}, \mathbb{P}$. Each is a membership
predicate on the integer $L$, decidable with no model and no input.

Two consequences follow immediately, and neither needs a run to establish:

$$\mathbb{P} \cap \{0,\dots,92\} = \{0, 12, 24, 36, 48, 60, 72, 84\}, \qquad |\mathbb{P}| = 8$$

which is section 5's $|\mathcal{S}^{(93)}| = 8$, derived rather than observed. And more
generally, the number of sources the aggregation $\mathrm{AR}$ sees at layer $L$ is

$$\big|\mathcal{S}^{(L)}\big| + 1 = \big|\{p \in \mathbb{P} : p < L\}\big| + 1
= \left\lceil \frac{L}{12} \right\rceil + 1$$

a function of $L$ alone. **The shape of every aggregation in the model is E.** No input can
change how many sources are summed at any layer, so nothing about the aggregation's
structure - its source count, its weight-vector length, its softmax width - has to wait for
a run.

### Step 2 result

| group | symbols | E | W | R | S |
|---|---|---|---|---|---|
| 1.1 input | 2 | 0 | 0 | 2 | 0 |
| 1.2 shapes | 22 | 22 | 0 | 0 | 0 |
| 1.3 scalars | 8 | 8 | 0 | 0 | 0 |
| 1.4 layer sets | 4 | 4 | 0 | 0 | 0 |

Nothing in section 1.2 to 1.4 depends on the checkpoint, let alone on the run. The first
**W** candidates cannot appear until section 1.5, which is where the weights are.

---

## Step 3 - the weights, and the first real W list

### The weights themselves

All 45 symbols in the table of section 1.5 are **W** by definition: $W_E$, $w^{\text{in}}_L$,
$w^{\text{post}}_L$, $n^A_L$, $p^A_L$, $n^M_L$, $p^M_L$, $W^q_L$, $W^k_L$, $W^v_L$, $W^b_L$,
$W^{fa}_L$, $W^{fb}_L$, $C^q_L$, $C^k_L$, $C^v_L$, $A^{\log}_L$, $\tau_L$, $w^o_L$,
$W^{qa}_L$, $w^{qan}_L$, $W^{qb}_L$, $W^{ka}_L$, $w^{kan}_L$, $W^{kb}_L$, $W^g_L$, $W^o_L$,
$W^{\text{gate}}_0$, $W^{\text{up}}_0$, $W^{\text{down}}_0$, $G_L$, $\gamma_L$,
$W^{\downarrow}_L$, $W^{\uparrow}_L$, $w^{\ell}_L$, $W^{(e)}_1$, $W^{(e)}_3$, $W^{(e)}_2$,
$W^s_1$, $W^s_3$, $W^s_2$, $n^O$, $p^O$, $w^F$, $W_{\text{lm}}$.

That is not interesting. What is interesting is everything section 1.5 and section 2
*derive* from them.

### The three folds

Section 1.5 states them outright:

$$f^A_L = n^A_L \odot p^A_L, \qquad f^M_L = n^M_L \odot p^M_L, \qquad f^O = n^O \odot p^O$$

Every symbol on the right is **W**, so by rule 2 all three are **W**. They are elementwise
products of weight vectors, and their values cannot be affected by any input.

$$93 + 93 + 1 = 187 \text{ vectors of width } E = 7168 = 1{,}340{,}416 \text{ floats}$$

### The exponentiated decay coefficient

Section 3.1 defines, inside the KDA block:

$$a_h = \exp\big(A^{\log}_L[h]\big), \qquad h = 0 \dots 95$$

$A^{\log}_L$ is **W**, so $a_h$ is **W**. It sits inside the per-position attention
expression and depends on nothing that varies. $|\mathbb{K}| = 69$ layers carry it, at 96
values each:

$$69 \times 96 = 6{,}624 \text{ exponentials}$$

Section 6 of the equation document records that $A^{\log}$ is stored at width 128 with 96
read, so 32 of every 128 stored values are **W** and also never referenced at all.

### The derived-W total, exactly

| quantity | count | floats |
|---|---|---|
| $f^A_L$ | 93 vectors of 7168 | 666,624 |
| $f^M_L$ | 93 vectors of 7168 | 666,624 |
| $f^O$ | 1 vector of 7168 | 7,168 |
| $a_h$ | 69 sets of 96 | 6,624 |
| **total** | | **1,347,040** |

**5.39 MB.** That is the entire closed-form derived-**W** set of the equation, and it
contains 1,340,416 multiplications and 6,624 exponentials that no run can influence.

### The dtype conversions, which are also W but do not close so neatly

Section 1.5 says dtype selects the operator, and section 2 specifies each conversion:

- $\mathbb{B}$ widens BF16 to float32 "exactly (a shift, not a rounding)"
- $\mathbb{Q}$ converts each int8 to float inside the FMA, and applies the row scale $s_o$
  **after** the reduction
- $\mathbb{X}$ decodes each 4-bit code and multiplies by its group's scale
  **in float32 before** the accumulation:
  $\tilde{w}_{r,i} = \mathrm{fl}_{32}\big(\mathrm{E2M1}[c_{r,i}] \cdot 2^{\,s_{r,g(i)}-127}\big)$

Each of these is a function of weights alone, so each is **W**. $\tilde{w}$ in particular is
a fully-defined **W** array with the same shape as the expert weights.

But they do not behave like the folds, and the difference is the point:

| | source form | **W** form | ratio |
|---|---|---|---|
| the three folds and $a_h$ | 2 x BF16 per element | float32 | 1.0x |
| $\mathbb{Q}$'s int8 to float | int8 + row scale | float32 | ~4x |
| $\mathbb{X}$'s $\tilde{w}$ | 4-bit code + shared E8M0 scale | float32 | ~7.5x |

**Being W is necessary for precomputing something and is not sufficient.** The folds
collapse two stored vectors into one of the same width, so the **W** form is strictly
smaller than the source. The dequantizations expand, and a quantization format exists
precisely to make the stored form smaller than the computed one. Materializing $\tilde{w}$
turns 1.45 TB of experts into roughly 10.9 TB.

The test that decides it is not "is this **W**" but **"is fetching the W form cheaper than
fetching the source and recomputing it"**. That question needs a cost model, which is step
8. The classification here only establishes which quantities are eligible.

### Step 3 result

| group | symbols | E | W | R | S |
|---|---|---|---|---|---|
| 1.5 stored weights | 45 | 0 | 45 | 0 | 0 |
| derived: folds | 3 | 0 | 3 | 0 | 0 |
| derived: $a_h$ | 1 | 0 | 1 | 0 | 0 |
| derived: dtype conversions | 3 | 0 | 3 | 0 | 0 |

Still no **R** symbol anywhere except the two in 1.1. Every quantity the equation has
introduced so far is fixed before the run starts.

---

## Progress

| step | section of the equation | status |
|---|---|---|
| 1 | scheme | done |
| 2 | 1.1 input, 1.2 shapes, 1.3 scalars, 1.4 layer sets | done |
| 3 | 1.5 weights and the three folds | done |
| 4 | section 2, the ten operators | |
| 5 | section 3, the attention blocks | |
| 6 | section 4, the MLP and MoE blocks | |
| 7 | section 5, composition, initial conditions, carried state | |
| 8 | the boundary, and what it costs | |
