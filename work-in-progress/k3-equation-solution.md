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

## Step 4 - the ten operators

### Every reduction schedule is E

Section 2 opens by saying association order is part of each definition. Each of those
orders is a fixed assignment of indices to lanes and a fixed tree, mentioning no weight and
no input:

| operator | the **E** structure |
|---|---|
| $\mathbb{Q}$ | 16 lanes $B_c$ on $i \equiv c \pmod{16}$; $A_j = B_j + B_{j+8}$; $((A_0{+}A_4)+(A_2{+}A_6)) + ((A_1{+}A_5)+(A_3{+}A_7))$ |
| $\mathbb{X}$ | 4 double vectors of 4 lanes, $V_m$ lane $c$ on $i \equiv 4m+c \pmod{16}$; $Q_c$ tree; $(Q_0{+}Q_2)+(Q_1{+}Q_3)$ |
| $\mathbb{B}$ | 16 double lanes on $i \equiv c \pmod{16}$; $U_c = (C_c{+}C_{c+4})+(C_{c+8}{+}C_{c+12})$ |
| $\mathcal{N}$ | double sum, one float32 rounding of the reciprocal square root, $(w_i x_i)$ then $\times\,\mathrm{inv}$ |
| $\mathcal{L}$ | double sum over 128, $\epsilon_6$, no division by $n$ |
| $\mathrm{SiTU}$ | $b_1, b_2$; sigmoid takes the uncapped $g$ |
| $\mathcal{C}$ | current input first, then history oldest to newest; $K_c = 4$ |
| $\mathrm{AR}$ | double sum of squares, float32 scored product, double $z$, float32 source-major weighted sum |
| $\Delta$ | sequential in $i$ ascending, no FMA, output read from the updated state |
| $\mathrm{SA}$ | causal $s \le t$, double sequential dot over the full 192, $m_{\text{sc}}$ over 192 |

None of this can vary. The hard part of this model - the part `k3-stages.md` records three
separate association-order failures for - is entirely **E**.

Two operators, $\mathcal{L}$ and $\sigma/\mathrm{SiTU}$, take **no W input at all**. They
are pure **E** structure applied to **R** data.

### $\mathbb{Q}$'s scalar tail is provably dead, from section 1.2 alone

$\mathbb{Q}$ carries a tail over $i \ge 16\lfloor n/16 \rfloor$. Section 6 reports that it
never executes. That is stated there as an observation; it is actually derivable, because
the predicate depends only on **E** shape constants. Every input width $\mathbb{Q}$ is
applied to in sections 3 and 4:

$$7168,\ 128,\ 12288,\ 1536,\ 512,\ 33792,\ 3584,\ 6144$$

and $16 \mid n$ for all eight. The tail is unreachable for this architecture, **decidable
without a model and without a run**. The same check on $\mathbb{X}$'s widths, 3584 and
3072, shows both divisible by $\mathrm{GRP} = 32$, so its group loop is exact too.

### The dequantization map is a 4096-entry E table

This is the one that matters. $\mathbb{X}$ decodes

$$\tilde{w}_{r,i} = \mathrm{fl}_{32}\big(\mathrm{E2M1}[c_{r,i}] \cdot 2^{\,s_{r,g(i)}-127}\big)$$

Step 3 labeled $\tilde{w}$ as **W** and noted that materializing it expands the model
7.5x. But look at the *domain* rather than the array:

- $c$ is a 4-bit code, so it takes **16** values
- $s$ is an E8M0 byte, so it takes **256** values
- $\mathrm{E2M1}[\cdot]$ is a fixed 16-entry table, **E**
- $2^{\,s-127}$ is a fixed function of a byte, **E**

The map $(c, s) \mapsto \tilde{w}$ therefore has a domain of exactly $16 \times 256 = 4096$
pairs, and both of its factors are **E**. **The dequantization map is not W at all - it is
E.** It can be tabulated before any checkpoint exists:

$$\mathrm{DQ}[s][c] = \mathrm{fl}_{32}\big(\mathrm{E2M1}[c] \cdot 2^{\,s-127}\big),
\qquad 4096 \text{ float32} = 16 \text{ KB}$$

and since $g(i) = \lfloor i/32 \rfloor$ is also **E**, the scale index is constant across
each group of 32, so a group reduces to selecting the row $\mathrm{DQ}[s]$ once and then
one indexed load per element - no exponential, no multiply.

This is bit-exact by construction: the tabulated entry is the value of the same expression,
so it is the same float32.

What step 3's array view missed is that a **W** array can be the image of an **E** function
over a small domain. $\tilde{w}$ has $2.72 \times 10^{12}$ elements and **4096 distinct
possible values**. The array is enormous and **W**; the map is 16 KB and **E**.

### Step 4 result

| | count | label |
|---|---|---|
| reduction schedules, lane maps, trees | 10 | **E** |
| $\mathbb{Q}$ tail predicate | 1 | **E**, and false for every shape here |
| $g(i) = \lfloor i/32 \rfloor$ | 1 | **E** |
| $\mathrm{E2M1}$ table | 1 | **E** |
| $\mathrm{DQ}[s][c]$ dequantization map | 1 | **E**, 16 KB |
| $\mathcal{L}$, $\sigma$, $\mathrm{SiTU}$ | 3 | **E**, no **W** input |
| row scale $s_o$, taps, norm weights, $f$ | - | **W**, already counted in step 3 |

Still nothing **R** beyond the two input symbols. Four steps in, the equation has
introduced ten operators and not one of them has a structure that depends on the run.

---

## Step 5 - the attention blocks

### Every value is R; every detail is E

Both blocks take $x_1$, which section 5 derives from the residual, which derives from the
embedding gather, which derives from $t_p$. So by rule 3 every quantity in section 3 is
**R**: $q, k, v, \beta, z, \alpha, o, \mathrm{attn.out}$ in KDA, and
$q, c, \mathrm{kv}, k^{\text{lat}}, k^{\text{rope}}, v, \mathrm{acc}, \mathrm{attn.out}$
in MLA. The conv history and the KDA state $S$ and the KV cache are **S**.

Every one of the details the equation warns about is **E**: the norm on $c$ covering the
latent 512 only, the rope slot being unrotated yet scored, $m_{\text{sc}}$ being over the
full 192, the gate applying before the output projection in MLA and after the norm in KDA,
$A^{\log}$ per head against $\tau$ per channel. These are structural facts and no input
touches them.

### One E fact that licenses a reordering

The equation notes of KDA: **the gate reads $x_1$, not anything attention produced**. The
same holds in MLA. So $\sigma(\mathbb{Q}[W^g_L]\,x_1)$ depends on nothing the recurrence
computes, and can be evaluated at any point after $x_1$ exists.

That is a dependency fact, visible in the equation, and it is exactly what licenses
hoisting the gate projection out of the position loop. It does not change a label - the
gate is still **R** - but it shows the equation carries scheduling information that a
naive left-to-right reading discards.

### A derived E bound

$\alpha_{hD+d} = \exp\big(\lambda\,\sigma(\cdot)\big)$ with $\lambda = -5$ and
$\sigma \in (0,1)$, so for every input and every position

$$\alpha \in \big(e^{-5},\ 1\big) \approx (0.00674,\ 1)$$

The decay is bounded away from zero by an **E** constant. Nothing here needs it, but it is
a property of the model derived from the equation rather than sampled from a run.

### Why $\sigma$ is not tabulatable and $\mathrm{DQ}$ was

Step 4 turned $\mathbb{X}$'s decode into a 16 KB table because both its factors were **E**
over a domain of 4096. The same reasoning applied to $\sigma$, or to the composite
$u \mapsto \exp(\lambda\sigma(u))$, fails - not because they are less **E**, but because
their argument is an arbitrary float32:

| **E** function | domain | table |
|---|---|---|
| $(c,s) \mapsto \mathrm{E2M1}[c]\cdot 2^{s-127}$ | $16 \times 256 = 4096$ | 16 KB |
| $u \mapsto \sigma(u)$ | $2^{32}$ | 16 GB |
| $u \mapsto \exp(\lambda\sigma(u))$ | $2^{32}$ | 16 GB |

**The criterion is not whether a function is E, but whether its domain is small.** Every
pointwise nonlinearity in this equation is **E** and none of them is tabulatable bit-exactly,
because they consume float32 rather than a quantization code. $\mathrm{DQ}$ was reachable
only because a 4-bit code and an 8-bit exponent are between them a 12-bit index.

### The one linear-linear composition, and why folding it is wrong twice over

Most adjacent projections in the equation have something nonlinear between them. Exactly
one pair does not:

$$z = \mathbb{Q}[W^{fb}_L]\ \big(\mathbb{Q}[W^{fa}_L]\,x_1\big), \qquad E \to 128 \to 12288$$

Algebraically $W^{fb}W^{fa}$ is a single $12288 \times 7168$ map, and it would be **W**.
It fails on both counts that matter:

- **Not bit-exact.** $\mathbb{Q}$ is not exact linear algebra. It is a 16-lane float32
  reduction with the row scale applied after the tree, and it rounds the 128-wide
  intermediate to float32 before the second projection. A folded matrix removes that
  rounding and changes the association order, which section 2 says is part of the
  definition.
- **Larger, not smaller.** $128 \times 7168 + 12288 \times 128 = 2{,}490{,}368$ parameters
  become $12288 \times 7168 = 88{,}080{,}384$, a **35.4x expansion**.

The same shape appears at $W^{qa} \to W^{qb}$ through 1536, $W^{ka} \to W^{kb}$ through
512, and $W^{\downarrow} \to W^{\uparrow}$ through 3584, but all three have an
$\mathcal{N}$ or an entire expert network in between, so they are not even algebraically
foldable.

**The low-rank factorizations in this equation are already the compressed form.** Step 3
found dequantization expands; this finds that undoing the rank structure expands too. The
model as written is at a local minimum of storage, and the two obvious ways to trade
compute for space both run the wrong way.

### Step 5 result

| group | E | W | R | S |
|---|---|---|---|---|
| 3.1 KDA values | 0 | 0 | 7 | 2 (conv history, $S$) |
| 3.2 MLA values | 0 | 0 | 8 | 1 (KV cache) |
| structural details of both | all | 0 | 0 | 0 |

No new **W** symbols. Section 3 introduces no quantity that a run could skip.

---

## Step 6 - the MLP and MoE blocks

### The values

Everything in section 4 descends from $x_2$, so $s_e$, $\mathcal{J}$, $\pi$, $\zeta$,
$\mathrm{accL}$ and $\mathrm{ffn.out}$ are all **R**. The structural rules are all **E**:
the bias participating only in selection, top-$k$ by repeated maximum with a strict `>` so
ties go to the lowest index, float32 expert-major accumulation, the norm applied to the
aggregate once rather than per expert, the shared expert added unweighted.

One **W** item the routing hides: section 4.2 says the router is **not** $\mathbb{Q}$ - the
int8 gate is widened per row and the dot accumulates in double, sequentially. That widening
of $G_L$ is a function of weights alone, so **W**, and it cannot borrow $\mathbb{Q}$'s
kernel because the reduction differs.

### Two more scheduling facts the equation gives away

$$\zeta = \mathbb{Q}[W^{\downarrow}_L]\,x_2 \qquad\text{and}\qquad
\mathbb{Q}[W^s_2]\ \mathrm{SiTU}\big(\mathbb{Q}[W^s_1]x_2,\ \mathbb{Q}[W^s_3]x_2\big)$$

Neither mentions $\mathcal{J}$ or $\pi$. **The expert latent and the entire shared-expert
branch are independent of the router**, so both can be evaluated before routing finishes.
Section 4.2 makes the second explicit - the shared expert "bypasses the router" - and the
first follows from reading the dependency.

### The operator application counts are E, and they check out exactly

The count of times each operator is applied is a function of the layer sets, $k$, and $T$.
None of it depends on the input's content. From sections 3 and 4:

$$
\begin{aligned}
\mathbb{Q}\ \text{per position} &= \underbrace{69 \times 8}_{\text{KDA}} + \underbrace{24 \times 6}_{\text{MLA}}
+ \underbrace{3}_{\text{dense } L=0} + \underbrace{92 \times 5}_{\text{MoE}} = 1159 \\
\mathbb{X}\ \text{per position} &= 3 \times k \times |\mathbb{E}| = 3 \times 16 \times 92 = 4416 \\
\text{router, top-}k\ \text{per position} &= |\mathbb{E}| = 92 \\
\mathbb{B} &= 1 \quad \text{(last position only)}
\end{aligned}
$$

At $T = 5$ that predicts 5795, 22080, 460, 460 and 1. The instrumented run in
`k3-data-problem.md` step 7 measured:

| operator | predicted from the equation | measured |
|---|---|---|
| $\mathbb{Q}$ | 5,795 | 5,795 |
| $\mathbb{X}$ | 22,080 | 22,080 |
| router | 460 | 460 |
| top-$k$ | 460 | 460 |
| $\mathbb{B}$ | 1 | 1 |

Exact, on all five. The counts were derived here from the layer sets and never needed
measuring; that they agree is a check on the instrumentation rather than a discovery about
the model.

The batched form checks too. Batching removes the position factor from every $\mathbb{Q}$
except layer 0's dense MLP, which step 8 of the other document did not convert:

$$1159 + (3 \times 5 - 3) = 1159 + 12 = 1171$$

against 1,171 measured.

### Where E stops and R begins, in one example

$\mathbb{X}$'s **unbatched** count is $3kT|\mathbb{E}|$ - pure **E** given $T$, because
every position evaluates exactly $k$ experts whatever it routes to. Its **batched** count
is $3|\mathbb{E}|\sum_L |\{\text{distinct experts at } L\}|$, which depends on how much the
positions' choices overlap, and that is **R**. Measured 17,049, and the equation cannot
predict it.

So the same operator has an **E** application count in one schedule and an **R** count in
another. **Batching converted an input-independent amount of work into an
input-dependent one** - which is why it saved anything at all, and also why its saving
cannot be stated as a constant.

### Step 6 result

| group | E | W | R | S |
|---|---|---|---|---|
| 4.1 dense values | 0 | 0 | 1 | 0 |
| 4.2 routing values | 0 | 0 | 3 | 0 |
| 4.2 compute and combine | 0 | 0 | 3 | 0 |
| $G_L$ widening | 0 | 1 | 0 | 0 |
| structural rules | all | 0 | 0 | 0 |
| operator application counts | all | 0 | 0 | 0 |

---

## Progress

| step | section of the equation | status |
|---|---|---|
| 1 | scheme | done |
| 2 | 1.1 input, 1.2 shapes, 1.3 scalars, 1.4 layer sets | done |
| 3 | 1.5 weights and the three folds | done |
| 4 | section 2, the ten operators | done |
| 5 | section 3, the attention blocks | done |
| 6 | section 4, the MLP and MoE blocks | done |
| 7 | section 5, composition, initial conditions, carried state | |
| 8 | the boundary, and what it costs | |
