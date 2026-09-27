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

## Step 7 - the composition, the state, and the one branch that is not E

### Initial conditions

$$r^{(0)}_p = W_E[t_p], \qquad \mathcal{S}^{(0)} = \varnothing, \qquad S_L = 0,
\qquad \texttt{cached} = 0$$

Three of the four are **E**. The fourth is the only place in the entire equation where the
input enters: $W_E$ is **W**, $t_p$ is **R**, and $r^{(0)}_p$ is a **gather**, not a
computation. **Everything downstream is a deterministic function of $r^{(0)}$.** The model
has exactly one input port and it is a table lookup.

### Every control-flow decision in the composition is E

Taking the six steps of $\mathrm{Layer}_L$ in turn, and separating the decision from the
value:

| step | the decision | label | the value |
|---|---|---|---|
| (1) | $\mathcal{S} \neq \varnothing$ | **E** - true iff $L \ge 1$, since $0 \in \mathbb{P}$ | $h$ is **R** |
| (2) | $L \in \mathbb{P}$ | **E** | pushes an **R** vector |
| (3) | - | - | $x_1$ is **R** |
| (4) | $L \in \mathbb{P}$, replace or add | **E** | $r$ is **R** |
| (5) | unguarded | **E** | $x_2$ is **R** |
| (6) | unconditional | **E** | $r$ is **R** |
| 3 | $L \in \mathbb{M}$ or $\mathbb{K}$ | **E** | attention out is **R** |
| 4 | $L = 0$ or $L \in \mathbb{E}$ | **E** | ffn out is **R** |

Step (4) is the one the equation flags as not looking like a residual stream, and even that
is decided by an **E** predicate on $L$.

### The single data-dependent branch

Searching the whole equation for a decision whose outcome depends on a value rather than on
an index, there is exactly one:

$$\mathcal{J} = \operatorname*{top-}k_{\ e}\ \big(s_e + \gamma_{L,e}\big)$$

Nothing else qualifies. The causal mask $s \le t$ is an index comparison. The maxima inside
$\mathrm{AR}$ and $\mathrm{SA}$ are value comparisons but they only shift a softmax; they do
not change what is computed. The layer type, the push schedule, the residual branch, the
dense-versus-MoE choice, the operator selected by dtype - all **E** predicates on $L$.

**$\mathcal{J}$ is the sole data-dependent branch in the model.** Which gives the answer
this document was written to find:

> Two runs with the same $T$ execute the *same operators, on the same shapes, in the same
> order*, and differ in exactly two ways: every numeric value, and the identity of
> $T \times 92 \times 16$ expert indices.

That is what changes per run. Not derived from any measurement - it is what the dependency
structure of the equation permits to change.

### A second provably dead computation

The boxed tail says the aggregation is evaluated at every position, while the norm, the
lm_head and the argmax are evaluated at $p = T-1$ only. Nothing consumes the other
positions' aggregates: section 5's composition has ended, and decode re-enters with a fresh
$\mathcal{S}$ and a new embedding rather than with these vectors.

So $\mathrm{AR}$ at the tail runs $T-1$ times for nothing. At $T=5$, with 9 sources of
width 7168, that is 4 unused aggregations, about $7.7 \times 10^5$ operations. Trivial in
size, and provably unnecessary **from the equation**, which is the same standing as step 4's
finding that $\mathbb{Q}$'s scalar tail is unreachable. Two pieces of dead work, both found
by reading rather than profiling.

### The carried state

| symbol | label | note |
|---|---|---|
| $S_L$, conv history, KV cache | **S** | carry across calls; initial values **E** |
| $\mathcal{S}$ | **R** | **not** carried - section 5 says it is rebuilt from scratch every call |
| $\texttt{cached}$ | **R** | $0$ on prefill, $T$ on the next call; an index, not a value |

$\mathcal{S}$ being rebuilt is the one place where a symbol that looks like state is not.
Each call aggregates over its own eight snapshots.

### Step 7 result

| group | E | W | R | S |
|---|---|---|---|---|
| initial conditions | 3 | 0 | 1 | 0 |
| layer control decisions | 8 | 0 | 0 | 0 |
| layer values | 0 | 0 | 6 | 0 |
| carried state | 0 | 0 | 2 | 3 |
| data-dependent branches | - | - | **1** | - |

---

## Step 8 - the boundary, and what it costs

### The answer, stated plainly

**What must be computed during a run:** the **R** set, and nothing else. That is every
numeric value downstream of the embedding gather - each $x_1$, $x_2$, $q$, $k$, $v$,
$\alpha$, $\beta$, $o$, $s_e$, $\pi$, $\zeta$, $\mathrm{accL}$, $r$, and the logits.

**What is already determined before the run:** everything else. Every shape, every scalar,
every layer-set membership, every reduction order and association tree, every control-flow
decision, every operator application count, all 45 weights, and every quantity derivable
from weights alone.

**What differs between two runs with the same $T$:** the values, and one set of indices.
$T \times 92 \times 16$ expert choices. Nothing else about the execution differs - not
which operators run, not their shapes, not their order, not how many times.

### The tally

| category | contents |
|---|---|
| **E** | 22 shapes, 8 scalars, 4 layer sets, 10 reduction schedules, $\mathrm{E2M1}$, $\mathrm{DQ}$, $g(i)$, the $\mathbb{Q}$-tail predicate, 3 initial conditions, 8 control decisions |
| **W** | 45 stored weights, 3 folds, $a_h$, 3 dtype conversions, the $G_L$ widening |
| **R** | 2 input symbols, and every value in sections 3, 4 and 5 |
| **S** | $S_L$, conv history, KV cache |

### The test that decides a precompute

Being **E** or **W** makes a quantity *eligible* to be computed once. It does not make it
*worth* computing once. The condition is

$$\mathrm{cost}\big(\text{fetch the stored form}\big) \;<\;
\mathrm{cost}\big(\text{fetch the source}\big) + \mathrm{cost}\big(\text{recompute}\big)$$

and on a machine where fetch dominates - which
`k3-data-problem.md` step 7 measured, with $\mathbb{Q}$ pinned at the RAM ceiling - the
left side is usually the whole story. Applying it to every candidate this document found:

| candidate | source | **W**/**E** form | verdict |
|---|---|---|---|
| $\mathrm{DQ}[s][c]$ | - | 16 KB, **E** | **yes.** costs nothing, removes ~$2\times10^{11}$ multiplies and ~$6.2\times10^9$ `exp2f` per run |
| $a_h$ | 512 B/layer (F32, width 128) | 384 B/layer | **yes.** smaller *and* removes 6,624 exponentials |
| $f^A_L,\ f^O$ | 2 x BF16 = 28,672 B | F32 = 28,672 B | **byte-neutral.** removes 673,792 multiplies, no I/O change |
| $f^M_L$ | 2 x I8R $\approx$ 14,344 B | F32 = 28,672 B | **no.** doubles the fetch to save 666,624 multiplies |
| $G_L$ widened | int8, 6.4 MB/layer | F32, 25.7 MB/layer | **no.** 4x the fetch, and the router is 0.5% of wall |
| $\tilde{w}$ materialized | 1.45 TB | ~10.9 TB | **no.** 7.5x |
| $W^{fb}W^{fa}$ folded | 2.49 M params | 88.1 M params | **no**, and not bit-exact either |

Only two pass, and the interesting one passes for a reason none of the others share.

### Why almost nothing passes, and what the exception tells you

The pattern is not a coincidence. **A quantized checkpoint is already stored in the form
that minimizes fetch cost** - that is what quantization is for. So any derived quantity
that is numerically *wider* than its source loses the test automatically, and every
per-element derived quantity here is wider: int8 to float is 4x, MXFP4 to float is 7.5x,
an I8R pair folded to float32 is 2x. Step 5 found the same thing from the other side: the
low-rank factorizations are already compressed, and undoing them expands 35x.

$\mathrm{DQ}$ escapes because it is not keyed on the weights at all. Its domain is the
*quantization alphabet* - 16 codes by 256 exponents - not the $2.72\times10^{12}$ elements
that draw from it. The array is **W** and enormous; the map is **E** and 16 KB.

> **"Compute it once" pays when the thing computed once is small. In a quantized model
> every per-element derived quantity is larger than its source, so the only precomputes
> that win are the ones keyed on the alphabet rather than on the weights.**

That is the result of this document, and it was reached without running anything.

### Two pieces of dead work, both found by reading

- $\mathbb{Q}$'s scalar tail is unreachable: all eight input widths are divisible by 16
  (step 4). Removing it saves no time, only a branch.
- The tail $\mathrm{AR}$ is evaluated at all $T$ positions and consumed at one (step 7).
  At $T=5$, four aggregations of nine sources, about $7.7\times10^5$ operations.

Neither is worth anything numerically. Both are recorded because they were found by
inspecting the equation, and neither profiling pass in `k3-data-problem.md` surfaced
either - a profiler shows you what ran, not what needn't have.

### What step 8 does not establish

- **Nothing here was measured, by design.** Every verdict above is analytic. The
  $\mathrm{DQ}$ saving in particular is a prediction: it says $\mathbb{X}$'s inner loop
  becomes one indexed load per element, and the effect on wall time is untested.
- **The cost model is one machine's.** "Fetch dominates" was measured on the Hetzner box
  in the other document. On hardware with a different bandwidth-to-FLOP ratio the verdicts
  for $f^M_L$ and $G_L$ could invert. The classification is machine-independent; the
  verdicts are not.
- **$\mathrm{DQ}$ is bit-exact by construction but has not been built**, so that claim
  rests on the entries being the value of the same expression and nothing more.
- **"$T \times 92 \times 16$ indices" counts slots, not distinct experts.** How much those
  overlap is an **R** quantity and the equation says nothing about it.
- **Sections 3.2 and the tail were transcribed from the walker**, not from `k3-stages.md`.
  Any classification of those symbols inherits that provenance.

---

## Step 9 - building DQ, and measuring the one verdict worth testing

Step 8 said $\mathrm{DQ}$ was the only precompute that clearly passes, and that the claim
was a prediction. This step builds it.

### Verifying the table before it goes near the model

All 4096 pairs, comparing the tabulated entry against the same expression recomputed at
use time, by **bit pattern** rather than by value:

```
pairs bit-identical : 4096 / 4096
entries +-inf       : 12
entries subnormal   : 8
entries negative 0  : 263
table size          : 16384 bytes
```

**The 263 negative zeros are the reason to compare bits and not values.** $\mathrm{E2M1}[8]$
is $-0.0$, and at $s = 255$ the expression is $\mathrm{E2M1}[c] \times 0.0$, which is
$-0.0$ for every negative code. A table written as "zero when $s=255$" would compare equal
under `==` and be wrong. Writing the table as the literal expression preserves it.

The 12 infinities and 8 subnormals are unreachable: the sweep of step 4 in
`k3-data-problem.md` shows the model's scale bytes span **109 to 124**, so the largest
representable magnitude is $6 \times 2^{-3} = 0.75$ and no product can overflow.

### A correction to step 8's estimate, found by reading the code

Step 8 predicted the change removes about $6.2\times10^9$ `exp2f` calls, computed as one
per group of 32. The implementation recomputed `e8` every **16** elements, twice per group,
so the real count was about $1.17\times10^{10}$.

Per run the kernel decodes

$$5{,}683 \text{ expert loads} \times 3 \times 11{,}010{,}048 = 1.877\times10^{11}\ \text{codes}$$

so the change removes $1.877\times10^{11}$ multiplies and $1.17\times10^{10}$ `exp2f`
calls. **Part of the saving is therefore not tabulation at all** - it is removing a
redundant recomputation the equation never asked for. The two are not separated below.

### Measured

Cold cache, threads pinned, three runs each, identical binary except the kernel:

```
                 X (s)                    mean      wall (s)                mean
baseline     12.256  12.100  12.215      12.190   28.17  27.69  27.80      27.89
with DQ       8.843   9.092   9.087       9.007   24.48  24.96  24.67      24.70
                                         -26.1%                            1.13x
```

All three runs 171,008/171,008 identical to the pre-change baseline, token 17374. The only
`exp2f` left anywhere in the program is the one inside `dq_init`.

### End to end

```
                                              inner wall      process total
eq.c, untouched                             53.30  52.82      56.97  55.56  -> 56.27
batched + prefetch + trunk RAM + pinned + DQ 24.81  24.54      32.56  32.29  -> 32.43
```

**1.74x end to end, 2.15x on the timed region, bit-identical throughout.**

$\mathbb{X}$ has gone from 43% of wall to 36%, and the prefetch is now the largest single
cost at 52%.

### What step 9 does not establish

- **The saving is not cleanly attributable.** Removing the redundant `exp2f` and
  tabulating the decode landed in one change and were not measured separately.
- **$\mathbb{X}$ is still scalar.** The 4x4 double accumulators are untouched, so the SIMD
  question is open and independent of this.
- **The table is 16 KB but only 256 entries are reachable** on this checkpoint, since 16
  scale bytes occur. That is a fact about the data, not the equation, and the equation
  correctly says 4096 - a different checkpoint could use any byte.
- **One prompt, prefill only**, as everywhere in this series.

### What step 9 says about the method

Step 8 reached this by reading the equation and asking what the domain of a function was,
not by profiling. The profiler in `k3-data-problem.md` step 7 had already reported
$\mathbb{X}$ as 43% of wall and compute-bound - it could say *that* the decode was
expensive, and it could not say *that the decode has 4096 possible answers*. One of those
facts is in the measurement and the other is only in the equation.

---

## Step 10 - vectorizing $\mathbb{X}$, and a gap in the equation it exposed

Step 8 of `k3-data-problem.md` argued that $\mathbb{X}$'s four 4-lane double accumulators
map exactly onto four AVX2 256-bit double registers, one lane per lane, and would therefore
be bit-exact. This step builds it.

### The thing that had to be checked first

The equation fixes which inputs land in which lane and in what order they combine. It does
**not** say whether the multiply and the add are fused. That distinction is invisible in
the mathematics and decisive in the result: a fused multiply-add rounds once, a separate
multiply and add round twice.

Compiling the exact accumulate expression both ways:

```
gcc -O2 -march=native -ffp-contract=off   ->   vmulsd + vaddsd     two roundings
gcc -O2 -march=native                     ->   vfmadd213sd         one rounding
```

The verified build carries `-ffp-contract=off`, so **the reference is multiply then add**,
and `_mm256_fmadd_pd` would have silently changed every value in the model. The SIMD kernel
uses `_mm256_mul_pd` followed by `_mm256_add_pd`, and the build was checked to contain zero
`vfmadd` instructions inside `Xm`.

### This is a genuine omission in the equation document

Reading section 2 again with this in hand, the other operators are specific and
$\mathbb{X}$ is not:

| operator | what it says about fusion |
|---|---|
| $\mathbb{Q}$ | "each accumulated by **single-rounded FMA** over inputs taken in blocks of 16" |
| $\Delta$ | "all float32, every sum accumulated sequentially in $i$ ascending, **no FMA**" |
| $\mathrm{AR}$ | specifies every rounding and widening explicitly |
| $\mathbb{X}$ | **silent** |

$\mathbb{X}$ is the one operator where fusion is unstated, and it is the one operator where
getting it wrong is easiest, because the natural SIMD translation of "accumulate into a
double vector" is `fmadd`. An implementer following the document alone would have a
plausible, wrong kernel. `k3-model-equation.md` has been corrected.

That is a third defect in that document, found the same way as the two its section 6
already records - by implementing from it and having to decide something it did not say.

### Measured

Cold cache, pinned, three runs:

```
                   X (s)                   mean     wall (s)                mean
scalar + DQ     8.843  9.092  9.087       9.007   24.48  24.96  24.67     24.70
SIMD + DQ       6.995  7.007  7.015       7.006   22.71  22.64  22.69     22.68
                                         -22.2%
```

All three runs 171,008/171,008 identical, token 17374.

### End to end

```
                                        process total
eq.c, untouched          54.76 - 59.83 over 7 runs      mean 56.33
final                    30.84  30.67                   mean 30.76
```

**1.83x end to end**, 2.34x on the timed region, bit-identical.

### Where the cost sits now

$\mathbb{X}$ has gone 12.190 -> 9.007 -> 7.006 s across the last two changes, and moves
99.72 GB at 14.24 GB/s against the 47.80 GB/s RAM ceiling - **still compute-bound, by about
3.4x**. But it is no longer the largest item:

```
prefetch (pure I/O)   12.83 s    56.6% of wall
X                      7.01 s    30.9%
Q                      1.35 s     5.9%
everything else        1.19 s     5.2%
```

The prefetch is now the dominant cost, and Step 6 of `k3-data-problem.md` already measured
that it runs at ~8 GB/s on the buffered path against 14.5 GB/s available with O_DIRECT.

### What step 10 does not establish

- **Register pressure was not investigated.** `__m256d acc[NPOS][4]` is 20 vectors against
  16 architectural YMM registers, so the compiler is spilling. Tiling over positions might
  do better and was not tried.
- **AVX-512 is available on this CPU and unused.** The lane structure is 4 doubles, which
  is 256 bits, so using 512-bit registers would require packing two positions per register
  and was not attempted.
- **The gain is measured at $T=5$ only**, and the kernel's arithmetic intensity depends on
  $T$.
- **One prompt, prefill only.**

---

## Step 11 - O_DIRECT expert reads

After step 10 the prefetch was 56.6% of wall, the largest single item, and step 6 of
`k3-data-problem.md` had already measured why: it runs on the buffered path at ~8 GB/s
while the same disk gives 14.4 GB/s with O_DIRECT.

### Why this needed more than a flag

**O_DIRECT cannot warm an mmap.** It bypasses the page cache, which is the only thing an
mmap reads from, so the two are mutually exclusive by construction. The bytes have to land
somewhere the kernels can address directly, which means an arena and a change to how every
expert pointer is resolved.

Three things the code had to handle that the idea does not mention:

- **Safetensors offsets are not 4096-aligned**, and O_DIRECT requires alignment on offset,
  length and buffer. Each range reads the aligned superset
  $[\lfloor \text{off}/4096 \rfloor \cdot 4096,\ \lceil (\text{off}+n)/4096 \rceil \cdot 4096)$
  and the usable pointer is that base plus the remainder.
- **The final aligned block can run past EOF**, giving a short read that is still complete
  for the bytes actually wanted. The code checks against the needed length rather than the
  aligned length, and falls back to a buffered `pread` if it comes up short.
- **Six call sites address expert bytes**, all previously `file_ptr(fid) + off`. They now
  go through a resolver that binary-searches the layer's range table and **falls back to
  the mmap** if a range is missing, so a bug in the arena cannot silently produce wrong
  bytes - it produces slow correct ones.

### Measured

Same binary, both modes, cold cache, pinned, three runs each:

```
                     wall (s)              mean    prefetch (s)          mean   major faults
mode 1, buffered   23.30 23.27 23.37      23.31   12.94 12.93 13.03    12.968        37,342
mode 3, O_DIRECT   16.51 16.52 16.50      16.51    7.05  7.04  7.04     7.043            27
```

**Prefetch down 45.7%.** 99.72 GB in 7.043 s is **14.16 GB/s** - the device rate from
step 6, now reached by the model on the data the model actually needs. Wall **1.41x**.

Major faults fall from 37,342 to **27**: the mmap is essentially untouched, which is the
direct confirmation that the arena is carrying the bytes and not the page cache.

All six runs 171,008/171,008 identical, token 17374.

### One thing I could not explain

Mode 1 on this binary measures 23.31 s, against 22.68 s for the same mode on the step 10
binary - a consistent 0.6 s that appeared with this change. The likely cause is the
resolver indirection replacing a direct pointer add, but **I did not establish it**. The
A/B above is unaffected, because both modes ran on one binary; the comparison against
step 10 carries that 0.6 s of unexplained difference.

### End to end

```
                                          process total
eq.c, untouched     9 runs, 54.76 - 59.83          mean 56.53
final                      22.92  22.83            mean 22.88
```

**2.47x end to end, 3.21x on the timed region, bit-identical.**

### Where the cost sits now

```
prefetch (pure I/O)    7.04 s   42.7%
X                      6.82 s   41.3%
Q                      1.35 s    8.2%
everything else        1.01 s    6.1%
unattributed           0.29 s    1.7%
```

The two remaining costs are now within 3% of each other. $\mathbb{X}$ moves 99.72 GB at
14.62 GB/s against a 47.80 GB/s memory ceiling, so it is still compute-bound by ~3.3x. The
prefetch is now **at** the device ceiling, so it cannot be improved by reading faster -
only by reading less, which is step 2 of `k3-data-problem.md`, the $k=8$ ablation, and that
is not bit-exact.

### What step 11 does not establish

- **The 0.6 s mode-1 regression is unexplained**, as above.
- **The arena is ~1.34 GB per layer and never shrinks.** Peak memory is now trunk 54.47 GB
  plus arena, and no memory-pressure testing was done.
- **The buffered fallback path was never exercised** in these runs, so it is untested code.
- **One prompt, prefill only**, and at $T=1$ there is far less expert overlap, so the
  arena's benefit at decode is unmeasured.

---

## Step 12 - overlapping the read with the arithmetic

After step 11 the two remaining costs were prefetch 7.04 s and $\mathbb{X}$ 6.82 s, and the
accounting showed them running strictly one after the other:

$$7.046 + 9.122 + 0.301 = 16.47\ \text{s}$$

Neither can be made faster on its own - the prefetch is at the device ceiling and
$\mathbb{X}$ is compute-bound. But they use different hardware, and nothing was overlapping
them.

### The equation licenses it

Section 4.2 sums the routed experts expert-major, and each expert's term
$\pi_j \mathbb{X}[W_2^{(j)}]\mathrm{SiTU}(\mathbb{X}[W_1^{(j)}]\zeta, \mathbb{X}[W_3^{(j)}]\zeta)$
reads **only that expert's weights**. No expert's computation depends on another's bytes,
so the order in which bytes arrive is free as long as each expert's have arrived before it
runs.

### The change

The arena work split in two. Planning - dedupe, sort, assign arena offsets, publish the
lookup table - stays where it was and does no I/O. The reads moved into `nreader` pthreads
that pull expert indices off an atomic counter and run ahead, setting a per-expert done
flag. The compute loop waits on the flag for expert $k$ and then computes it.

Bit-exactness is not at risk: the same bytes land at the same addresses, the experts are
computed in the same order, and each writes into its own rank slot. Only *when* a byte
arrives changed.

### Measured

Cold cache, pinned, three runs each:

```
                       wall (s)              mean    X (s)    stall (s)
mode 3, blocking     16.60 16.54 16.54      16.56     6.83      -
mode 4, 8 readers    12.32 12.31 12.32      12.32     8.40     1.08
mode 4, 4 readers    11.96 12.01 11.97      11.98     8.23     0.95
```

All nine runs 171,008/171,008 identical, token 17374.

### The reader count is not monotone, which is the interesting part

```
readers     2      4      6      8     12     16
wall     13.88  11.96  12.02  12.30  12.73  13.34
stall     3.05   0.92   0.82   1.08   1.49   1.98
```

Two readers cannot keep up and the compute waits 3.05 s. But past six readers **the stall
goes back up** - 1.08, 1.49, 1.98 - even though there is more read capacity. The readers
are competing with the compute threads for cores and memory bandwidth, so adding readers
slows the consumer they are feeding and themselves. Four is the optimum on this machine.

### Where the 7.04 s went

```
mode 3   wall 16.56 = ops  9.22 + prefetch 7.04 + 0.30
mode 4   wall 11.98 = ops 10.72 + stall    0.95 + 0.31
```

Of the 7.04 s of prefetch: **1.50 s reappeared as compute inflation** ($\mathbb{X}$ 6.83 to
8.23, contention with the readers), **0.95 s remained as stall**, and **4.58 s was
genuinely absorbed**. A perfect overlap would have reached
$\max(7.04, 9.22) + 0.3 \approx 9.5$ s, so 2.5 s of the ideal is still being lost to
contention and stall.

### End to end

```
                                    process total
eq.c, untouched   11 runs, 54.76 - 59.83      mean 56.40
final             18.43  18.32                mean 18.38
```

**3.07x end to end, 4.42x on the timed region, bit-identical.**

### What step 12 does not establish

- **The contention was not attacked.** 1.50 s of $\mathbb{X}$'s growth is reader
  interference; pinning the readers to specific cores, or using `io_uring` to issue reads
  without threads at all, would likely recover some of it. Neither was tried.
- **Four readers is this machine's optimum**, measured on one prompt with one thread
  count. It is not a portable constant.
- **Thread create and join happen per layer** - 93 layers times 4 threads - and the 0.31 s
  of unattributed time was not broken down.
- **One prompt, prefill only.** At $T=1$ there are fewer distinct experts per layer, so the
  pipeline has less to run ahead into.

---

## Step 13 - the 2.5 s is mostly not recoverable, and io_uring was the wrong tool

Step 12 recorded 1.50 s of compute inflation plus 0.95 s of stall as "still lost to
contention", and named `io_uring` as the fix. Before building it, the contention was
measured.

### The syscalls are not the cost

`io_uring` exists to cut syscall overhead. Comparing process CPU accounting:

```
                   wall    user    sys     cpu
mode3 blocking    22.78   151.4   25.4    775%
mode4 pipelined   18.35   184.2   28.0   1156%
```

**System time moves 25.4 to 28.0 s.** Two and a half seconds of kernel time, spread over
16 threads. There is nothing there for `io_uring` to take. The reason to suspect otherwise
was that *user* time jumped 32.8 s - which turned out to be something else.

### The spinning is real and free

That user-time jump is OpenMP workers spinning at the barrier while the main thread waits
on the pipeline. Turning it off:

```
                            wall     X (s)   stall   user
mode4 default              12.08     8.233   1.03   184.2
mode4 OMP_WAIT_POLICY=passive 12.53   8.844   0.74   133.2
mode4 GOMP_SPINCOUNT=0     12.49     8.857   0.72   133.1
```

Passive waiting removes **51 s of user CPU** and makes the wall clock **worse**, 12.08 to
12.53, because the many small parallel regions then pay wakeup latency. The spinning was
burning cores that had nothing else to do. It is waste, and it is not a cost.

### It is not core starvation either

If the readers were stealing cores from the compute, the penalty would grow with the number
of compute threads competing for them. It does not:

```
                  X mode3   X mode4   penalty
16 compute thr     6.828     8.233     1.206x
12 compute thr     8.935    10.498     1.175x
```

Near-constant, and slightly *smaller* at 16 threads. Whatever the readers cost, it is not
contention for cores.

### It is the memory controller, measured directly

```
RAM read bandwidth, idle disk              45.29 GB/s
RAM read bandwidth, O_DIRECT running       32.29 GB/s
      the concurrent disk rate             14.00 GB/s
```

**DMA writing at 14.00 GB/s removes 13.00 GB/s of read bandwidth from the CPU** - very
nearly one for one. The device controller and the cores share one memory controller, and
bytes landing in RAM are not free just because no instruction issued them.

`io_uring` would move the same 99.72 GB into the same arena at the same rate. It cannot
help, and building it would have been wasted work discovered after the fact.

### What this does to Step 12's floor

Step 12 computed a perfect-overlap floor of $\max(7.04, 9.22) + 0.3 \approx 9.5$ s and
concluded 2.5 s was being lost. **That floor was wrong**, because it assumed overlapping is
free. Overlapping costs the compute about 17% of its throughput. The real floor is

$$\max\big(7.04,\ 9.22 \times 1.17\big) + 0.3 \approx 11.1\ \text{s}$$

against 11.98 measured. **The pipeline is within about 0.9 s of what this hardware allows,
not 2.5 s.** Step 12's figure is corrected here rather than in place, since the reasoning
that produced it is the point.

### What Step 13 does not establish

- **`io_uring` was not built**, so "it cannot help" rests on the bandwidth measurement and
  the sys-time figure, not on a comparison against a working implementation.
- **The 1:1 bandwidth trade was measured with one synthetic reader** (`diskbench` at 4 MB,
  8 threads) against one synthetic consumer (`membench`), not inside the model.
- **The 17% factor is taken from $\mathbb{X}$'s own slowdown** and applied to all operators
  when computing the floor, which is approximate.
- **Nothing was changed.** This step produced no code and no speedup - it closed an avenue
  and corrected a number.

---

## Step 14 - the decode was the kernel, and two optima that move together

$\mathbb{X}$'s uncontended compute was 6.83 s at 14.6 GB/s against a 45 GB/s memory
ceiling. Step 10 recorded the likely cause as register pressure: 20 `__m256d`
accumulators against 16 architectural YMM registers.

### That note was wrong, and the assembly says so

```
Xm._omp_fn.0 : 222 instructions
   vmulpd 4   vaddpd 7   vcvtps2pd 4   vmovapd 15
   vector regs used: 12, max index 11, zmm: no
```

Four `vmulpd`, not twenty: **the compiler never unrolled the position loop**, so
`acc[t][m]` stayed an array in memory, loaded and stored each iteration. The kernel was
not short of registers; it never asked for any. There was no pressure to relieve.

### Unrolling works, and is worth almost nothing

Writing the position loop out per $T$ so the accumulators are named variables:

```
            T=1     T=2     T=3     T=5    (GB/s, isolated, bit-identical)
V0        15.68   14.30   12.58    9.01
V1        15.45   15.08   14.88   12.04
```

1.34x at $T=5$, and the unrolled $T=5$ path does use 32 registers with max index 31, so
AVX-512's extended register file is reachable once the code needs it. But the real
distribution of positions per expert is not 5:

```
m=1: 4409 (77.6%)   m=2: 957 (16.8%)   m=3: 239 (4.2%)   m=4: 70 (1.2%)   m=5: 8 (0.1%)
mean m = 1.295
```

**Step 8's batching mostly does not batch** - 78% of experts are used by exactly one
position. At $T=1$ the unrolled version is slightly *slower*. Weighted by the real
distribution the whole change is worth about **1%**, and it was not adopted.

### Where the m=1 kernel actually goes

A diagnostic variant that skips the table lookup entirely - wrong results, purely to
price the decode:

```
V0 wd[] array          15.55 GB/s
V2 set_pd              33.47 GB/s     2.15x, bit-identical
V3 byte table          35.50 GB/s     2.28x, bit-identical
VD no decode (WRONG)  120.91 GB/s     diagnostic only
```

**The decode is 87% of the kernel**, and the cost is not the table - it is the `wd[16]`
array. Sixteen doubles written to the stack and read straight back as four vectors, 128
bytes of store-to-load round trip per 16 codes. Building the vectors directly with
`set_pd`, or reading two doubles at a time from a byte-indexed table, removes the trip.

### In the real model, the saving went straight into stall

```
                    X (s)   stall (s)   wall (s)
xdec=0 wd[] array    8.16      1.00      12.01
xdec=1 set_pd        5.58      3.40      11.88
xdec=2 byte table    4.33      4.84      12.03
```

$\mathbb{X}$ nearly halved and **the wall clock did not move**. The compute had become
faster than 4 reader threads could feed it, and every second saved turned into a second of
waiting.

Taken alone this reads as "no improvement". It is not - it is a mistuned second parameter.
The reader count was optimized in Step 12 against a kernel that no longer exists.

### Re-tuning the readers finds the gain

```
readers    4      6      8     12     16     20
wall   11.98  11.13  10.84  10.54  10.56  10.71
stall   4.80   3.74   3.29   2.83   2.74   2.92
```

The optimum moves from 4 to **12**. Confirmed over three runs: 10.57, 10.54, 10.56, all
171,008/171,008 identical, token 17374.

**This is the finding worth keeping from this step.** Two optima that depend on each other:
tuning the reader count, then improving the kernel, then measuring, shows nothing. The
speedup only exists after re-tuning. A change that appears worthless can be a change whose
benefit is being absorbed somewhere else.

### End to end

```
                                    process total
eq.c, untouched  13 runs, 54.59 - 59.83      mean 56.20
final            16.87  16.90                mean 16.89
```

**3.33x end to end, 5.02x on the timed region, bit-identical.**

### What Step 14 does not establish

- **The byte table is 1 MB** (256 scale rows x 256 byte values x 16 B), of which 64 KB is
  hot for this checkpoint's 16 scale bytes. It exceeds L1 and lives in L2. `set_pd` reaches
  94% of its speed with a 32 KB table and no such concern; the byte table was chosen on
  measurement alone.
- **The unrolled variant was measured and discarded, not deleted.** At a longer prompt,
  where m would be larger, it would start to pay.
- **The reader optimum is this kernel's, on this machine.** It has now moved once and will
  move again if either side changes.
- **One prompt, prefill only.**

---

## Step 15 - the stall was a work-granularity bug, and the measurement found it

After step 14 the run was 10.56 s: 7.32 s of operators, 2.83 s of stall, reads at the
device ceiling. The stall was the only addressable item left, and rather than guess at it
the wait was split by position in the layer:

```
pipeline stall   2.784 s total
   first expert of each layer   1.563 s   56%
   next 11 experts              0.105 s
   rest                         1.116 s
```

**1.563 s over 92 layers is 17 ms per layer, spent waiting for one expert.**

### The work unit was wrong

Each reader thread took a whole expert and read its six ranges **in sequence**. So the
first expert of a layer was finished by one thread walking 17.5 MB at one thread's
bandwidth, while the other eleven readers worked on experts nobody needed yet. The
compute could not start until that one thread finished.

The fix is to make the work unit a range rather than an expert, with a per-expert counter
of ranges outstanding. Six readers then converge on the first expert at once and it
completes in roughly one range time instead of six.

### Measured

```
                    wall     stall    first expert
gran=0, 12 readers  10.54    2.833      1.560
gran=1, 12 readers   9.74    1.129      0.293      5.3x less
gran=1, 16 readers   9.65    1.124      0.296
```

The first-expert stall falls 5.3x, which is what six-ranges-in-parallel predicts, and the
total stall falls from 2.833 to 1.129 s.

### And the reader optimum moved a third time

```
readers   12     14     16     20     24     32
wall    9.74   9.65   9.67   9.69   9.71   9.81
stall   1.13   0.98   0.99   0.96   0.92   0.79
```

12 in step 12, 12 again after step 14's kernel, now **14 and flat out to 24**. Finer work
units make the reader count less critical, which is worth more than the 0.09 s: the
parameter stopped being sharp.

Three confirming runs at 14 readers: 9.66, 9.64, 9.64, all 171,008/171,008 identical,
token 17374.

### End to end

```
                                    process total
eq.c, untouched  15 runs, 54.59 - 59.83      mean 56.30
final            15.96  15.99                mean 15.98
```

**3.52x end to end, 5.49x on the timed region, bit-identical.**

### Where it stands now

```
operators        8.19 s   (X 5.52, Q 1.32, rest 1.35)
pipeline stall   1.07 s
overhead         0.39 s
                 9.65 s
```

The pure I/O is 7.04 s and the operators are 8.19 s, so **the run is compute-limited
again** - the reads now finish before the arithmetic needs them, except for 1.07 s. Note
$\mathbb{X}$ rose from 4.73 to 5.52 s as more readers run concurrently, which is step 13's
DMA bandwidth tax being paid harder; it is still a net win of 0.9 s.

### What Step 15 does not establish

- **The remaining 1.07 s of stall was not decomposed further.** 0.30 s is still the first
  expert, 0.28 s the next thirteen, 0.48 s the rest.
- **Range granularity assumes six ranges per expert.** The `r / 6` mapping is hardcoded to
  this architecture's three tensors times two kinds.
- **The optimum is flat, not proven optimal.** 14 through 24 readers differ by 0.06 s,
  which is within what this machine varies by.
- **One prompt, prefill only.**

---

## Step 16 - huge pages, and a gain that was mostly outside the timer

$\mathbb{X}$ ran at 18 GB/s in the model against 36 GB/s in step 14's isolated benchmark.
Before chasing that, the benchmark itself had to be checked: its working set was 140 MB
and this CPU has **128 MB of L3**, so it could have been measuring cache.

```
working set   23MB   70MB  140MB  351MB  877MB  2340MB
V0           14.74  14.59  15.36  15.55  15.71  15.53
V3           35.84  33.17  35.88  36.56  42.66  42.16
VD          129.51 121.31 125.65 128.21 149.08 150.68
```

Flat across a hundredfold range, with no cliff at L3. The benchmark was sound and the gap
is real.

### Nothing was asking for huge pages

```
/sys/kernel/mm/transparent_hugepage/enabled:  always [madvise] never
AnonHugePages in the running process:         0 kB
```

THP on this box is **madvise-only**, so a mapping gets huge pages only if it asks. Neither
the 1.34 GB expert arena nor the 54.47 GB trunk did. That is 13.6 million 4 KB pages
against a TLB of a few thousand entries.

Two `madvise(MADV_HUGEPAGE)` calls, one on each mapping. Verified rather than assumed:
**AnonHugePages 53,182 MB** in the running process afterwards.

### Measured, and absorbed again

```
              wall     X (s)    stall
huge=0        9.63     5.55     1.05
huge=1        9.36     4.56     1.92
```

$\mathbb{X}$ fell 18%, and **1.0 s of the 1.0 s saving went into stall**, exactly the
pattern of step 14. Re-sweeping readers this time found nothing to recover - 14 through 32
all land between 9.34 and 9.39 - because step 15's range granularity already flattened
that parameter. The timed region gained only 0.27 s.

### The gain was somewhere the timer could not see

```
                      inner wall     process total
before step 16           9.65            15.98
after                    9.35            13.36
```

**The timed region moved 0.30 s and the process total moved 2.62 s.** `eq.c` starts its
own clock after the trunk is loaded, so everything before and after is invisible to it:
faulting 54.47 GB into existence and tearing it down again costs 13.3 million page
operations with 4 KB pages and about 26 thousand with 2 MB ones.

Had I judged this change by the number the program prints about itself, it would have
looked like a 3% curiosity. It is a 16% end-to-end win. The instrument that had been
adequate for fifteen steps was measuring the wrong span for this one.

> **Corrected below.** Three faults in the paragraph above: it names the wrong file, the
> comparison it rests on is confounded, and the mechanism it gives is half wrong. The
> conclusion survives; the reasoning did not.

### Correction - the controlled measurement

**First, the naming.** The timed region described is `eqp.c`'s. `eq.c` is only the
untouched baseline built as `eq_orig`. Both place `T0` after the trunk load, but the
measurement under discussion was never `eq.c`'s.

**Second, the comparison was confounded.** 15.98 came from step 15's end-to-end run at
`K3_NREADER=14` with no huge pages; 13.36 from step 16's at `K3_NREADER=22` with them.
Two variables moved. Re-run with the reader count held fixed and only `K3_HUGE` changing:

```
nreader = 14        start->T0    timed    teardown   process total    sys
huge = 0               3.77       9.615      2.50        15.885      29.30
huge = 1               3.80       9.375      0.195       13.370      16.25
                      +0.03      -0.240     -2.305       -2.515     -13.05
```

The reader count turns out not to matter at all here: 14 against 22 differs by 0.05 s in
both configurations. So the effect is real and the figure is **2.52 s**, not 2.62.

**Third, the mechanism was half wrong.** The paragraph above says "faulting into existence
and tearing it down again". The trunk load is **unchanged** - 3.77 against 3.80 s, very
slightly *worse* with huge pages - so faulting costs nothing measurable, presumably
because it overlaps the 14.5 GB/s of I/O that provokes it. **The entire 2.31 s is
teardown**: the kernel unmapping 54.47 GB at process exit, 13.3 million page-table entries
against about 26 thousand. `sys` time falling 13.05 s is the direct evidence, and it is
evidence the original paragraph did not have.

This is work the program never asks for and cannot see. It happens after the last line of
`main`, and no timer inside the process can reach it.

### End to end

```
                                    process total
eq.c, untouched  17 runs, 54.59 - 59.83      mean 56.23
final            13.37  13.35                mean 13.36
```

**4.21x end to end, 5.67x on the timed region, bit-identical.**

### What is still unexplained

$\mathbb{X}$ now runs at 21.9 GB/s in the model against 42 GB/s isolated. Step 13's DMA
tax accounts for 0.71x, predicting about 30 GB/s. **The remaining 1.37x is not
accounted for.** Candidates not tested: the arena is written by DMA and read once with no
reuse, whereas the benchmark re-reads buffers the CPU itself wrote; and the benchmark uses
one input vector where the model uses a different $\zeta$ per layer. Neither was measured,
and the gap is recorded as open rather than explained.

### What Step 16 does not establish

- **Huge pages were granted, not proven beneficial in isolation.** The 53,182 MB figure
  confirms the mapping changed; the 18% on $\mathbb{X}$ is the whole evidence of effect.
- **The 2.62 s outside the timer was not broken down** between load-time faulting and
  exit-time teardown.
- **1.37x of $\mathbb{X}$'s gap is open**, as above.
- **One prompt, prefill only.**

---

## Step 17 - the unmap, measured directly instead of by subtraction

Step 16's correction left the teardown named but not isolated: 2.31 s had been arrived at
by subtracting two measured spans from a third. That is an inference, not an observation,
and it could not say *which* mapping cost the time.

### Measuring it instead of deducing it

Exit unmaps everything anyway, so calling `munmap` explicitly before returning moves the
same work to where a clock can see it. The trunk and the arena are timed separately:

```
            arena (1.63 GB)   trunk (54.47 GB)   total    start->T0   timed   proctotal
4 KB            0.056              2.387         2.444      3.77      9.58     15.85
THP 2 MB        0.003              0.136         0.140      3.80      9.37     13.37
```

The three spans now sum to 15.79 and 13.31 against process totals of 15.85 and 13.37, so
0.06 s is unaccounted in both - process startup and libc teardown. The subtraction in
step 16 was right, and it was the **trunk**: 2.387 s of it, 179 ns per 4 KB page.

### Separating the four phases

Getting 54.47 GB into RAM is four distinct things - reserving the address range, faulting
it (which is the kernel zeroing each page before handing it over), filling it over
O_DIRECT, and releasing it. The full run cannot separate them, so `tbench.c` does, at
16 threads, cold cache before each:

```
                 mmap      fault/zero    fill            unmap
4 KB            0.0000       2.576       3.731 (14.60)   2.323
THP 2 MB        0.0000       2.511       3.740 (14.56)   0.128
hugetlb 2 MB    0.0000       2.395       3.742 (14.56)   0.006
```

Three findings, none of which the full run could have shown:

**The fill does not care about page size at all.** 14.55 to 14.62 GB/s across every
policy, which is the device ceiling measured back in step 6. Nothing about this problem
is addressable by changing how memory is mapped.

**The kernel zeroes 54.47 GB and it is free.** Faulting the range costs 2.4 to 2.6 s when
timed on its own, but doing it first does not make the fill any faster (3.731 without,
3.735 with). It hides completely under the I/O that provokes it. Prefaulting is a pure
loss of 2.5 s - worth recording because it is exactly the kind of "optimization" that
looks obviously right.

**hugetlb unmaps 20x faster than THP.** 6 ms against 128 ms, and 350x faster than 4 KB.

### What 1 GB pages would have given, and why they are not available

55 pages instead of 27,235. The CPU advertises `pdpe1gb`, the pool exists in sysfs, and
124 GB was free. The kernel still granted **0 of 55**, before and after an explicit
`compact_memory`. Runtime allocation of 1 GB pages needs contiguity this kernel would not
assemble; it would have to come from `hugepagesz=1G hugepages=55` on the boot command
line, which is a reboot of someone else's machine and was not done. **Untested, not
rejected.**

One defect of my own here: the first attempt carried `MAP_NORESERVE` into the hugetlb
mapping, which turns an empty pool into a SIGBUS at first touch rather than a failed
`mmap`. Two core dumps before I read the flag I had copied.

### In the real program

`K3_HUGE=2` maps the trunk from the hugetlb pool and falls back to THP, loudly, if the
pool is not reserved.

```
                            load          unmap    process total
THP, pool empty         3.77 / 3.78       0.136    13.33 / 13.36
hugetlb, no pool          3.78 (fell back)  0.137    13.33
hugetlb, pool reserved  3.74 / 3.74       0.006    13.17 / 13.16
THP, pool reserved        3.78            0.132    13.34
```

The last row is the control. If simply reserving 55 GB had changed the machine's memory
behavior, THP would have moved too; it did not. So the 0.18 s is the page size.

It is 0.13 s of unmap plus 0.04 s of load - the fill also edges from 14.43 to 14.56 GB/s,
which is the one place page size did show up in I/O. Logits byte-identical to the
preserved baseline, `md5 23d162dcefb18211a7540ef12948f1eb`, token 17374.

### What it costs

Reserving the pool takes 2.541 s and releasing it 0.121 s, one time, system-wide, outside
the process. 55 GB then belongs to hugetlb and to nothing else. For this program that is
memory it was going to occupy anyway, but it is a system configuration change rather than
a property of the binary, and it should be read that way.

**The trunk unmap has gone 2.387 -> 0.136 -> 0.006 s.** As a share of the run, 15.1% ->
1.0% -> 0.05%. This one is finished.

## Step 18 - the 1.37x gap was a bad baseline, and the fix for it pays nothing

Since step 13 this document has carried an unexplained 1.37x: $\mathbb{X}$ runs at
21.9 GB/s inside the model and 42 GB/s in `xdec`, and the DMA tax was supposed to account
for 0.71 of that, predicting 30. The residue was the largest open question here.

It was not a residue. It was three differences between the two measurements, none of
which I had checked before naming the leftover a mystery.

### The two kernels are not the same code

`xdec`'s kernel keeps four accumulators in named locals. `Xm` keeps them in
`acc[NPOS][4]`, indexed by a **runtime** `t`. A runtime-indexed array cannot be promoted
to registers, and the assembly says so plainly:

```asm
vcvtps2pd (%rcx), %ymm0
vmulpd    %ymm9, %ymm0, %ymm0
vaddpd    -128(%rax), %ymm0, %ymm0    ; accumulator loaded from memory
vmovapd   %ymm0, -128(%rax)           ; and stored back
```

Every accumulation is a load-modify-store, **including when T is 1**, which is 77.6% of
calls. This is step 14's `wd[]` stack round trip again, in a different place, introduced
by step 8's batching and invisible for ten steps because nothing had compared the two
kernels as code.

### Measuring all three under identical conditions

`xgap.c` runs the benchmark kernel, `Xm` copied verbatim, and a T==1 specialization on the
same buffers, same threads, 2.34 GB working set:

```
                        no I/O, best of 5
flat (xdec style)             42.02
Xm verbatim                   35.90
T==1 specialized              43.06      identical output, 0 of 3072 floats differ
```

So the round trip is worth **1.20x** - and the model's kernel still runs at 35.9, not 21.9.

### The rate collapses with the position count

The other thing `xdec` never measured is T > 1. Per-m rates, no I/O, best of 5:

```
T = 1      2      3      4      5
35.90  25.98  21.70  18.35  15.91
```

Weighting by the model's own measured distribution (m=1 77.6%, m=2 16.8%, m=3 4.2%,
m=4 1.2%, m=5 0.1%) gives

$$\left(\sum_m \frac{p_m}{r_m}\right)^{-1} = 32.5\ \text{GB/s}$$

for the model's kernel at the model's mix, with no I/O at all.

### The DMA tax, measured on this kernel rather than borrowed

Step 13's 0.71 came from a separate memory benchmark. Running `xgap` with 14 background
O_DIRECT readers, which achieve 14.35 to 14.45 GB/s:

```
          no I/O   with I/O   factor
T = 1     35.90     20.09      0.56
T = 2     25.98     16.97      0.65
T = 4     18.35     11.67      0.64
T = 5     15.91     10.52      0.66
```

The real tax is 0.56 to 0.66, harsher than 0.71. Weighted the same way, the loaded rate is
about 18.4 to 19.0 GB/s. T=3 came back at 9.18, out of line with its neighbors, so that
run is noise and the bracket is approximate.

### The gap closes

```
model kernel, model mix, no I/O         32.5 GB/s
observed in the model                   22.5
model kernel, model mix, continuous I/O ~18.7
```

The observed rate sits inside the bracket, nearer the loaded end. The run is only doing
I/O for about 75% of its wall time (7.04 s of pure I/O in 9.33 s), and interpolating on
that fraction predicts 20.9 against 22.5 observed. **There is no unexplained residue.**
The 1.37x was an artifact of comparing against a different kernel, at one position, warm,
best of five, with no concurrent I/O - a baseline that shared nothing with the thing being
measured except its name.

### Fixing the real defect, and being paid nothing for it

The stack round trip is a genuine defect, so it is fixed: a T==1 path in `Xm` with named
accumulators. The assembly confirms it - the new path has **0 accumulator-in-memory adds**
against 13 in the general path - and the logits are byte-identical.

```
          X seconds   X GB/s   stall   wall   process total
before      4.425      22.54   2.025   9.33      13.33
after       4.303      23.17   2.154   9.33      13.32
```

$\mathbb{X}$ got 0.125 s faster and the stall got 0.125 s longer. Sweeping the reader
count, which is what rescued the same situation in step 14, does not rescue it here:

```
nreader     10     14     18     22     26     30
stall     2.275  2.171  2.111  2.004  1.966  1.914
wall      9.40   9.32   9.35   9.34   9.36   9.36
```

Stall falls monotonically across the whole range and wall does not move. The reason is
visible in the operator table:

```
             operators   stall     X
nreader=14     6.843     2.154   4.303
nreader=30     6.986     1.895   4.429
```

More readers buy less stall and pay for it in the arithmetic, at close to one for one.
**The run is at a bandwidth equilibrium**: 99.72 GB of experts and 54.47 GB of trunk have
to cross the same memory controller as every operator's operands, and which side of the
boundary the time is charged to is a bookkeeping choice, not a saving.

The specialization is kept. It is bit-exact, it makes the operator measurably faster, and
it will pay if the I/O side ever gets cheaper. Today it is worth 0.00 s end to end, and
recording that honestly matters more than the change did.

## Step 19 - the trunk is the same every run, so what can be done once?

The experts are chosen by the input. The trunk is not: the same 54.47 GB is used by every
run of every prompt. That asymmetry has been sitting in plain sight since step 2's
classification, and it raises a question the classification never asked - if it never
changes, why is it being done again?

Three different things could be meant by "do it once", and they have different answers.

### Does a run even need all of it?

`K3_COVER=1` counts reads per 4 KB page of the trunk, marked at every operator that
takes a trunk pointer.

```
touched               53.83 GB of 54.47 GB      98.82%
never read             0.64 GB
read exactly once     53.10 GB
read more than once    0.73 GB     most-read page: 10 times
total page reads      13,851,843 = 56.74 GB of traffic
```

56.74 GB of traffic against $\mathbb{Q}$'s independently counted 56.73 GB, which is the
check that the instrument works. **There is no unused trunk and essentially no reuse.**
Nothing to prune, and nothing to cache between operators - 97.6% of the bytes are touched
once and never looked at again.

### Can anything be precomputed from it?

Every tensor in the census is a $W$ appearing as $W \cdot x$, and $x$ depends on the
input. No product survives the run, so nothing of that shape can be hoisted.

The one transform that is genuinely input-independent is the dequantization: the stored
form is a 4-byte scale plus `in` int8 values per row, and turning that into floats needs
no input at all. It could be done once and stored.

It would lose, and the reason is measurable rather than arguable. $\mathbb{Q}$ moves
56.73 GB in 1.302 s = **43.58 GB/s**, against a RAM read ceiling measured on the same
machine in the same state of **45.01 to 45.57 GB/s**. $\mathbb{Q}$ is at 96% of what the
memory system can deliver, so its time is set by how many bytes it reads and by nothing
else. The stored form is about 1.001 bytes per weight. Precomputed bf16 would be 2.0 and
fp32 4.0, so the same operator would take at least twice as long.

**The quantized form is not a compression of the weights that costs time to undo. It is
the reason the operator is fast.** Precomputing the dequantization would be paying to
make the bottleneck bigger.

### What can be done once is the reading

That leaves the part that really is repeated work: 54.47 GB comes off the disk on every
single run, into memory, to be arranged exactly as it was the run before. Put the file in
tmpfs and map it, and the load stops existing.

```
                        start->T0    timed    process total   minor faults
disk -> anon RAM           3.80      9.40        13.40             93,718
tmpfs, mapped              0.02      9.26        10.54            893,539
                           0.02      9.33        10.63            893,516
                           0.02      9.34        10.61            893,535
```

Logits byte-identical. **13.40 -> 10.6 s**, and the timed region does not move, which is
the control that matters: reading the trunk through a tmpfs mapping during the arithmetic
costs nothing over having it in anonymous RAM.

### The fault-around trap

Shmem THP on this box is `[never]`, so the mapping is 4 KB pages - 13.3 million of them.
Yet the run takes 893 thousand faults, not 13.3 million, because the kernel maps 16 pages
per fault. Then:

```
                          minor faults   timed   process total    sys
tmpfs, steady state           893,539     9.26      10.54        11.9
after drop_caches          13,359,543    10.74      12.03        28.5
next run, no drop          13,359,582    10.57      11.85        25.5
```

`drop_caches` costs 1.3 s of wall and 13 s of system time by turning every fault back
into a single page, **and it stays that way for later runs**. The cold-cache discipline
that made every other measurement in this document trustworthy is the one thing that
breaks this one. A benchmark that drops caches before each run would have measured 11.9
here and concluded residency was worth 1.5 s instead of 2.8.

### What was not tested

hugetlbfs would give 2 MB pages and remove the faults entirely, but the arm failed and is
**untested, not rejected**: the pool granted 11,740 of the 27,800 pages asked for because
the tmpfs copy still held the memory, and `dd` wrote 0 bytes because hugetlbfs has no
`write()` - it has to be filled through a mapping. Given that tmpfs already matches
anonymous RAM in the timed region, there is little left for it to win.

### What it costs, stated plainly

54.47 GB of RAM held permanently, out of 124 GB, and a one-time 26.7 s copy. And the
comparison is not like for like: 13.40 s is a cold start that reads the model from disk,
10.6 s is a warm start that does not. **It is not a faster program. It is the same
program not repeating work between runs** - which is the right architecture for anything
that serves more than one request, and is exactly what the asymmetry implies.

## Step 20 - everything above was measured on five tokens

Nineteen steps rest on one prompt of five tokens, prefill only. The m-distribution that
gives mean 1.295, the bandwidth equilibrium of step 18, the 99.72 GB of expert traffic,
the reader optimum - all of it. `NPOS` was already a compile-time constant used
everywhere, so only the token list was hardcoded; `K3_IDS` makes it settable.

**Control first**: rebuilt at `NPOS=5` with the ids passed through the new path, logits
still byte-identical to the preserved baseline. The parameterization did not change the
program.

Longer prompts come from a 124-token passage tokenized with a greedy longest-match
encoder, whose first five ids reproduce `1008,10484,318,15383,387` exactly - which is the
check that it is not producing nonsense.

### The regime changes completely

```
NPOS   wall    read GB   mean m   m=1 share   stall      operators
  5    9.35     156.70    1.295     77.6%     23.25%      73.16%
 16   20.17     258.97    2.047     60.6%      9.00%      88.85%
 32   36.82     358.49    2.744     51.6%      2.98%      95.58%
 64   65.77     521.15    3.565     43.0%      1.00%      97.81%
```

**At five tokens the run is I/O bound with 23% stall. At sixty-four it is compute bound
with 1%.** Step 18's "bandwidth equilibrium", where every second taken out of an operator
reappeared as stall, is a property of the five-token prompt and not of the program. At
64 tokens there is no stall left to absorb anything, so arithmetic improvements would
convert directly into wall time.

Per token the run gets cheaper - 1.87 s/token at five, 1.03 s/token at sixty-four -
because the expert bytes are amortized over more positions.

### A metric that nearly fooled me

Measured in weight bytes, $\mathbb{Q}$ appears to collapse:

```
NPOS       5      16      32      64
Q GB/s  42.81   16.99    7.09    4.53
```

That is a 9.5x fall and I was ready to call it a defect. It is mostly an artifact.
$\mathbb{Q}$ uses each weight byte $T$ times, so arithmetic intensity rises with the
prompt and GB/s of weights measures less and less of the work. Counting
multiply-accumulates instead:

```
NPOS          5      16      32      64
Q GFLOP/s   405.9   451.7   319.6   313.1
X GFLOP/s   113.3   146.0   173.4   206.6
```

$\mathbb{X}$ **improves** with prompt length, from 113 to 207 GFLOP/s, for the same
reason its GB/s falls - it is reading each weight once and using it more.
$\mathbb{Q}$ peaks at T=16 and then loses 31%, which is a real degradation but 1.44x,
not 9.5x.

The lesson is the same one as step 18 in a different costume: **a rate is a ratio, and
when the denominator's meaning changes with the parameter being swept, the rate stops
being a measurement.**

### Where $\mathbb{Q}$'s 31% goes, as a prediction to test

`Qm` holds `__m256 v0[NPOS], v1[NPOS]` and indexes them by a runtime `t`, the same shape
that cost $\mathbb{X}$ 1.20x in step 18. At T=64 that is 4 KB of accumulators per row,
and the inner loop also walks 64 separate `Xs[t]` input streams against a 32 KB 8-way L1.

The prediction, recorded before testing: tiling the position loop into blocks of about
eight should recover most of the 31%, because the weight row is only 7 KB and stays in
L1 across the blocks, so re-reading it per block is nearly free. If instead the loss is
the accumulator array alone, tiling will recover little. Step 21 will say which.

### What these runs do and do not establish

The `NPOS=5` control is byte-identical, which is what licenses the parameterization. The
longer runs have **no reference output to check against** - the "engine emitted 17374"
line is hardcoded to the five-token prompt, so its "DIFFERENT" verdict at other lengths
means only that a different prompt gave a different answer. These runs establish cost
structure, not correctness.

Softmax attention is $O(T^2)$ and still only 0.09% of wall at 64 tokens, so it is not
yet a concern - but it is the one term that will eventually dominate, and 64 tokens is
still a short prompt.

## Step 21 - the prediction was wrong, and the discriminator said why

Step 20 predicted that $\mathbb{Q}$'s 31% loss came from `v0[NPOS]/v1[NPOS]` being
indexed by a runtime `t`, and that tiling the position loop would recover it. The first
half of that was wrong.

### One experiment that could tell the two causes apart

There were two candidates - the accumulator array, and the T separate input streams - and
tiling fixes both at once, so a tiling result could not have distinguished them. The
discriminator is to run the model's own kernel with all T pointers aimed at **one**
buffer: identical arithmetic, identical accumulator pressure, one stream instead of 64.

```
T=64, model kernel      GFLOP/s
64 distinct streams      424.7
1 shared stream          761.8
```

**1.79x from the streams alone.** It is not the accumulators. The reason is a number I
had not looked at: $x$ is float32, so one position vector is $4 \times 7168 = 28.7$ KB,
and a 32 KB L1 cannot hold even two of them, let alone sixty-four.

That also explains why tiling disappoints. Blocking positions into groups of 8 still
walks 8 x 28.7 KB of input per pass; it reduces the problem without removing it.

### The fix the diagnosis implies

If the constraint is x traffic, the answer is to use each x load more - block the
**output rows**, so one loaded `x0/x1` pair feeds OB rows instead of one. That raises
arithmetic intensity by OB and leaves the per-output summation order untouched, so it
stays bit-exact. Measured on `qgap.c`, all variants verified against the model kernel at
0 differing floats:

```
                 T=5     T=16    T=32    T=64
model           463.1   446.4   373.9   458.7
tile 1          611.6   539.3   618.5   535.8
tile 8          365.6   530.8   604.2   507.0
tile 16         417.7   602.7   668.3   514.0
rows 4 x pos 4  384.5   693.7   644.3   649.9
rows 8 x pos 1  436.9   685.5   667.5   652.4
```

No variant wins everywhere. `rows 8 x pos 1` is best or near-best from T=16 up but is
**worse than the model at T=5**, so it is applied only for $T \ge 8$ and the short-prompt
path is left physically untouched - which makes the five-token bit-exactness structural
rather than something to hope for.

One oddity recorded and not explained: `tile 2` is consistently worse than both `tile 1`
and `tile 4`, at every T.

### In the model

```
NPOS    Q seconds        Q GFLOP/s      wall           process total
  5   1.325 -> 1.301   405.9 ->  413.3   9.35 ->  9.31   13.35 -> 13.32
 16   3.811 -> 3.147   451.7 ->  547.0  20.17 -> 19.45   24.18 -> 23.46
 32  10.771 ->  5.963  319.6 ->  577.3  36.82 -> 31.93   40.83 -> 35.94
 64  21.991 -> 11.993  313.1 ->  574.1  65.77 -> 55.88   69.79 -> 59.89
```

`NPOS=5` logits byte-identical to the preserved baseline.

**$\mathbb{Q}$ is 1.83x faster at 64 tokens and the degradation is not merely recovered
but reversed**: its rate now climbs with prompt length and plateaus near 575 GFLOP/s
instead of falling to 313. End to end that is 1.17x at 64 tokens and 1.14x at 32.

And unlike step 18, **it shows**. Step 20 established why in advance: at five tokens
there is 23% stall waiting to absorb any arithmetic saving, and at sixty-four there is
1%. The same change would have been worth nothing a step earlier and worth 9.9 seconds
here. Per token, 1.03 s -> 0.87 s.

### What this cost me to learn

The wrong prediction was cheap because it was written down before the test - there was
no way to quietly re-interpret the result afterwards. What made it recoverable was
building an experiment that could **separate** the candidates rather than one that would
have improved things under either. A tiling benchmark alone would have shown 1.18x, I
would have shipped it, and the 1.83x would still be sitting there.

## Step 22 - what the reading buys, and why the same fix fails on $\mathbb{X}$

Twenty-one steps measured bytes and seconds. None measured the values. The question
worth asking of a run that moves 156 GB is what it gets for them.

### 1240 bytes per number

```
operator     output floats   weight bytes/out   flops/out
Q             45,765,600           1,239.6       11,753.0
X             71,598,080           1,392.8        6,790.7
```

The whole five-token run reads 156.45 GB and produces 117.4 million floats - 470 MB.
**About 1,240 bytes read for every number produced, a ratio of 341 to 1.** That is not
an inefficiency to be fixed; it is what a trillion-parameter mixture is, at a prompt this
short. It is also the cleanest statement of why longer prompts are cheaper per token:
the same reading produces more numbers.

### And none of those numbers are throwaway

```
magnitude     exact 0     <1e-6     <1e-3        <1      <1e3    >=1e3
Q             0.0000%   0.0043%   1.7139%   75.8771%  22.4046%  0.0000%
X             0.0000%   0.0016%   1.3587%   97.1139%   1.5257%  0.0000%
```

Not one output of either operator is exactly zero, and under 2% are below 1e-3. This
closes a line of attack the dead-neuron census had left open: there is no output
sparsity here, nothing that could be skipped or approximated away without changing the
answer. The 341:1 ratio buys dense, uniformly-scaled values.

### What is the same in every run

The trunk is constant across runs, but its outputs are not, because they depend on the
input. The part that *is* constant is decided by causality: a position can only depend on
tokens at or before it, so any two runs sharing a prefix must agree on that prefix.

That is a claim about the implementation, not just the equation, and it had never been
checked. Dumping the final per-position state for the five-token prompt and for a
sixteen-token prompt whose first five ids are the same:

```
pos 0: BYTE-IDENTICAL (7168 floats)
pos 1: BYTE-IDENTICAL
pos 2: BYTE-IDENTICAL
pos 3: BYTE-IDENTICAL
pos 4: BYTE-IDENTICAL
```

**Exactly identical, not approximately.** So the answer to "what is $\mathbb{Q}$ in every
run" is: for a shared prefix, the very same numbers, every time, to the bit. Reusing them
across runs is not an approximation with an error budget - it is free and exact. Every
serving system that caches a prefix relies on this, and here it is demonstrated rather
than assumed.

### The same fix does not work on $\mathbb{X}$

Step 21's row blocking gave $\mathbb{Q}$ 1.83x. $\mathbb{X}$ is the larger operator -
54% of wall at 64 tokens against $\mathbb{Q}$'s 21% - so it was the obvious next target.
It fails:

```
              T=1      T=2      T=4
model        35.81    26.94    12.21
rows 2 x 1   29.17    16.84     7.67
rows 4 x 1   34.81    21.38     9.58
rows 1 x 2   30.86    24.90     8.96
```

Slower at every T and every blocking, while remaining bit-exact. Two reasons, both
visible in numbers already collected:

**$\mathbb{X}$ has no x traffic problem to fix.** Row blocking helps by reusing an x load
across output rows. $\mathbb{Q}$ needed that because it applies one weight matrix to
every row and position, so x dominates. $\mathbb{X}$ reads each expert's weights **once
and never again** - 1,392.8 weight bytes per output against $\mathbb{Q}$'s 1,239.6, but
with no reuse at all. There is nothing for the blocking to amortize.

**And it cannot afford the registers.** $\mathbb{X}$ accumulates in four double lanes per
(row, position) against $\mathbb{Q}$'s two float lanes, so `rows 2 x pos 1` already needs
8 accumulator registers plus 4 weight and 4 input, exactly filling the file. Blocking
buys nothing and pays in spills.

A harness note, since the output looks alarming: the `A flat` row reports 3072 differing
floats because it is compared against the model's buffer before the model has been run.
The k=0 comparison is meaningless; step 18 checked that pairing in the correct order. All
the row-blocked variants report 0.

## Step 23 - what prefix reuse could be worth, before building it

Step 22 left prefix reuse as the largest unexploited lever. It is also the most
expensive thing to build in this document - it needs a persistent cache and a two-phase
entry point, not a tuning change. So the question to answer first is what it could
possibly be worth, because a lever that removes 10% is not worth that.

### The cost of one more position

```
T        1     2     4     8    16    24    32    48    64
wall  3.80  5.46  8.21 12.46 19.57 25.33 31.79 43.87 55.22
```

From T=16 upward this is almost exactly linear:

$$\text{wall}(T) = 7.69 + 0.743\,T$$

which reproduces T=24, 32 and 48 to within 1.2%. Below T=16 it is concave, because the
expert set is still filling up - X's weight bytes go 25.83 GB at one position to 201.85
at sixteen, but only 463.65 at sixty-four.

A fit is worth no more than its next prediction, so before using it: **predicted
wall(40) = 7.69 + 0.743 x 40 = 37.4 s. Measured 37.62 s**, an error of 0.6%.

### The ceiling, and the floor

Prefix reuse removes position-dependent work. It cannot remove the 7.69 s, which is the
per-layer work that happens whatever the prompt length - above all reading the trunk,
which $\mathbb{Q}$ does in full even for a single position (53.83 GB at T=1).

So **a cached prefix is worth at most 0.743 s per position, against a floor of 7.69 s**:

```
prompt   prefix cached   projected   against   speedup
  64          16           43.3 s     55.22     1.27x
  64          32           31.4       55.22     1.76x
  64          48           19.5       55.22     2.83x
```

Those are projections from the fitted model, not measurements of a built system, and
they should not be quoted as anything else. What they establish is that the lever is
large enough to be worth the work when the shared prefix is most of the prompt - which
is exactly the shape of a system prompt followed by a short query.

### The precondition, which had not been checked

Step 22 showed the *final* state of the prefix positions was byte-identical across two
prompts. That is not sufficient. A cache has to restore the prefix at **every layer**,
so the prefix state must be identical at every layer, or reuse would silently diverge
partway down the stack.

Dumping each layer's input for both runs and comparing all prefix positions:

```
layers dumped: 93 and 93
compared 465 (layer, prefix position) states of 7168 floats each
identical: 465    differing: 0
```

3.3 million floats, zero differences. **The prefix state is reusable at every one of the
93 layers**, not merely at the output.

### Why it stops here

The measurement and the precondition are done; the build is not, and it is a larger
piece of work than it looks. The 24 MLA layers would need `klat`, `v` and `rp` kept per
prefix position - 24,640 floats per position per layer, so 113.5 MB for a 48-token
prefix, computed from the tensor shapes. The 69 KDA layers carry a recurrent state whose
size I have not established. The snapshot machinery at `L % 12 == 0` has to be restored
too.

That is a change to what the program is rather than to how fast it runs, and it deserves
to be a deliberate decision rather than something appended to the end of a long session.
What is settled is that it is sound, that it is exact, and roughly what it is worth.

## Step 24 - prefix reuse, built

Step 23 measured the ceiling and stopped. This builds it.

### The design the measurements dictated

A position below the cache boundary is never recomputed. Its influence on the rest
reaches through two channels, and only those two have to be stored:

- **MLA layers (24)** keep per-position key/value state, so the cache holds `klat`, `v`
  and `rp` for each cached position - 24,640 floats per position per layer.
- **KDA layers (69)** carry a *recurrent* state, so what matters is its value after the
  last cached position: `St[96][128][128]` at 6.29 MB and `convbuf` at 0.44 MB per
  layer. **This part is a fixed 464 MB whatever the prefix length.**

Nothing else is needed. The cached positions' residual stream, snapshots and MoE outputs
feed only into those two channels, so they can simply not exist in a reuse run.

The code change is a window: a global `TLO` below which positions are skipped. Because
every batched call already takes *pointer arrays*, the arrays can be offset by `TLO` and
the count reduced, leaving all indexing absolute - so the layer body did not have to be
reindexed. `TLO` is 0 in a normal run, which makes the existing path structurally
untouched rather than merely tested.

### It is exact

```
NPOS=8, prefix 4        wall     process total
full run               12.44         16.44
save run               12.76         16.76     cache 474 MB
reuse run               7.19         11.18

save  == full: IDENTICAL
reuse == full: IDENTICAL
md5 92f140ea3bcba8694f4075863a6f2ade for all three
```

Saving the cache costs 0.32 s. Reuse is checked against the full run's logits, not
against a tolerance.

### What it is worth

```
prompt  cached    full      reuse    speedup   cache   logits
   16        8   19.36 s   11.11 s    1.74x    461 MB  IDENTICAL
   32       16   32.02     17.91      1.79x    479 MB  IDENTICAL
   32       24   32.04     11.23      2.85x    497 MB  IDENTICAL
   64       48   55.58     19.65      2.83x    551 MB  IDENTICAL
```

Step 23 projected 19.5 s for the last row from a fit that had never seen a reuse run.
**Measured 19.65 s** - 0.8% out, and 2.83x against a projected 2.83x. The smaller cases
beat their projections, because the fitted fixed term overstates the cost when few
positions remain.

### What it costs, and what it does not do

The cache is about 500 MB almost regardless of prefix length, because the KDA recurrent
state dominates and is fixed. For a 4-token prefix that is a poor trade; for 48 tokens
it is 551 MB to remove 36 seconds.

It does not make the model faster. It removes work that a previous run already did, and
the floor from step 23 still stands: the 7.69 s of per-layer reading is paid by every
run however much is cached. Combined with step 19's resident trunk, those are the two
things a serving system would do, and neither changes a single arithmetic result.

## Progress

| step | | status |
|---|---|---|
| 1 - 8 | the classification, from scheme to boundary | done |
| 9 - 11 | DQ, SIMD $\mathbb{X}$, O_DIRECT | done |
| 12 - 13 | pipelining, and why the remainder is bandwidth | done |
| 14 | the decode was the kernel | done |
| 15 | the stall was a work-granularity bug | done |
| 16 | huge pages, and a gain outside the timer | done |
| 17 | the unmap, measured directly | done |
| 18 | the 1.37x was a bad baseline | done |
| 19 | the trunk is read once, and need not be read at all | done |
| 20 | the five-token picture does not generalize | done |
| 21 | $\mathbb{Q}$'s input streams, not its accumulators | done |
| 22 | what the reading buys, and why $\mathbb{X}$ refuses the same fix | done |
| 23 | the cost of a position, and what prefix reuse could be worth | measured, not built |
| 24 | prefix reuse, built and byte-identical | done |

## The arc, end to end

At five tokens, cold start:

| change | wall | end to end | bit-exact |
|---|---|---|---|
| `eq.c` as it was | 53.0 s | 56.23 s | - |
| batched positions | 27.9 | - | yes |
| + DQ table | 24.70 | 32.43 | yes |
| + SIMD $\mathbb{X}$ | 22.68 | 30.76 | yes |
| + O_DIRECT arena | 16.51 | 22.88 | yes |
| + pipelined reads | 11.98 | 18.38 | yes |
| + decode without the stack trip | 10.56 | 16.89 | yes |
| + range-granular reads | 9.65 | 15.98 | yes |
| + THP | 9.35 | 13.35 | yes |
| + hugetlb trunk | **9.33** | **13.16** | yes |
| + T==1 accumulators | 9.33 | 13.16 | yes, and worth nothing |
| + row-blocked $\mathbb{Q}$ | 9.31 | 13.32 | yes, and worth nothing here |

**4.27x end to end, cold start, five tokens.** Resident trunk, warm, is 10.6 s or 5.3x.

At longer prompts the last change is the one that matters:

| NPOS | before step 21 | after | gain |
|---|---|---|---|
| 16 | 24.18 s | 23.46 | 1.03x |
| 32 | 40.83 | 35.94 | 1.14x |
| 64 | 69.79 | 59.89 | 1.17x |

## What this leaves to do

- $\mathbb{X}$ is 54% of wall at 64 tokens and has now resisted the one fix that worked
  for $\mathbb{Q}$. Its weights are read once, so the levers left on it are byte count,
  and those are not bit-exact
- the KDA recurrent state makes the cache a flat 464 MB whatever the prefix. For short
  prefixes that dominates; whether it compresses is untested
- the five-token configuration is at its bandwidth equilibrium and nothing arithmetic
  will move it; the long-prompt configuration is compute bound and has a proven lever
- untested, not rejected: 1 GB pages (need a boot-time pool), hugetlbfs for a resident
  trunk (needs filling through a mapping), and any prompt long enough to make $O(T^2)$
  attention matter
