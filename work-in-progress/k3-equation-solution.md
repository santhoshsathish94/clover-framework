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

> **Superseded.** Those 17 `eq.c` runs were taken unpinned, while `final` is pinned. The
> like-for-like baseline is 48.57 s, so this is **3.64x**, not 4.21x. See the correction
> above the arc table.

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

## Step 25 - the state of the program, read off the source

A synthesis rather than an experiment: what `eqp.c` now treats as constant, how it
handles the trunk and the experts, what it must still derive, and what is left. All of
it read from the current 1,892-line source and one instrumented run, not from memory.

### What is constant, and never recomputed

**Shapes**, compile-time: `H 96, D 128, P 12288, KC 4, VOCAB 163840, NLAY 93, LAT 3584,
I_ 3072, SI 6144, DI 33792, NEXP 896, TOPK 16, GRP 32, QN 128, QR 64, QH 192, VH 128,
KVL 512, KVW 576, KVD 256, QLORA 1536`. **Scalars**: `EPS5 1e-5, EPS6 1e-6, LAM -5.0,
B1C 4.0, B2C 25.0`.

**The dequant map.** `E2M1[16]` is fixed by the MXFP4 format and the scale is one of 256
bytes, so $E2M1[c]\cdot 2^{s-127}$ has a closed domain of 16 x 256. `dq_init()` builds
three views - `DQ` 16 KB, `DQd` 32 KB, `DQ2` 1 MB byte-indexed. **About 1.1 MB of tables
stands in for roughly $1.9\times10^{11}$ multiplies and $1.2\times10^{10}$ `exp2f`
calls.**

**The address index** (`K3EQ`): the slot table for 93 layers, the model records, and
`erec` covering all 494,592 expert tensors. `expert_rec()` is arithmetic on it -
`idx = (L-1)*896*6 + e*6 + which*2 + kind` - with no runtime search.

### The trunk

One 54.47 GB file of int8 rows, each `4-byte scale + in values`. `load_ram()` takes an
anonymous mapping, asks for huge pages, and fills it with parallel O_DIRECT `pread` in
4 MB chunks - **3.80 s at about 14.4 GB/s**. Addressing is `slot_ptr(L,s) = trunk +
slots[..].off`, pure offset arithmetic. Only the small vectors - norms, biases - are
dequantized, by `slot_vec()`, on demand and freed.

Coverage, measured: **53.83 of 54.47 GB touched (98.82%), 53.10 GB read exactly once**,
0.73 GB more than once, 56.74 GB of traffic.

### The experts

Three phases per layer. **Route** - top-16 of 896 per position, needing only `x2b`, which
exists before any expert weight is touched. **Plan** - the union of
`NPOS x 16 x 3 tensors x 2 kinds` ranges, sorted by (file, offset) and deduplicated, so
the layer's entire byte list is known before a byte is read. **Stream** - ranges are
4096-aligned into a contiguous arena, 14 reader threads `pread` O_DIRECT into it while
compute works on earlier experts, and `res_ptr()` maps a tensor to its arena address.

Each expert's weights are read **once and never reused**: 99.72 GB at five tokens,
463.65 GB at sixty-four.

### What must still be derived, every run

```
Q  int8 projection      1.312 s  13.9%   409.8 GFLOP/s
X  mxfp4 expert proj    4.237 s  44.9%   114.8 GFLOP/s
SiTU + sigma            0.636 s   6.8%
B  bf16 lm_head         0.251 s   2.7%
router dot product      0.133 s   1.4%
AR snapshot aggregate   0.063 s   0.7%
C  shortconv            0.060 s   0.6%
D  kda delta-rule       0.046 s   0.5%
alpha / beta / gate     0.021 s   0.2%
N  rmsnorm              0.015 s   0.2%
L  l2 per-head          0.008 s   0.1%
top-k selection         0.004 s   0.1%
SA softmax attention    0.001 s   0.0%
SUM                     6.787 s  72.0%   of 9.43 s wall
```

Every line depends on $x$, so none of it hoists.

### What the table shows that had been missed

**`SiTU + sigma` is 0.636 s, 6.8% of wall - the third largest operator**, ahead of the
lm_head and five times the router, across 7,825 calls of elementwise work. Twenty-four
steps went past it because attention was always on the two big matrix operators. It is
the obvious next candidate, with the caveat that at five tokens stall would absorb any
gain and it would only show from about 32 tokens up.

$\mathbb{X}$ remains 45% of wall, reads each weight once, and has refused row blocking;
the only lever left on it changes byte count and is not bit-exact.

## Step 26 - the trunk is now resident

Step 19 measured this and left it as a finding. This does it.

```
cp /root/k3trunk_i8/trunk.bin /dev/shm/trunk.bin      # 26.76 s, once
K3_TRUNKRAM=0 K3_TRUNKPATH=/dev/shm/trunk.bin         # every run after
```

```
                  start->T0    timed    process total   logits
disk, cold          3.81       9.38        13.39        IDENTICAL
                    3.80       9.38        13.37
tmpfs, resident     0.02       9.26        10.46        IDENTICAL
                    0.02       9.51        10.74
                    0.02       9.43        10.63
```

**13.38 -> 10.61 s, a 2.77 s saving, bit-identical.** The load does not get faster; it
stops happening. The timed region is unchanged, which is the control that matters - a
tmpfs mapping is as good as anonymous RAM once the arithmetic starts.

### What it costs

50 GB of the machine's 124 GB, held until `rm /dev/shm/trunk.bin` or a reboot; 71 GB
remains free. The process's own RSS is unchanged at 56.8 GB because it maps those pages,
but they are **shared** - a second process would map the same copy rather than reading
another 54.47 GB.

### The trap that comes with it

The mapping is 13.3 million 4 KB pages and shmem THP is `[never]` here, yet a run takes
only **893 thousand faults**, because the kernel maps 16 pages per fault. `drop_caches`
destroys that:

```
                  faults        timed    process total
steady state       893,537       9.26       10.46
after drop_caches  13,359,542    10.79      12.03
next run           13,359,577    10.59      11.82
```

It costs 1.4 s and **does not recover on the next run**. The cold-cache discipline that
makes every other measurement in this document honest is the one thing that spoils this
one, so the resident runs above are measured the way a serving machine would actually
run: without dropping caches.

## Step 27 - the unattributed quarter, decomposed

Step 26's table ends with `wall - ops - prefetch  2.366  25.27%  unattributed`, and I
flagged it as a quarter of the run nobody had accounted for. That was misleading, and
reading the line that computes it says why:

```c
wall - tot - pf_secs        /* tot = SUM of operators; the stall is NOT subtracted */
```

The pipeline stall sits **inside** that figure. It is printed on its own line two rows
above, so it was accounted for all along - just not deducted. Timing the two remaining
pieces that nothing measured:

```
unattributed              2.387 s   25.57% of wall
  stall                   2.013     84.3%    already reported separately
  slot_vec                0.272     11.4%    1296 calls, 606.4M floats
  progress print          0.001      0.0%
  residual                0.102      4.3%    = 1.1% of wall
```

**Only 0.102 s, 1.1% of wall, is genuinely unaccounted.** The claim of a missing quarter
was wrong.

### What the decomposition did find

`slot_vec` dequantizes trunk tensors to float32 and no operator counts it: **0.272 s
across 606.4 million floats.** Of those, 92 MoE layers x 896 x 7168 = **590.9 million,
97.4%, are the router's gate matrix**, materialized as 25.7 MB of float per layer and
freed again at the end of the layer.

Set against the operator table, that is the striking part:

```
router dot product     0.153 s      the arithmetic
slot_vec for the gate  ~0.265 s     just turning its weights into float
```

**Dequantizing the router's weights costs more than routing with them.** It was invisible
because it happens in a helper called before the timer starts, and because every
optimization pass went to the two big matrix operators.

### Why this one may be recoverable

The gate matrix is stored as int8 rows with a per-row scale, exactly like the matrices
$\mathbb{Q}$ reads. $\mathbb{Q}$ never materializes a float copy - it applies the scale
inside the dot product. The router could do the same: read 6.4 MB per layer instead of
writing and then reading 25.7 MB, and skip the allocation entirely.

It should be bit-exact, because the value fed to the accumulator would be the same
`(float)w[i] * sc` that is written to the array today, promoted to double at the same
point. That is a claim to test, not to assume - it is the next step.

## Step 28 - the router reads its own weights

Step 27 ended with a claim to test: the gate is int8 rows with a per-row scale, so the
router could apply the scale inside the dot product instead of having `slot_vec` build a
25.7 MB float copy of it every layer.

### Checking the assumption first

The claim rests entirely on the gate being I8R, and I had not looked. The index says:

```
S_GATE = 29    dtype 2 (I8R)    nbytes 6,426,112 = 896 x (4 + 7168)
               rows 896 = NEXP, cols 7168 = E
```

Exactly the layout $\mathbb{Q}$ reads. (My first probe used slot 8 and returned a
(12288, 7168) tensor - the enum has 37 slots and the gate is the thirtieth. Reading the
enum beat guessing at it.)

### The change

```c
const float gv = (float)w[i] * sc;        /* must round to float here */
a += (double)gv * (double)x2b[t][i];
```

The float temporary is deliberate: it forces the same rounding the stored array had, so
the accumulator sees the identical value. `K3_GFUSE=0` keeps the old path for comparison.

### It is exact, and it pays

```
                      materialized      fused
slot_vec                0.2715 s        0.008 s     606.4M floats -> 15.5M
router dot product      0.1425 s        0.1265 s    11.82 GB -> 2.96 GB of weights
wall                    9.245 s         9.045 s
process total          10.51 s         10.315 s
```

Both paths give `md5 23d162dcefb18211a7540ef12948f1eb`, identical to the preserved
baseline, and on the second prompt the two paths agree byte for byte.

**0.20 s, about 1.9% end to end.** Step 27 predicted "up to ~0.27 s plus part of the
router's 0.153 s"; the operators gave up 0.28 s and the wall kept 0.20 of it, the rest
lost in run-to-run variance at n=2.

`slot_vec` is now 0.008 s and 15.5 million floats - it still dequantizes the norms, the
conv weights and the biases, which is what it was for.

### The rate metric lies again

The router's GB/s falls from 83 to 23 while getting **faster**, because it now reads a
quarter of the bytes for the same arithmetic. Third time in this document that a ratio
moved the wrong way while the thing underneath improved: always read the seconds.

## Step 29 - SiTU was running on one core

At 64 tokens `SiTU + sigma` is 7.809 s, 14.12% of wall, third behind $\mathbb{X}$ and
$\mathbb{Q}$ and never examined in 28 steps.

### The obvious idea, killed by one measurement

Each element costs three scalar libm calls:

```c
float a  = (B1C * tanhf(g[i] / B1C)) * sigf(g[i]);   /* sigf = 1/(1+expf(-x)) */
float uu = B2C * tanhf(u[i] / B2C);
```

Both `tanhf` and the sigmoid reach exactly representable values well inside float
range - `tanhf(x)` is exactly `1.0f` for `x > 8.66`, so `g > 34.7` would short-circuit.
A bit-exact fast path, if the data goes there. It does not:

```
SiTU inputs: 25,605,120 elements   g [-56.012, 79.077]   u [-16.590, 27.712]
|g| > 34.7   0.0001%      g > 17   0.0009%      |u| > 216.6   0.0000%
```

Essentially nothing saturates. One run, idea dead.

### What was actually wrong

`situ` has no `#pragma omp`. Neither do `rmsnorm`, `rmsnorm_blocks`, `l2_blocks` or
`AR`; of the small operators only `Bf` is parallel. **It was running 25.6 million
elements and 77 million libm calls on a single core of a sixteen-core machine.**

It is elementwise - `y[i]` depends only on `g[i]` and `u[i]` - so splitting it changes
no arithmetic at all. Nested parallelism is off by default, so the pragma is inert at
the call sites that sit inside other parallel work.

```
                  SiTU        wall        process total
NPOS=5   serial    0.688 s     9.065        10.335
         parallel  0.0655     8.97          10.245      10.5x on the operator
NPOS=64  serial    7.809      55.30         56.61
         parallel  0.799      51.20         52.51        9.8x on the operator
```

Bit-identical to the preserved baseline at five tokens, and at sixty-four the two paths
agree byte for byte. **4.10 s end to end at 64 tokens, 7.2%.**

### Where the other 2.9 s went

$\mathbb{X}$'s operator gave up 7.01 s but the wall only kept 4.10. The stall grew
0.681 -> 2.485 s and $\mathbb{X}$ itself slowed 29.715 -> 30.806 s. Step 14's lesson
says re-sweep the reader count when a kernel changes, so:

```
nreader     8       14      20      26
wall      51.24   51.22   51.24   51.43
stall      2.821   2.479   2.363   2.310
X         30.536  30.820  30.895  31.105
```

Stall falls monotonically, $\mathbb{X}$ rises by the same amount, wall does not move.
**Nothing to reclaim - the bandwidth equilibrium has come back.** Taking 7 s of pure
compute out of a 55 s run re-exposed the I/O that the compute had been hiding, and at
64 tokens the run is no longer purely compute bound.

### What this leaves

The same mistake is still sitting in four more operators. At 64 tokens `AR` is 0.849 s,
`C shortconv` 0.812, `N rmsnorm` 0.208 and `L l2` 0.120 - **about 2 s of elementwise
work still on one core.** Whether it pays is now in doubt for the same reason the reader
sweep found nothing, but it is cheap to test.

## Step 30 - the other four, and a claim I made from one run

Step 29 left `AR`, `C shortconv`, `N rmsnorm` and `L l2` still serial, about 2 s at 64
tokens.

### These are not the same case as SiTU

SiTU was safe to split because it is purely elementwise. Three of these four contain
**reductions**, and a reordered sum is not bit-exact. Read individually:

- **`rmsnorm`** - `ss += x[i]*x[i]` is a running sum. Left serial. Only its second loop,
  `y[i] = (w[i]*x[i]) * inv`, is parallelized.
- **`rmsnorm_blocks`, `l2_blocks`** - the reduction lives *inside* each block and blocks
  are independent, so splitting by block reorders nothing.
- **`AR`** - same per source; the final accumulation was restructured to sum per output
  so each `out[i]` still adds its sources in `s` order.
- **`C shortconv`** - index `i` touches only its own conv history, so it splits cleanly.

### The result, after I got it wrong

My first measurement said this made the run **7.5 s slower** at 64 tokens, with
$\mathbb{X}$ collapsing from 30.8 to 41.9 s. I reported that. It was a single run and it
was an outlier - $\mathbb{X}$ has measured 30.84 to 31.08 s in every one of the nine runs
since. Something else was on the box.

Per operator, one run each, and then the two endpoints at n=3:

```
                 wall     X      process total
none (mask 0)   51.37  30.861      52.69
conv only (16)  50.75  31.084      52.06
AR only (8)     50.86  30.987      52.18
norms only (7)  51.13  30.944      52.44
all (31)        49.84  30.945      51.17

n=3   mask 0    52.72 / 52.69 / 52.73
      mask 31   51.28 / 51.15 / 51.13
```

**1.53 s at 64 tokens, 2.9%, and the spread is 0.03 s.** Every mask is byte-identical to
mask 0, and the default now verifies byte-identical to the preserved baseline.

Per operator: `C` 0.836 -> 0.080, `AR` 0.767 -> 0.210, `N` 0.207 -> 0.156,
`L` 0.104 -> 0.028. The operators give up 1.44 s and the wall keeps 1.53 - consistent
within noise, and unlike step 29 nothing leaks into the stall.

At five tokens it is worth 0.055 s, which is the bandwidth equilibrium again.

### What it cost to learn

Nothing about the code - the change was right the first time. What was wrong was
reporting a 7.5 s regression from n=1 when the run-to-run spread on a good day is 0.03 s
and this document has already recorded two cases of a single measurement misleading it.
The repeat took four minutes.

## Step 31 - can a token look up its own experts?

The proposal: if a token id maps to a vector, and the vector picks the experts, then a
link table from token to its 16 experts per layer would remove the routing entirely.

The reason to doubt it is in the equation - the router reads `x2b[t]`, the hidden state
at layer L, which has accumulated the whole prefix through L layers. The token is only
its seed. But how much the choice is driven by the token rather than the context is an
empirical question, so `K3_DUMPSEL` was added to dump `(layer, position, token id, its
16 experts)` and three prompts were run:

```
A  The capital of France is                    1008,10484,318,15383,387
B  France is a country and France is in Europe 93705,387,261,5141,316,15383,387,306,6715
C  The capital of Japan is                     1008,10484,318,10417,387
```

B carries token 387 at positions 1 **and** 6 - the same token, in one run, under the
same weights, with different context. A and C put 387 at the same position with a single
word of context differing.

### The answer is no, and not marginally

```
                                        shared of 16     layers all-16
control: identical prefix               16.00  (100.0%)    92 of 92
same token, same position,
  one word of context differs           10.38  ( 64.9%)     0 of 92
same token, same run, different context  3.42  ( 21.4%)     0 of 92
same token, different run and position   3.70  ( 23.1%)     0 of 92
null: two random (layer, position) picks 0.29  (  1.8%)
```

The control is the instrument check: an identical prefix gives 16 of 16 on every layer.

**The intuition is half right.** There is real token signal - 21 to 23% against a 1.8%
null is twelve times chance - and it is strongest at the start, where layer 1 shares 11
to 14 of 16, decaying to 3 to 6 by layer 6. The token does shape the choice.

But a table has to be right, and **not one layer of 92 reaches 16 of 16 in any
non-trivial case** - not even for the same token at the same position with one word
changed between "France is" and "Japan is".

### It cannot pay as a hint either

The program already knows the exact expert list **before it reads a single expert byte**:
step 5 established that routing needs only `x2b`, so phase 2 plans the layer's whole byte
list in advance. Within a layer there is nothing for a prediction to be early for.

Speculating across layers - predicting L+1 while computing L - would need roughly four
times the bytes at 23% accuracy, on a run that is bandwidth bound. Clearly negative.

The underlying reason is that **the router is already cheap and already early**: 0.127 s
of a 9-second run, for the exact answer. A table would replace an exact, cheap,
well-timed computation with an inexact one.

### A defect in my own instrument

The first run labeled every prompt with the *default* token ids and printed `0` for
position 6, because `g_ids = ids` was placed before `K3_IDS` reassigns `ids` - so it
pointed at the hardcoded five-element array and read past its end. The expert sets come
from `idsel_all` and were never affected, so the comparisons held, but the labels were
wrong until it was moved and re-run. The corrected run reports token 387 against 387
throughout and identical numbers.

## Step 32 - prefix reuse, re-verified and then actually used

Step 24 built prefix reuse and left it there. Steps 28, 29 and 30 then changed code
**inside the TLO window** it depends on - the router loop, `AR`, and `shortconv` inside
the KDA position loop where the recurrent state is saved. A feature built seven steps
ago, with the ground moved under it, and never re-tested.

### It survived

```
NPOS=8,  prefix 4    save == full: IDENTICAL    REUSE == full: IDENTICAL
                     md5 92f140ea3bcba8694f4075863a6f2ade
second prompt        REUSE == full: IDENTICAL
NPOS=64, prefix 48   full 51.44   save 51.76   reuse 19.98    IDENTICAL
```

That md5 is **the same value step 24 recorded**, through gate fusion, a parallelized
SiTU and four more parallelized operators.

**2.57x at 64/48**, down from step 24's 2.83x only because the full run has got faster
since; the reuse run barely moved, 19.65 to 19.98 s, because what it does is mostly the
fixed per-layer work the other steps did not touch.

### The case that had never been tested

Every check so far reused a cache for the **same prompt**, which is not what a cache is
for. The real use is one prefix serving different continuations. Two 18-token prompts
sharing their first ten tokens and diverging after:

```
A  ...Paris, a city on | the river Seine that has served as the
B  ...Paris, a city on | a wide plain far from any sea coast
```

```
1. A full                                  20.97 s   token 10583
2. A full, saving the 10-token prefix       21.36
3. A reusing A's cache                      11.70    IDENTICAL to A full
4. B full, no cache                         21.58    token 13
5. B reusing A's cache                      11.41    IDENTICAL to B full
control: A and B differ, so the test is meaningful
```

**A cache built from one prompt serves a different one exactly.** B gets its own answer,
token 13, byte for byte with its own full run, at 1.89x. This is the serving case, and
until now it was assumed rather than shown.

The cache is 488 MB for a 10-token prefix and 578 MB for a 48-token one - the difference
is only the MLA per-position state, since the KDA recurrent part is a fixed 464 MB.

### What it is and is not

It is not a faster program and cannot be a default: a one-shot binary has no earlier run
to reuse. It is a two-phase shape - pay once for a prefix, then answer many
continuations against it - and what these numbers establish is that the shape is sound
and exact, not approximate.

## Step 33 - "bit-exact" was never checked against a fixed build

Setting up a capture run, I compared its logits to the preserved baseline out of habit
and they did not match. The baseline is `eq_c_logits.baseline.bin`, md5
`23d162dcefb18211a7540ef12948f1eb`. Today's program gives
`4a6365b5fabc65f9fb461adf8e11f1ee`.

Every row of the arc table above says **bit-exact: yes**.

### What the difference actually was

91.1% of the floats differ - 155837 of 171008, including 6801 of the 7168 in the final
normalized state, so it starts in the body and not at the lm_head. But the **largest
absolute difference is 5.722e-06** on logits of magnitude 18, which is two to three
units in the last place. Both pick 17374.

My first guess was that steps 29 and 30 changed a summation order. **Wrong, and cheaply
disproved**: `K3_PAR2=0`, `K3_SITUPAR=0`, `K3_GFUSE=0`, all three at once, every `K3_XDEC`
setting, and `OMP_NUM_THREADS=1` all give the same `4a6365`. The parallel work is not
the cause.

### The real cause

The intermediate sources were still on the box, so the question was answerable rather
than arguable:

```
                      -O3 -march=native      -O3 -march=native -ffp-contract=off
eq.c                  81349a44...            23d162...  == BASELINE
eqp_prebase.c         81349a44...            23d162...  == BASELINE
eqp.c                 4a6365b5...            23d162...  == BASELINE
eq.c  -O3 (no native) 23d162...  == BASELINE
eq.c  -O2 (no native) 23d162...  == BASELINE
```

**Every version of the source is bit-identical to the baseline once FMA contraction is
off.** The arithmetic never changed. What changed is the compiler's *opportunity* to
fuse a multiply and an add into one instruction that rounds once instead of twice.

The baseline was produced by a binary built without `-march=native`, so no FMA
instruction existed to fuse into. That is why `-O3` and `-O2` without it still reproduce
it exactly.

The site is step 30's restructuring of `AR`:

```c
/* before */                            /* after */
for (i) out[i] = 0.0f;                  for (i) {
for (s) for (i)                             float o = 0.0f;
    out[i] = out[i] + pi * v[i];            for (s) o = o + pis[s]*srcs[s][i];
                                            out[i] = o;
                                        }
```

The order of summation over `s` is identical. What moved is the accumulator - from a
float array element that has to be written to memory each step, to a local the compiler
can keep in a register and contract. Reverting **only** that block and rebuilding with
contraction still on gives `81349a44...`, exactly the pre-step-30 value, which settles
the attribution rather than inferring it.

### The defect is mine, and it is methodological

The program was right the whole time. The *check* was wrong: it compared against a
binary built with different flags and never recorded them. It passed for thirty steps
because nothing had yet given the compiler a new place to fuse, and the first
restructuring that did was read as a regression.

A bit-exactness claim is meaningless without the build that produced the reference.
Everything from here is built `-O3 -march=native -ffp-contract=off`, which reproduces
the baseline exactly and, at five tokens, **costs nothing measurable** - 8.81 / 8.80 s
against 8.93 / 8.81 s, inside the noise, because this configuration is bandwidth-bound
and fusing arithmetic buys nothing. That is not yet checked at 64 tokens, where the
program is genuinely FLOP-bound.

## Step 34 - thirty-four prompts, capturing identity instead of values

Everything so far rested on one prompt. The question was what more prompts would show,
and the honest way to capture a run is every value it computes - but $\mathbb{X}$ alone
emits 916M floats, 3.7 GB a run, and the box has 23 GB free.

So the capture records **what each value was computed from, not what it was**: for every
trunk weight the layer and slot, for every expert block the layer, expert, part and byte
range. Identity is also the thing that varies between prompts; the weights do not change.
34 runs came to **129 MB**.

`K3_PROV` hooks `slot_ptr` and `slot_vec` themselves, so it records the pointers actually
taken rather than a guess at which a layer needs. Capture is verified not to disturb
anything: prov on and prov off give the same logits, and run 1 reproduces the baseline.

Prompts were tokenized with the model's own tokenizer, 5 to 24 tokens, across fact,
code, arithmetic, prose, deliberate repetition, shared prefixes and three languages.

### It answers, and not just for the prompt it was built on

```
The capital of France is          -> 17374 ' Paris'
La capitale de la France est      -> 17374 ' Paris'
Die Hauptstadt von Frankreich ist -> 17374 ' Paris'
The chemical symbol for gold is   -> 70135 ' Au'
The largest planet ...            -> 75591 ' Jupiter'
The author of Pride and Prejudice -> 33197 ' Jane'
In 1969 the first humans landed   -> 28396 ' moon'
Seven multiplied by eight equals fifty -> 101055 '-six'
... and Berlin is the capital of  -> 16458 ' Germany'
... and Rome is the capital of    -> 19509 ' Italy'
```

**Stated carefully: these are the equation's own outputs, decoded with the model's
tokenizer and judged by me. The engine was not run on these 34 prompts**, so this is
evidence the implementation is semantically sound, not a cross-check against the model.

### The trunk side is provably input-independent, now measured

**All 34 runs have the identical trunk signature** - the same 2455 (layer, slot, use)
resolutions, 1251 used in place and 1204 dequantized. One signature, 34 prompts.

The slot counts confirm section 1.4's layer sets exactly: the MLA slots appear 24 times,
the KDA slots 69, the MoE slots 92, the dense MLP slots once at layer 0, and the
per-layer norms 93. That classification was argued from the text in steps 2 to 8. It is
now observed.

### Routing is not reusable, in three independent ways

| what was asked | answer |
|---|---|
| (layer, expert) pairs any prompt touched | 65072 of 82432, **78.9%**, from 34 short prompts |
| pairs used by *every* prompt | **994**, 1.2% |
| pairs used by exactly one prompt | 16685 |
| mean pairwise Jaccard between prompts | **0.194** |

The most similar pair is 0.818 - the two prompts sharing a seven-token prefix. The least
similar are around 0.13, code against prose.

**The same token never picks the same experts twice.** In the deliberately repetitive
prompts, comparing each position's 16-expert set against the first occurrence of the
same token in the same layer:

```
yes yes yes ... (12)   identical expert set: 0 of 1012
the the the ... (12)   identical expert set: 0 of 1012
one two three   (11)   identical expert set: 0 of  920
a a a a b b b b (24)   identical expert set: 0 of 2116
```

Not 5%, not 1% - zero, in 5060 comparisons. This is a far stronger form of step 31's
negative: routing depends on the position's whole context, never on the token alone.

**A shared prefix routes identically, everywhere.** For the two prompt pairs built to
diverge after a common prefix, every shared (layer, position) selects the same 16
experts with the same weights: **644 of 644**, and **828 of 828**. That is the
routing-level confirmation of what prefix reuse depends on.

### Two ideas killed by the data

**Dropping low-weighted experts.** The top-16 is flat, not peaked - rank 0 carries a mean
of 16.4% of the routed mass and the top four together only 43.4%. The tail is not
negligible: rank 15 averages 0.0368, and only 1.89% of sites have it below 0.01.
Dropping the bottom one of sixteen to save 6.2% of reads loses 3.7% of the routed mass
on average and 29% at the worst site. There is no free tail here.

**A cross-prompt expert cache.** Over 34 runs, 296158 expert reads totalling 5196.8 GB,
covering 1141.8 GB of distinct blocks - a 4.55x ceiling. But LRU gets **0.0% up to 96 GB
and 9.5% at 124 GB**, because a single run's working set is ~150 GB and evicts
everything before it can be reused. The whole 1142 GB is needed to reach the 78%. On a
124 GB box, a cross-prompt expert cache is worth nothing.

### What did scale

Distinct experts per run grows about **195 per additional token** while draws grow by
1472, so the distinct-to-draw ratio falls from 0.77 at five tokens to 0.27 at
twenty-four. Repetition reduces the distinct count sharply - the 24-token `a a a a b b
b b` touches 9386 where 23-token code touches 13686 - so repeated text does share
experts in aggregate even though it never repeats a full selection.

## Step 35 - the cross-check step 34 was missing

Step 34 said plainly that the engine had not been run on those 34 prompts, so "it answers
correctly" was my reading of decoded output rather than a comparison. That is the gap
this closes.

The engine has a directly comparable channel: `--ids` takes the same token ids, `--gen 1`
stops after one token, and that path calls `argmax_` on the logits with no sampling, so
it is greedy by construction. `--greedy` is a `--chat`-only flag and is deliberately not
passed - passing it aborts the run.

### All thirty-four agree

```
AGREE 34 / 34  (100.0%)   DISAGREE 0
```

Every prompt, identical token id:

| | engine == equation | | engine == equation |
|---|---|---|---|
| The capital of France is | 17374 ' Paris' | yes yes yes ... | 15024 ' yes' |
| The chemical symbol for gold is | 70135 ' Au' | the the the ... | 276 ' the' |
| The largest planet ... | 75591 ' Jupiter' | one two three ... | 3499 ' three' |
| The author of Pride and Prejudice | 33197 ' Jane' | a a a a b b b b ... | 261 ' a' |
| In 1969 the first humans landed | 28396 ' moon' | ... and Berlin is the capital of | 16458 ' Germany' |
| The mitochondria ... powerhouse of | 5362 ' cell' | ... and Rome is the capital of | 19509 ' Italy' |
| Seven multiplied by eight equals fifty | 101055 '-six' | La capitale de la France est | 17374 ' Paris' |
| def fibonacci(n): ... | 326 ' n' | Die Hauptstadt von Frankreich ist | 17374 ' Paris' |
| SELECT name, COUNT(*) ... | 1530 ' name' | El idioma oficial de Mexico es | 1236 ' el' |

This is the claim step 34 could not make. The equation reproduces the released engine's
output on fact, code, arithmetic, prose, deliberate repetition, shared prefixes and three
languages - none of which it was built against. Only the first of the 34 was ever used
during development.

It also closes the loop on the older validation: previously the equation had been checked
against the engine on exactly two prompts.

### On the times, stated carefully

```
engine   total 1344.2 s   mean 39.54 s
equation total  455.9 s   mean 13.41 s   ratio of means 2.95x
```

**That 2.95x is not a controlled comparison and should not be quoted as one.** Three
things differ:

- the equation holds the whole trunk resident in `/dev/shm`, while the engine re-reads
  54.47 GB from disk on every one of the 34 runs, roughly 10.6 s each. A long-lived
  server would pay that once, not 34 times
- the equation runs had `K3_PROV` capture on. That was measured as negligible at five
  tokens (8.72 against 8.75 s) but it is not zero and was not controlled at every length
- the engine is a general program holding a KV cache and a 30 GB expert cache so it can
  keep generating; `eqp.c` prefills once and exits

The figure is consistent with the 3.10x "trunk already resident" row recorded earlier,
and it is **not** the like-for-like 1.73x from the controlled single-prompt comparison,
where both programs held comparable memory. Both numbers are real; they answer different
questions.

## Step 36 - where the time goes by source, and what the values show

Two questions. Where does the program actually spend itself - trunk, experts, or
elsewhere? And did the capture record enough to see the equation's *variables*, not just
which weights produced them?

### Time, attributed to where the bytes came from

Read off the call sites: every `Q`/`Qm` takes a `slot_ptr` weight and the router reads
`S_GATE`, so those are **trunk**; every `Xm` takes `res_ptr`, so that is an **expert
block**, and the pipeline stall is time waiting for exactly those reads; `B` and the
embedding read the **model file**. Everything else reads no weights at all.

Summed over all 34 prompts, 455.9 s of wall clock, from the per-operator tables the runs
already produced:

| source | time | share | bytes | rate |
|---|---|---|---|---|
| **experts** - X plus the stall, from NVMe | 363.3 s | **79.7%** | 5196.8 GB | 14.30 GB/s |
| **trunk** - Q, router, norms, conv, AR, resident in `/dev/shm` | 74.4 s | **16.3%** | 2375.7 GB | 31.93 GB/s |
| tables - lm_head | 1.8 s | 0.4% | 79.9 GB | 43.73 GB/s |
| **neither - pure arithmetic** | 8.7 s | **1.9%** | - | - |
| residual and copies | 7.4 s | 1.6% | - | - |
| unaccounted | 0.3 s | **0.1%** | | |

Two things worth stating. The experts are only **2.2x the bytes** of the trunk but
**4.9x the time**, entirely because trunk bytes come from RAM at 31.9 GB/s while every
expert byte crosses the NVMe at 14.3 GB/s. And arithmetic that reads no weights is
**1.9% of the whole campaign**. This is a data-movement program with some arithmetic
attached, not the other way round.

### The capture recorded identity, not values - so this adds values

`K3_PROV` deliberately recorded *which* weights each value came from. `K3_VALUE` already
existed but gives only a six-bucket magnitude histogram for Q and X, summed over all 93
layers, which cannot show a variable's behavior through the network.

`K3_LSTAT` adds per-layer, per-position statistics - min, max, mean, rms, count of
zeros, negatives and non-finites - for the seven named variables of a layer and the KDA
recurrent state. 245 KB a run, 22 MB for all 34.

Cost and correctness, measured rather than assumed - my first reading said 60.72 s and
was an artifact of a mangled command running two things at once:

```
off        8.99 s   == baseline      lstat_on   9.40 s   == baseline
off_after  8.83 s   == baseline
```

About 0.4 s, roughly 5%, and **all four runs reproduce the preserved baseline exactly**.

### What the values say

**Nothing is near a numerical edge.** Across 34 prompts and every layer: **zero**
non-finite values, and the largest magnitude anywhere is **7712** against float32's
3.4e38 - a headroom of 4.4e34. Whatever else is true, this computation is nowhere near
overflow.

**Nothing is sparse.** Of 277,315,584 values recorded per variable, the number exactly
zero is **6** in the attention output, **4** in the MoE output, **1** in the residual.
Step 22 found no output sparsity in $\mathbb{X}$; it holds for every variable.

**The MoE does more of the work than attention.** The MoE output is the larger of the
two in 2219 layer-prompts against 943, median rms ratio 0.791 - consistent with
$\mathbb{X}$ dominating the cost.

**The residual grows about 1.43x per layer, 100x end to end** - and then there are eight
layers where it does not.

```
layers with L % 12 == 0  (8) : mean resid_out/resid_in 0.1081   median 0.0637
all other layers        (85) : mean                    1.4285   median 1.3787
```

Seven of the eight lowest ratios in all 93 layers are exactly those layers, and the one
intruder, L13, sits immediately after snapshot layer L12.

`L % 12 == 0` is the snapshot cadence - an **E** fact from section 1.4. Reading the
source rather than guessing at the mechanism:

```c
/* (4) residual: replace at a snapshot layer, add otherwise */
if (have_prefix) resid[t][i] = resid[t][i] + aout[t][i];
else             memcpy(resid[t], aout[t], ...);
```

At a snapshot layer the residual is saved into `snap` and then **restarted from the
attention output** instead of accumulating. **Stated honestly: this is already specified,
with that comment, in `eq.c` from the first version.** The value capture recovered a
documented rule from the numbers alone. That is a validation of the instrument, not a
discovery about the model, and it should not be dressed up as one.

**The value profile is structural; the routing is not.** Across all 34 prompts the
residual rms at a given layer varies by only **1.3x to 2.0x**, at every depth sampled.
Set that against step 34: the *routing* never repeats a selection even for the same
token, yet the *magnitudes* are nearly a fixed property of the layer. The model varies
enormously in which experts it consults and hardly at all in how large the result is.

### The honest verdict on "does this show something we are missing"

**On performance, no.** The value data closed doors rather than opening them. The
enormous float32 headroom is not exploitable, because the cost is reading MXFP4 and int8
*weights*, and activations are already a negligible share of the traffic. The absence of
sparsity removes the skip-work idea for every variable, not just $\mathbb{X}$. Nothing
in the values points at work that could be avoided.

What it did establish is that the program is numerically healthy everywhere, that the
equation's structural rules are being executed as written, and - with the attribution
above - that 79.7% of the time is one thing: expert bytes crossing a disk.

## Step 37 - per-invocation capture, and the one weight that was never batched

Step 36 answered with per-layer aggregates. That was the wrong granularity: it averages
away the thing worth seeing. $\mathbb{Q}$ is called about 1171 times in a five-token run
and $\mathbb{X}$ about 17049 - **per layer** hides 1078 and 16956 of them.

`K3_STAGE` records one row per *invocation*: which trunk slot or which expert part, the
shapes, and min/max/mean/rms/zeros of what that call produced. `Q` takes a bare weight
pointer, so the slot name is recovered by matching the address back against the layer's
slot table rather than by labeling call sites. Two prompts, 1.4 MB and 2.1 MB.

### What 1171 invocations look like

```
slot    calls    in     out     rms min    rms mean   rms max   |max|
G          93   7168   12288    0.39356     1.3163     3.4855   49.557
O          93  12288    7168   0.0023959   0.087488    0.4849   19.802
EDOWN EUP SH1 SH3 SH2   92 each     every MoE layer
Q K V B FA FB           69 each     every KDA layer
QA QB KA KB             24 each     every MLA layer
MGATE MUP MDOWN          5 each     <-- five tokens.  THIRTEEN on a 13-token prompt
```

Every count is fixed by the architecture except the last row, which scales with the
prompt. The stage sequence says why:

```
layer 0:  Q K V B FA FB G O  MGATE MUP MDOWN  MGATE MUP MDOWN  ...  x NPOS
layer 1:  Q K V B FA FB G O  EDOWN EUP SH1 SH3 SH2                  once
layer 3:  QA QB KA KB G O    EDOWN EUP SH1 SH3 SH2                  once
```

**Layer 0's dense MLP was the only weight in the network re-read once per position.**
Every other weight goes through the batched `Qm` and is read once whatever $T$ is; this
one called the single-position `Q` inside a `for (t)` loop. Those three weights are
727 MB.

It also explains a number measured seventeen steps earlier and never questioned:
$\mathbb{Q}$'s weight bytes were 56.73 GB at five tokens and 99.62 GB at sixty-four. The
difference is 42.9 GB over 59 extra positions - **0.727 GB per position, exactly this
loop**. The entire growth of trunk reading with prompt length was one un-batched MLP.

### Batching it

```
              wall    Q sec  Q calls  Q weight GB  Q B/out   logits
old, off     42.59    7.199   1348       99.62      170.1    49d914...
new, off     41.94    6.452   1159       53.83       91.9    49d914...  == old
old, on      42.21    7.277   1348       99.62      170.1    dc63bb...
new, on      41.42    6.437   1159       53.83       91.9    dc63bb...  == old
```

**$\mathbb{Q}$'s weight traffic falls 99.62 to 53.83 GB, a 46% cut**, and its call count
stops depending on the prompt at all: 1159 at five tokens and at sixty-four. Bit-exact
under both contraction settings, and the five-token runs still reproduce the baseline.

**The time is small: 0.65 to 0.79 s, about 1.7%.** My first measurement said 4.31 s and
9.2%. That was n=1 and an outlier - it recorded the old binary at 46.80 s where the
controlled runs put it at 42.59 and 42.21. Exactly the error of step 30, caught the same
way, by measuring again before writing it down. The bytes are the real result; the
seconds are modest because at 64 tokens the run is expert-bound and the trunk has
headroom.

### A loose end from step 33, now closed

Step 33 pinned the build to `-ffp-contract=off` and said the cost was measured only at
five tokens. At 64 tokens, from the table above: **42.59 against 42.21, and 41.94
against 41.42 - about 0.4 to 0.5 s, roughly 1.2%.** Real but small, and worth it for a
reference that reproduces exactly.

## Step 38 - if the weights never change, why not precompute $\mathbb{Q}$?

Asked directly, and it is the right question to ask of a document whose whole premise is
finding work that does not depend on the input. The answer needs evidence, not the
classification restated at it.

### $\mathbb{Q}$'s weights are W. $\mathbb{Q}$'s output is R.

$\mathbb{Q}$ computes $W \cdot x$. $W$ is fixed by the checkpoint - which is why, since
step 37, it is read exactly once per run. $x$ is the run's activations. Comparing mean
output rms per stage between two prompts:

```
slot     prompt A     prompt B        slot     prompt A     prompt B
Q         2.36128      2.24256        K         2.81212      2.57641
V         2.05119      1.74323        G         1.31632      1.28676
...  identical stages: 0 of 20
```

Precomputing $W \cdot x$ is precomputing the answer.

### Except in one place, and that one is real

At layer 0 there is no cross-position mixing before the attention projections:
$nsnap = 0$, so $hb = resid = \mathrm{embed}[id]$ and $x1 = \mathrm{rmsnorm}(hb)$ depends
on the **token id alone**. So layer 0's pre-attention projections are a pure function of
the token, and would be precomputable per vocabulary entry.

That is testable rather than arguable. Every token id appearing at two or more
(prompt, position) places across the 34 prompts, comparing layer 0's values exactly:

```
comparisons: 186   IDENTICAL: 186   differ: 0
```

43 distinct repeated tokens, no exceptions. The structural claim holds.

### And it is worth nothing

The qualifying stages are layer 0's Q, K, V, G, B, FA and FB:

```
weight read per run     355.6 MB    = 0.65% of the trunk's 54.47 GB
the time that is worth  ~0.011 s
cache for the vocabulary  61,664 floats x 163,840 tokens x 4 B = 40.4 GB
```

**40 GB of RAM to save 11 milliseconds**, on a box that already gives 54 GB to the
trunk. Nothing past layer 0's attention qualifies at all, because from there positions
mix.

There is a second, independent reason. At 64 tokens $\mathbb{Q}$ runs at **1067
GFLOP/s** and only 8.34 GB/s - it is arithmetic-bound, not byte-bound. Making its
weights cheaper to fetch would not move it.

This is step 8's precompute test applied again with better instruments. The one thing
that ever passed was $\mathrm{DQ}$, and it passed because its input domain is finite -
4096 entries. A token-keyed layer 0 cache has 163,840 entries at 240 KB each. **The test
is not "is the weight constant" but "is the input domain small enough that the answer
can be enumerated".** $\mathbb{Q}$'s input domain is every activation the model can
produce.

## Step 39 - write the function out instead of storing the data?

The natural follow-up to step 38. If layer 0 is a pure function of the token, express the
*function* rather than tabulate its outputs.

### For a linear map over a finite domain, the function is the data

$\mathrm{rms}(\mathrm{embed}[id])$ is a scalar constant for a given token, so layer 0's
projection collapses to something genuinely linear:

$$Q_{out}[id] \;=\; \frac{1}{c_{id}}\,\underbrace{W_Q \cdot \mathrm{diag}(w) \cdot E^{\top}}_{M}\,e_{id}
            \;=\; \frac{M[:,\,id]}{c_{id}}$$

The closed form **is** a matrix. Written out for all seven qualifying stages it is
$61{,}664 \times 163{,}840$ entries - **40.4 GB**, the identical figure step 38 gave for
the table, because for a linear operator over a finite domain the symbolic form and the
lookup table are the same object.

$M$ can be kept smaller only by keeping it **factored**, and its minimal factorization is
$W_Q$ (88 MB int8) times the embedding table (2.35 GB bf16) - which is precisely what the
program already stores, and evaluating that factorization on demand is precisely what it
already does. The compressed form of the function was the original program.

### So how much information is in the weights at all?

That is the question the framing actually opens, and it had never been measured. Scanning
the whole int8 trunk and a sample of the MXFP4 expert blocks:

```
int8 trunk payload, 54.38 GB across 1437 slots
  exactly zero          1.5932%          distinct values used   255 of 256
  first-order entropy   7.0070 bits of 8
  entropy-optimal code  47.63 GB vs 54.38 GB  = 87.6%
  most common values    0:1.59%  1:1.40%  -1:1.40%  2:1.38%  -2:1.38%  -3:1.36%

MXFP4 expert payload, 2011 blocks sampled
  exactly zero nibble   5.7535%          distinct nibbles       16 of 16
  first-order entropy   3.7544 bits of 4  = 93.9%
  nibbles  5.75 10.97 9.55 7.66 7.73 5.29 2.52 0.51 | 5.75 10.97 9.55 7.66 7.73 5.29 2.52 0.51
```

**The weights are already at their information-theoretic floor.** int8 carries 7.01 of its
8 bits and MXFP4 3.75 of its 4. The distribution is close to uniform - the single most
common int8 value occurs 1.59% of the time where uniform would be 0.39%, and 255 of 256
values are in use.

The nibble histogram is exactly mirror-symmetric between its halves. That is the sign bit:
a perfectly balanced and wholly incompressible bit, leaving 2.75 of 3 bits in the
magnitude. The format is well matched to what it stores.

### What that rules out

A lossless re-encoding could take at best **12.4% off the trunk** and **6.1% off the
experts**, and only by paying entropy decoding on every weight read - on a program where
$\mathbb{Q}$ is already arithmetic-bound at 64 tokens (1067 GFLOP/s, 8.34 GB/s) and where
five-token runs sit within 1.25x of a memory-bandwidth floor. Spending CPU to save a
tenth of the bytes is the wrong direction on both.

Together with step 38 this closes the "do less work by knowing the weights are fixed"
family: the weights are read exactly once, their outputs depend on the input, their
closed form is the same size as their table, and the bytes themselves are nearly
incompressible. What remains is not representation, it is the 79.7% of wall clock that is
expert bytes crossing a disk.

## Step 40 - a second un-batched read, and why it was free

Step 37 found layer 0's MLP re-reading its weights once per position. The obvious
question is whether anything else does the same. The router does:

```
 5 tokens   router  460 calls = 92 layers x  5 positions    2.96 GB
64 tokens   router 5888 calls = 92 layers x 64 positions   37.84 GB
            37.84 / 5888 = 6.43 MB = exactly one gate matrix
```

`for (t) { parallel for (e) { read gate row e; dot with x2b[t] } }` - the whole gate is
re-read for every position. Inverting the nesting so each row is read once and applied
to all positions makes the call count **92**, one per MoE layer, at five tokens and at
sixty-four alike. Bit-exact: the five-token runs still equal the preserved baseline, and
old and new are byte-identical at 64 tokens.

### And it is worth nothing

Interleaved A/B, n=4 each so drift hits both equally:

```
old mean   wall 41.89   SUMops 38.20   router 1.605 s   router 37.84 GB
new mean   wall 41.86   SUMops 38.30   router 1.559 s   router  0.59 GB
```

**37.25 GB of counted traffic removed, 64x less, for 0.03 s** - individual runs span
41.84 to 41.95, so the difference is inside the noise.

The reason matters more than the change. **The gate is 6.43 MB per layer and this box has
128 MB of L3.** Re-reading it sixty-four times was re-reading it from cache. The byte
counter faithfully counted 37.84 GB that never crossed the memory bus.

That is exactly why step 37's fix did pay and this one does not: layer 0's MLP is
**727 MB**, far beyond any cache, so those re-reads were real DRAM traffic.

### What this corrects

Step 36 attributed 2375.7 GB of trunk reading across the campaign and derived a trunk
rate of 31.93 GB/s from it. **Both figures are inflated**: they count the router's
per-position re-reads, which were cache hits. The 31.93 GB/s was never a DRAM rate - it
was bytes-requested over time, and requests served by L3 made the apparent rate look
better than the memory system actually is. The *shape* of step 36's conclusion survives
untouched - experts are still the overwhelming majority of the time - but the trunk
column should be read as bytes requested, not bytes fetched.

The change is kept. It is strictly less work, it makes the accounting mean what it says,
and on a machine with less L3 than this one it would matter. It is simply not a speedup
here, and recording it as one would have been wrong.

### A discipline note

The first A/B on this said 43.86 against 42.20 - a 1.7 s win. That was n=1, and the same
binary measures 41.84 to 41.95 across four runs. This is the third time in this document
that a single run produced an exciting number that evaporated (steps 30 and 37 were the
others). **At 64 tokens on this box the run-to-run spread is about 0.1 s within a session
and up to 2 s across sessions. Nothing below that is a result.**

## Step 41 - what the 6.43 MB actually is, and a correction to step 33

Step 40 established how often the gate is read without ever saying what it is. Asked
directly: is it a matrix computation already run on numbers and stored, so the run does
not have to recompute it?

### It is a trained parameter

From the checkpoint's own tensor list for a MoE layer:

```
block_sparse_moe.gate.weight                 12,845,056 B   [896, 7168]   bf16
block_sparse_moe.gate.e_score_correction_bias     3,584 B   [896]
```

and from section 1 of the equation, $G_L,\ \gamma_L$ = `block_sparse_moe.gate`,
`e_score_correction_bias`, dtype I8R / F32, role "router".

So the 6.43 MB is $G_L$: **896 rows, one per expert, of 7168 model dimensions** - each
row that expert's signature vector, whose dot product with the position's activation is
that expert's affinity score. It is bf16 in the checkpoint at 12.85 MB and stored in the
trunk as int8 with a per-row scale, $896 \times (4 + 7168) = 6{,}426{,}112$ bytes.

**Nothing on this side computed it.** The only things this program computes once and
stores are $\mathrm{DQ}$, the 4096-entry dequantization table from step 9, and the int8
trunk itself - the bf16 checkpoint quantized offline when the trunk was packed, which is
exactly what halves 12.85 MB to 6.43 MB and 1267.74 MB of layer-1 bf16 tensors to about
half that.

### The router's slowness is the specification, not a defect

Having established the gate cannot be avoided, the fair question is how well it is
consumed. Badly, on the face of it:

```
router   92 x 896 x 7168 x 64 x 2 = 75.6 GFLOP in 1.558 s = 48.5 GFLOP/s   (0.38 GB/s)
Q                                                         = 1067 GFLOP/s
```

A factor of 22, and nowhere near bandwidth-bound. The assembly says why - the router's
inner loop is pure scalar double with a dependent accumulation chain:

```asm
vcvtss2sd  (%rbx,%rdx), %xmm3, %xmm0    ; float -> double
vcvtss2sd  (%rcx,%rdx), %xmm3, %xmm1    ; float -> double
vmulsd     %xmm1, %xmm0, %xmm0
vaddsd     %xmm0, %xmm2, %xmm2          ; on the critical path
```

Zero packed-double instructions, where $\mathbb{Q}$ uses AVX2 with sixteen accumulators.
Worth about 1.5 s of a 42 s run if it could be vectorized.

**It cannot.** Section 4.2 of the equation is explicit:

> Route. **Not through $\mathbb{Q}$** - the int8 gate is widened per row and the dot
> product accumulates in **double, sequentially**.

The scalar loop is correct *by specification*. Splitting that reduction into lanes
computes a different function, exactly as section 6 records for $\mathbb{Q}$, where
collapsing sixteen lanes into eight "is arithmetically different and reproduces 2,012 of
12,288 values". The 22x gap is the price of the definition, not an oversight, and the
1.5 s is not available bit-exactly.

### A correction to step 33

Step 33 spent a bisect across every surviving intermediate source to conclude that the
build must be pinned to `-ffp-contract=off`. Section 6 of the equation document already
said so:

> the verified build carries `-ffp-contract=off`, which compiles the accumulate to
> `vmulsd` + `vaddsd` rather than `vfmadd213sd`

That sentence was added after step 10 of this document, long before step 33. **The
requirement was written down and I had not read it.** What step 33 genuinely added was
the attribution - that step 30's `AR` restructure created a *new* contraction site, which
is why a build that had been passing suddenly diverged. The diagnosis was new; the
conclusion was already on the page.

The rule that failed here is the first one: read the source document before measuring
against it. A bisect is not a substitute for the specification.

## Step 42 - can a projection be computed from only what is needed?

Put simply: rather than storing a full matrix and reading all of it, compute only what
the equation actually uses. The question is fair and it has a precise answer.

### The three stores are different objects

First, because the premise usually hides here. The embedding, the trunk and the experts
are not versions of one another:

```
embedding    [163840, 7168] bf16     2.35 GB    token id -> vector, a true lookup table
lm_head      [163840, 7168] bf16     2.35 GB    same shape, DIFFERENT tensor
trunk        2455 slots, 93 layers  54.47 GB    int8 + per-row scales
   [12288,7168] 26.44   [7168,12288] 8.19   [6144,7168] 8.11   [7168,6144] 4.05
   [ 7168,3584]  2.37   [3584, 7168] 2.36   [ 896,7168] 0.59   [18432,1536] 0.68
experts      82,432 pairs = 92 x 896  1446.46 GB   MXFP4
   gate/up  payload [3072,1792] + scales [3072,112]
   down     payload [3584,1536] + scales [3584, 96]
```

They are independent trained tensors. What makes them look related is that 7168 keeps
appearing - that is the residual width, the model's bus, so anything reading or writing
the residual stream has it on one side. **A shared interface dimension, not shared
content.** Inside the MoE block the widths differ: the residual is projected down to a
3584 latent, the experts work at 3072, and the result is projected back up.

Nothing here is precomputed by this program. All three are trained parameters; the only
work done ahead of time is quantization, done once by the packer, and the 4096-entry
$\mathrm{DQ}$ table of step 9.

### The selection already happens, and it is the whole design

```
trunk     54.47 GB     100% read, every run, whatever the prompt
experts   1446 GB      a five-token run reads 99.72 GB = 6.9% of it
```

Only 16 of 896 experts are consulted per layer per position, which is exactly why the
run touches 6.9% of the expert weights. **Computing only what is needed is already what
the program does** - at the level where selection exists.

The trunk has no such level. Every projection applies to every position, so all of it is
needed. That is why it is read entirely and why steps 37 and 40 could only make it
read-once, never read-less.

### Within a matrix that is selected, all of it is needed

For a dense $y = Wx$, each $y_i$ is a dot product against row $i$, so producing all
outputs requires all weights. There are four escapes, and all four are now closed by
measurement rather than argument:

| escape | measured | verdict |
|---|---|---|
| some outputs unused | **0 fully-zero rows of 801,898,496** | no |
| some inputs unused | 162,022 fully-zero columns of 844,103,680 = **0.019%** | no |
| sparse | **11.58%** of 2.72 trillion weights are zero | 88.5% dense, no |
| structured / low-entropy | 3.75 of 4 bits, near-uniform (step 39) | no |

That scan covered all 247,296 expert tensors, 1361 GB read. **Not one output row of one
expert matrix is dead.** Every row has something in it.

Sparse formats need indices, which cost more than the 4-bit values they would replace,
and win below roughly 20% density. At 88.5% they lose badly.

### A correction to step 39

Step 39 reported 5.75% of expert weights exactly zero, read off the nibble histogram.
That is wrong: **E2M1 codes 0 and 8 are both zero**, plus and minus zero, which is why
that histogram was exactly mirror-symmetric. The true zero fraction is **11.5%**, and
the independent full scan above says 11.58%. The entropy figure of 3.7544 bits is
unaffected - it is the entropy of the 4-bit symbol, which is the right quantity for a
re-encoding question - but the sparsity statement was understated by half.

### So the answer

Yes for the experts, and it is already done: 6.9% of 1.45 TB is read because the router
selects 16 of 896. No for everything else. Within any matrix the program does read,
every row contributes to an output that is used, the values are 88.5% dense and nearly
incompressible, and a dense matrix-vector product has no way to produce its outputs
without touching its inputs.

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
| 25 | the state of the program, read off the source | done |
| 26 | the trunk is now resident | done |
| 27 | the unattributed quarter, decomposed | done |
| 28 | the router reads its own weights | done |
| 29 | SiTU was running on one core | done |
| 30 | the other four small operators | done |
| 31 | can a token look up its own experts? | no, measured |
| 32 | prefix reuse, re-verified and actually used | done |
| 33 | "bit-exact" was never checked against a fixed build | corrected |
| 34 | 34 prompts, capturing identity instead of values | done |
| 35 | the engine agrees on all 34, not just the one | done |
| 36 | time by byte source, and the variables themselves | done |
| 37 | per-invocation capture; layer 0's MLP was never batched | done |
| 38 | why $\mathbb{Q}$ cannot be precomputed | measured, rejected |
| 39 | the function is the data, and the data is incompressible | measured, rejected |
| 40 | the router re-read the gate per position; free, because it fits in L3 | done, no gain |
| 41 | the gate is a trained weight; the router's cost is the specification | answered |
| 42 | the three stores, and why a selected matrix is read whole | answered |

## The comparison that matters: the equation against the engine

Everything below the arc compares `eqp.c` against `eq.c`. **Both of those are the
equation.** `eq.c` is the first implementation of it and `eqp.c` is the optimized one, so
that arc measures how much better the implementation got - it never compared the equation
against the model at all.

The model is the released Kimi K3 engine, `kimi-k3-in-c`, the thing that emitted 17374 in
the first place. Its runs were on the box the whole time, in `/root/k3flow`.

Two of them use exactly this prompt:

```
             ids   step 0  token   cache hit   expert GB
v1.log         5   32.10 s  17374     98.5%       99.72
v2.log         5   30.39 s  17374     98.5%       99.72
```

Same `/root/k3trunk_i8`, `forward.T = 5`, embed `n = 35840 = 5 x 7168`, and **99.72 GB of
expert reads - the same figure `eqp.c` reports to two decimal places.** The two programs
move the same bytes for the same prompt and produce the same token, which is what makes
the times comparable at all.

```
                                        engine        equation
prefill step                        32.10 / 30.39 s     13.16 s
  plus its pre-step loading            +1.64 s          included
end to end, both loads included        ~32.9 s          13.16 s      2.50x
with the trunk already resident        ~32.9 s          10.6 s       3.10x
expert bytes read                      99.72 GB        99.72 GB
emitted token                           17374           17374        logits identical
```

> **Corrected, twice over.** The `v1`/`v2` figures above were recorded on 24 September
> under machine conditions that no longer hold, and the comparison also gave the
> equation pinned threads while leaving the engine unpinned - the same confound
> corrected for `eq.c` on the same day.
>
> Re-run back to back, cold cache, **both pinned**, engine at `--cache-gb 30` so the two
> hold comparable memory:
>
> ```
>                       engine        equation
> time                  23.20 s        13.39 s     1.73x
> CPU (user+sys)       190.0 s        161.0 s      1.18x less
> peak RAM              51.09 GB       58.20 GB    1.14x MORE
> disk read            154.2 GB       156.7 GB     the same
> token                 17374          17374
> ```
>
> **The equation is 1.73x faster and uses MORE memory, not less.** It holds the whole
> trunk in RAM; the engine streams it through a two-slot ring. The earlier "1.5x less
> RAM" compared against the engine's auto-sized 64 GB expert cache, which it had only
> taken because RAM happened to be free.
>
> The engine today is genuinely faster than its own 24 September logs (19.07 s step 0
> pinned, against 32.10 s then) and its code has not been optimized since: the diff is
> **126 insertions, 0 deletions, all trace taps**. The I/O time is identical then and
> now (~14 s); the entire difference is compute, which points at what else the box was
> doing on 24 September. **Those older figures are not comparable and are not used.**

### Validated on a prompt the equation had never seen

Every correctness check until now used the one prompt `eqp.c` was built against. Running
a genuinely new one - "The chemical symbol for gold is", 6 ids - through both:

```
engine   token 70135  ' Au'
equation token 70135  ' Au'
```

Engine 29.12 s / 268 s CPU / 85.1 GB against equation 13.94 s / 171 s CPU / 58.4 GB.
The equation's own "DIFFERENT" line there is only its hardcoded 17374 label, which is
meaningless off the original prompt.

**So the equation runs this prompt about 2.5x faster than the engine, and 3.1x with the
trunk resident** - not the 3.69x the arc suggests, because the arc's baseline was never
the model.

### What the engine is doing that the equation is not

This is not a like-for-like race, and the difference favors the engine's design rather
than its clock:

- The engine maintains a **30 GB expert cache and a KV cache** so it can keep generating.
  v2 goes on to produce eight tokens at 11.2, 10.4, 7.4 s each. `eqp.c` prefills once and
  exits; it cannot continue at all.
- The engine streams the trunk through a pinned set at 9,951 MB/s. `eqp.c` reads it
  O_DIRECT into RAM at 14,400 MB/s and keeps it.
- The engine is a released, general program. `eqp.c` is one prompt length, prefill only,
  with the token ids compiled in until step 20.
- n=2 for the engine, on an earlier machine state, not a controlled repeat set.

What the comparison does establish is that the engine's prefill is **45.4% I/O by its own
accounting** (trunk 5.5 s + experts 9.1 s of 32.1 s), and that most of the 2.5x came from
attacking exactly that.

## The arc, end to end

> **Baseline corrected.** The `eq.c` row below was long recorded as 53.0 s wall and
> 56.23 s end to end. Rebuilding the untouched `eq.c` and running it today shows that
> figure is an **unpinned** measurement, while every optimized row is pinned. Same
> binary, cold cache each run:
>
> ```
> eq.c pinned    (n=3)   wall 46.50 / 45.81 / 45.24   end to end 49.22 / 48.53 / 47.95
> eq.c unpinned  (n=2)   wall 53.41 / 52.69           end to end 56.01 / 55.32
> ```
>
> The recorded 53.0 / 56.23 sits inside the unpinned pair. Its logits are byte-identical
> to the preserved baseline, so the program was right; the *comparison* was not.
> **The correct like-for-like baseline is 45.85 s wall and 48.57 s end to end**, and the
> headline falls from 4.27x to **3.69x**. Memory recorded "always pin threads, every
> measurement before step 8 was unpinned" - and then the baseline that predates step 8
> was never re-measured.

At five tokens, cold start, all rows pinned:

| change | wall | end to end | bit-exact |
|---|---|---|---|
| `eq.c` as it was | 45.85 s | 48.57 s | - |
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

**3.69x end to end, cold start, five tokens.** Resident trunk, warm, is 10.6 s or 4.58x.

The intermediate rows between `eq.c` and step 8 were also taken unpinned and are not
re-measured; they are kept as a record of the order things happened, not as a like-for-
like series. Only the first and last rows are directly comparable.

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
