# The equation as `clover-k3.c` computes it

`k3-model-equation.md` states the model as one composed expression, transcribed from
`k3-stages.md` and the verified walker. This file states the same model as it is **actually
evaluated by `clover-k3.c`**, read out of the source rather than out of a prior document,
and then records where the two disagree.

The two are not the same document and should not be merged. One is the model as derived;
this one is the model as executed by the artifact that carries the gate
`23d162dcefb18211a7540ef12948f1eb`. Where they differ, the difference is evidence about the
document, not about the model.

Section 8 lists every disagreement found, each with the measurement that established it.

---

## 1. Constants, as compiled

From the `#define` block:

```
E     7168     H      96     D      128    P     12288   KC     4
VOCAB 163840   NLAY   93     LAT    3584   I_    3072    SI     6144
DI    33792    NEXP   896    TOPK   16     GRP   32
QN    128      QR     64     QH     192    VH    128
KVL   512      KVW    576    KVD    256    QLORA 1536
NPOS  5                                    (compile-time, -DNPOS=5)
```

Scalars, and note the types — each is written as a float literal widened to double, not as a
double literal:

```c
static const double EPS5 = (double)1.0e-5f;
static const double EPS6 = (double)1.0e-6f;
static const float  LAM  = -5.0f;
static const float  B1C  = 4.0f;
static const float  B2C  = 25.0f;
```

`q_sc = 1.0f/sqrtf(128.0f)` and `m_sc = 1.0f/sqrtf(192.0f)` are formed at their use sites.

**`routed_scale` does not appear anywhere in the program.** See §8.2.

### Layer type predicates

These are computed inline at the top of the layer loop, not read from a table:

```c
const int isMLA = ((L % 4) == 3 && L <= 91) || (L == 92);
const int isMoE = (L >= 1);
/* snapshot push */  if (L % 12 == 0) { ... }
```

giving MLA on $\{3,7,\dots,91\}\cup\{92\}$, MoE on $\{1..92\}$, and a snapshot push on
$\{0,12,24,36,48,60,72,84\}$ — eight pushes, so nine sources at the tail.

---

## 2. The operators

### $\mathbb{Q}$ — int8 projection

$$\big(\mathbb{Q}[W,s]\,x\big)_o \;=\; s_o\Big[\big((A_0{+}A_4)+(A_2{+}A_6)\big) + \big((A_1{+}A_5)+(A_3{+}A_7)\big)\Big], \qquad A_j = B_j + B_{j+8}$$

with $B_0\dots B_{15}$ sixteen float32 lanes, $B_c$ accumulating $i \equiv c \pmod{16}$ by
single-rounded FMA.

The AVX2 path does not look like sixteen lanes and is sixteen lanes. Two `__m256`
accumulators `v0`, `v1` hold $B_{0..7}$ and $B_{8..15}$; `_mm256_add_ps(v0,v1)` **is** the
pairing $A_j = B_j + B_{j+8}$; the `castps256_ps128 + extractf128` fold gives
$[A_0{+}A_4,\ A_1{+}A_5,\ A_2{+}A_6,\ A_3{+}A_7]$; `movehl` then `shuffle/add_ss` produce
exactly the bracketing above. The scalar `#else` branch writes the same thing literally.

The row layout is `4 + in` bytes — a float32 scale followed by the int8 row — and the scale
multiplies **after** the reduction. A scalar tail `for (; i < in; i++)` exists and no shape
in this model reaches it.

### $\mathbb{X}$ — MXFP4 projection

$$\big(\mathbb{X}[W,s]\,x\big)_r = \mathrm{fl}_{32}\big[(Q_0{+}Q_2)+(Q_1{+}Q_3)\big], \qquad Q_k = (v_{0k}{+}v_{2k})+(v_{1k}{+}v_{3k})$$

$v$ is `double v[4][4]`; lane $v[m][k]$ takes inputs $i \equiv 4m+k \pmod{16}$.

Two properties are load-bearing and both are visible in the code:

- The weight is **rounded to float32 before the double accumulation**. `dq_init` builds
  `DQ[s][c] = E2M1[c] * e8` in float, and `DQd`/`DQ2` are exact widenings of that float.
- The accumulation is `v[m][k] += (double)wd[...] * (double)x[...]` — a **separate multiply
  and add**, which is why the build carries `-ffp-contract=off`.

`E2M1` is written as the literal table, preserving its negative zeros.

The dequantisation is tabulated three ways selected by `K3_XDEC`: `DQ` float (0), `DQd`
double (1), `DQ2` byte-pair (2, default). All three are the same arithmetic; only the lookup
differs. `DQ2` is the fastest — see §8.3 for the measurement.

```c
float e8 = (s == 255) ? 0.0f : exp2f((float)s - 127.0f);
```

The `s == 255` case is the E8M0 NaN encoding mapped to zero. It is not in
`k3-model-equation.md`, and it never fires — see §8.3.

### $\mathbb{B}$ — BF16 projection

$$\big(\mathbb{B}[W]\,x\big)_r = \mathrm{fl}_{32}\big[(U_0{+}U_1)+(U_2{+}U_3)\big], \qquad U_c = (C_c + C_{c+4}) + (C_{c+8} + C_{c+12})$$

$C_0\dots C_{15}$ double lanes, lane $c$ taking $i \equiv c \pmod{16}$; the BF16 value is
widened by a 16-bit left shift, which is exact. Used only by the lm_head.

### $\mathcal{N}$ — RMSNorm

$$\mathcal{N}_n(x;w,\epsilon)_i = (w_i x_i)\cdot \mathrm{fl}_{32}\!\Big(1/\sqrt{\tfrac{1}{n}\textstyle\sum_j x_j^2 + \epsilon}\Big)$$

The sum of squares is **double and serial** — the code says so, and keeps it out of the
`omp parallel for` that covers only the scaling loop. The reciprocal square root rounds to
float32 once; the product is float32 left to right.

`rmsnorm_blocks(y,x,w,nb,blk)` applies this per block with one shared `blk`-wide weight and
$\epsilon_5$; each block owns its own reduction, so it parallelises without reordering.

### $\mathcal{L}$ — per-head L2

$$\mathcal{L}(v)_i = v_i \cdot \mathrm{fl}_{32}\!\Big(1/\sqrt{\textstyle\sum_j v_j^2 + \epsilon_6}\Big)$$

No $1/n$, no learned weight, $\epsilon_6$ not $\epsilon_5$. Applied to $q$ and $k$ only.

### $\sigma$, SiTU

$$\sigma(x) = 1/(1+\exp(-x)) \qquad \mathrm{SiTU}(g,u) = \big[(4\tanh(g/4))\cdot\sigma(g)\big]\cdot\big[25\tanh(u/25)\big]$$

`sigf(g[i])` takes the **uncapped** $g$.

### $\mathcal{C}$ — ShortConv with fused SiLU

$$a = \Big(\big((w_{c,3}x_t) + w_{c,0}b_0\big) + w_{c,1}b_1\Big) + w_{c,2}b_2, \qquad \mathcal{C}(x)_{c,t} = a\,\sigma(a)$$

Current input first, then history oldest to newest. The history shifts in the same loop.

### $\mathrm{AR}$ — snapshot aggregation

$$\mathrm{AR}(V;f)_i = \sum_{s} \pi_s\, v_s[i] \quad\text{(float32, } s \text{ ascending)}, \qquad \pi_s = \mathrm{fl}_{32}\big(e_s / z\big)$$

with $e_s = \exp(\mathrm{sc}_s - \max \mathrm{sc})$ and $z$ accumulated in double. Per source,
the sum of squares is double, `inv` is float32, the scored product `v[i]*inv` is formed in
**float32 and then widened** to double for the dot with $f$.

$\pi_s$ is a **double division rounded once**: `pis[s] = (float)((double)ex[s] / z);`. Compare
§8.1 — the MoE router does not do this.

The output loop is written $i$-outer, $s$-inner, which keeps each `out[i]` accumulating over
$s$ ascending in float32. That is the same arithmetic as a source-major sum.

### $\Delta$ — KDA delta rule

$$S \leftarrow \operatorname{diag}(\alpha)S, \qquad u = S^\top k, \qquad S \leftarrow S + \beta\,k(v-u)^\top, \qquad o = S^\top(q\,q_{sc})$$

All float32, sequential in $i$, no FMA, output read from the **updated** state. The code
fuses this into two passes over `St[h]` instead of four; the $i$ and $j$ orders are
unchanged, so every accumulation happens in the same sequence.

### $\mathrm{SA}$ — softmax attention

$$\mathrm{score}_s = \mathrm{fl}_{32}\big(\langle q, [k^{\text{lat}}_s; k^{\text{rope}}_s]\rangle_{64}\big)\cdot m_{sc}, \qquad o = \sum_{s\le t} \mathrm{fl}_{32}(e_s/z)\, v_s$$

The dot product is double and sequential over the full 192 — 128 latent then 64 rope — and
$m_{sc} = 1/\sqrt{192}$ is over that full width.

### Batched forms

`Qm`, `Xm` and the batched `situ` read a weight once and apply it to every active position.
Each output element remains an independent reduction over $i$ in the original order, so they
are bit-identical to the per-position calls by construction, and the end-to-end gate confirms
it.

---

## 3. The attention block

### 3.1 KDA, $L \notin \mathbb{M}$

$$
\begin{aligned}
q,k,v &= \mathcal{C}\big(\mathbb{Q}[W^q_L]x_1\big),\ \mathcal{C}\big(\mathbb{Q}[W^k_L]x_1\big),\ \mathcal{C}\big(\mathbb{Q}[W^v_L]x_1\big) \\
q,k &\leftarrow \mathcal{L}(q),\ \mathcal{L}(k) \quad\text{per 128-wide head; } v \text{ untouched} \\
\beta_h &= \sigma\big((\mathbb{Q}[W^b_L]x_1)_h\big) \\
z &= \mathbb{Q}[W^{fb}_L]\big(\mathbb{Q}[W^{fa}_L]x_1\big) \quad E \to 128 \to 12288 \\
\alpha_{hD+d} &= \exp\!\Big(\lambda\,\sigma\big(\exp(A^{\log}_L[h])\cdot(z_{hD+d} + \tau_{L,hD+d})\big)\Big) \\
o &= \Delta(q\,q_{sc}, k, v, \alpha, \beta) \\
\mathrm{attn.out} &= \mathbb{Q}[W^o_L]\Big(\mathcal{N}^{\text{blk}}_{12288,128}(o; w^o_L)\ \odot\ \sigma\big(\mathbb{Q}[W^g_L]x_1\big)\Big)
\end{aligned}
$$

`A_log` is loaded at width `D` = 128 and only the first `H` = 96 entries are read:
`for (int h = 0; h < H; h++) ah[h] = expf(alog[h]);`. `dt_bias` is per channel, width `P`.

**Norm first, then gate.**

### 3.2 MLA, $L \in \mathbb{M}$

$$
\begin{aligned}
q &= \mathbb{Q}[W^{qb}_L]\ \mathcal{N}_{1536}\big(\mathbb{Q}[W^{qa}_L]x_1;\ w^{qan}_L\big) \\
c &= \mathbb{Q}[W^{ka}_L]x_1 \in \mathbb{R}^{576}, \qquad c \leftarrow \big[\mathcal{N}_{512}(c_{0:512}; w^{kan}_L)\,;\ c_{512:576}\big] \\
\mathrm{kv} &= \mathbb{Q}[W^{kb}_L]\,c_{0:512}, \qquad k^{\text{lat}}_h = \mathrm{kv}_{h,0:128}, \quad v_h = \mathrm{kv}_{h,128:256} \\
k^{\text{rope}} &= c_{512:576} \quad\text{shared by all 96 heads, not rotated, still scored} \\
\mathrm{attn.out} &= \mathbb{Q}[W^o_L]\Big(\mathrm{SA}(\cdot)\ \odot\ \sigma\big(\mathbb{Q}[W^g_L]x_1\big)\Big)
\end{aligned}
$$

The norm on $c$ covers the latent 512 only; the 64 rope components are `memcpy`'d through
unnormalised. **Gate before the output projection, with no norm** — the opposite order from
KDA.

---

## 4. The MLP block

### 4.1 Dense, layer 0

$$\mathrm{ffn.out} = \mathbb{Q}[W^{\text{down}}_0]\ \mathrm{SiTU}\big(\mathbb{Q}[W^{\text{gate}}_0]x_2,\ \mathbb{Q}[W^{\text{up}}_0]x_2\big)$$

### 4.2 MoE, layers 1 to 92

**Route.** The gate is read in its stored int8 form. Each row is widened as
`tmp[i] = (float)w[i] * sc` — **rounded to float32 per element** — and then dotted in double,
sequentially:

$$s_e = \sigma\Big(\mathrm{fl}_{32}\big(\langle G_{L,e}, x_2\rangle_{64}\big)\Big)$$

`K3_GFUSE=0` materialises a float copy of the gate instead; same arithmetic.

**Select.** A single insertion pass, ascending $e$, inserting only on a strict `>`:

$$\mathcal{J} = \operatorname*{top-}k_e\big(s_e + \gamma_{L,e}\big)$$

This yields descending value with the lower index winning ties, which is what a repeated
linear maximum produces. The bias participates **only in selection**.

**Weight.** The kept weight is the unbiased $s_e$, normalised by a **reciprocal formed once
and then multiplied**:

$$\boxed{\ \pi_j \;=\; s_{\mathcal{J}_j}\cdot \mathrm{fl}_{32}\!\Big(\frac{1}{\sum_{j'} s_{\mathcal{J}_{j'}} + 10^{-20}}\Big)\ }$$

```c
double ssum = 0.0;
for (int j = 0; j < TOPK; j++) ssum += (double)wts_all[t][j];
const float iv = (float)(1.0 / (ssum + 1e-20));
for (int j = 0; j < TOPK; j++) wts_all[t][j] = wts_all[t][j] * iv;
```

This is **not** $\mathrm{fl}_{32}(s_j / \sum s)$. See §8.1 for the measurement that
distinguishes them.

**Compute.**

$$\zeta = \mathbb{Q}[W^{\downarrow}_L]x_2 \in \mathbb{R}^{3584}$$

$$\mathrm{accL} = \sum_{j=0}^{15} \pi_j\ \mathbb{X}\big[W^{(\mathcal{J}_j)}_2\big]\ \mathrm{SiTU}\Big(\mathbb{X}\big[W^{(\mathcal{J}_j)}_1\big]\zeta,\ \mathbb{X}\big[W^{(\mathcal{J}_j)}_3\big]\zeta\Big)$$

accumulated float32 with $j$ as the outer loop — expert-major.

The evaluation order is not $j$ order. The code deduplicates first: `seen[]` collects the
distinct experts across all positions, position-outer and rank-inner, and each distinct
expert is decoded **once** against however many $(t,j)$ slots requested it. On the 5-token
prompt this collapses 7,360 draws to 5,683 distinct decodes, mean multiplicity 1.295. The
per-expert results land in `slot[t][j]`, and the accumulation loop above then runs in $j$
order, so the arithmetic is unchanged.

**Combine.** The norm is applied to the aggregate once, and the shared expert reads the
full-width $x_2$ and is added unweighted:

$$\mathrm{ffn.out} = \mathbb{Q}[W^{\uparrow}_L]\,\mathcal{N}_{3584}\big(\mathrm{accL}; w^{\ell}_L\big)\;+\;\mathbb{Q}[W^s_2]\,\mathrm{SiTU}\big(\mathbb{Q}[W^s_1]x_2,\ \mathbb{Q}[W^s_3]x_2\big)$$

---

## 5. The whole model

Per layer, with $f^A_L = n^A_L \odot p^A_L$ and $f^M_L = n^M_L \odot p^M_L$ folded in float32
at the top of the loop:

$$
\begin{aligned}
h &= \begin{cases}\mathrm{AR}(\mathcal{S}\cup\{r\};\,f^A_L) & \texttt{nsnap} > 0\\ r & \text{otherwise}\end{cases} && \text{(1) guarded}\\[4pt]
\mathcal{S} &\leftarrow \mathcal{S}\cup\{r\},\quad \texttt{have\_prefix} \leftarrow 0 && \text{(2) if } L \bmod 12 = 0\\[4pt]
x_1 &= \mathcal{N}_{7168}(h;\,w^{\text{in}}_L) && \text{(3)}\\[4pt]
r &\leftarrow \begin{cases} r + \mathrm{ATTN}_L(x_1) & \texttt{have\_prefix}\\ \mathrm{ATTN}_L(x_1) & \text{otherwise}\end{cases},\quad \texttt{have\_prefix}\leftarrow 1 && \text{(4) add or replace}\\[4pt]
x_2 &= \mathcal{N}_{7168}\big(\mathrm{AR}(\mathcal{S}\cup\{r\};\,f^M_L);\,w^{\text{post}}_L\big) && \text{(5) unguarded}\\[4pt]
r &\leftarrow r + \mathrm{MLP}_L(x_2) && \text{(6) unconditional}
\end{aligned}
$$

Step (4) is not a residual stream at a snapshot layer: the incoming $r$ is discarded from the
stream and survives only inside $\mathcal{S}$.

### The tail

$$\hat{t} = \operatorname*{arg\,max}_{0 \le w < 163840}\Big(\mathbb{B}[W_{\text{lm}}]\ \mathcal{N}_{7168}\big(\mathrm{AR}(\mathcal{S}\cup\{r\};\,f^O)_{p=T-1};\,w^F\big)\Big)_w$$

with $|\mathcal{S}| = 8$, so nine sources. The argmax scans ascending with a strict `>`, so
ties go to the lowest index.

The code evaluates the tail aggregation at $p = T-1$ **only**, not at every position. See
§8.4.

### The gate file

`K3_LOGITS` writes the 7,168-wide final norm followed by the 163,840 logits — 171,008
float32 values, 684,032 bytes. Its md5 is the gate:

```
23d162dcefb18211a7540ef12948f1eb      emitted token 17374
```

---

## 6. What the program adds that is not arithmetic

None of this changes a value, and all of it is gated by the md5 above: the O_DIRECT expert
arena and its two halves (`K3_ARENA2`), the reader pool (`K3_NREADER`), the cross-layer
lookahead (`K3_NX`), the routing cache (`K3_ROUTELOAD`, which re-checks every row against the
live router and dies on disagreement), the software cache preload (`K3_PFCACHE`, default off),
the dequantisation variants (`K3_XDEC`), the gate-fusion path (`K3_GFUSE`), huge pages
(`K3_HUGE`), and the prefix cache (`pfx_mode`).

---

## 7. Where this was read from

Every statement above is read from `clover-k3.c` at the revision whose build produces
`23d162dcefb18211a7540ef12948f1eb`. Anchors: `Q` at the int8 projection, `X`/`Xm` at the
MXFP4 projection, `Bf`, `rmsnorm`, `rmsnorm_blocks`, `l2_blocks`, `situ`, `AR`, the KDA and
MLA branches of the layer loop, the MoE block, and the tail.

---

## 8. Disagreements with `k3-model-equation.md`

Everything else in that document matches this one: all ten operators including association
order and accumulator widths, the $\mathbb{Q}$ sixteen-lane tree, $\mathbb{X}$'s unfused
multiply-add and float32 pre-rounding, $\mathcal{L}$'s missing $1/n$ and $\epsilon_6$, SiTU's
uncapped sigmoid, the conv tap order, AR's float32 source-major sum, the delta rule's updated
-state read, MLA's 192-wide scale and unrotated-but-scored rope slot, the gate/norm order
inversion between KDA and MLA, `A_log` at width 128 with 96 used, the layer type sets, the
add-or-replace residual, and the nine-source tail.

### 8.1 $\pi_j$ is a reciprocal-multiply, not a division — **defect**

`k3-model-equation.md` §4.2 writes

$$\pi_j = \mathrm{fl}_{32}\!\left(\frac{s_{\mathcal{J}_j}}{\sum_{j'} s_{\mathcal{J}_{j'}} + 10^{-20}}\right)\cdot\rho$$

The program forms one float32 reciprocal and multiplies by it. The two round differently.
Both forms were built into one binary and selected at runtime:

```
reciprocal then multiply   md5 23d162dcefb18211a7540ef12948f1eb    <- the gate
divide once                md5 a82dfcd542c37d2c7c6aa4ab0df787b9

157,370 of 171,008 values differ   92.02%
max absolute delta                 6.199e-06
argmax unchanged, token 17374 under both
```

The emitted token survives, so this is invisible to anything that only checks the answer. It
is not invisible to the gate.

By that document's own standard — *"Association order is part of each definition, not a
formatting choice"* — this is a defect of the same kind as the three already recorded in its
§6.

**It also sits against its verification claim.** Its §6 states the document was transcribed
into a program that reproduced 79,742,816 floats exactly. With the division form that cannot
hold, so the program that was executed must have used the reciprocal form the document does
not describe. One of the two statements needs correcting.

### 8.2 `routed_scale` is listed but never read — **scope, not a defect**

$\rho$ appears in §1.3 and is multiplied into $\pi_j$ in §4.2. `clover-k3.c` contains no such
multiply and never reads the tensor. At $\rho = 1$ this is exact and the outputs agree, so
nothing is wrong today — but the program would be silently wrong for any checkpoint where
$\rho \neq 1$. It is an assumption, not a constant.

### 8.3 E8M0 scale byte 255 maps to zero — **unstated, and unreachable**

`dq_init` special-cases `s == 255` to `0.0f`; the document's $2^{s-127}$ has no such case.
`Xm` was instrumented to count every group whose scale byte is 255 across a full run:

```
E8M0 scale byte 255 seen: 0        (gate still 23d162dc)
```

It never fires on this prompt. It belongs in §6's "not covered" list beside $\mathbb{Q}$'s
scalar tail, rather than in the equation. Whether it fires on other prompts is untested.

### 8.4 The tail aggregation position — **benign**

§5 prose says the final aggregation "is evaluated at every position"; `clover-k3.c` evaluates
it only at $p = T-1$. The boxed equation only ever reads $p = T-1$, so this is the program
doing less work than the engine, not a disagreement about the model.

---

## 9. Not covered

Everything `k3-model-equation.md` §6 excludes still applies here, and this file adds nothing
to the verification evidence — it is a reading of the source, checked against one gate on one
prompt.

Specifically untested by anything above: whether scale byte 255 occurs on other prompts;
whether $\rho \neq 1$ exists in any checkpoint of this family; and the $\pi_j$ difference at
prompt lengths other than five, where the mean expert multiplicity rises from 1.295 toward
1.7 and the summed $\sum s$ therefore changes magnitude.
