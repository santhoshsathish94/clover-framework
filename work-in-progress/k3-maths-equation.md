# The K3 equation

**What each stage of a token's journey actually computes, written in closed form and checked
against the running engine.**

Not an approximation and not a fit. Every line here is transcribed from
[`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) and then required to
reproduce vectors captured from a real forward pass. A stage is only marked verified when
the written equation regenerates the engine's own output from the engine's own input.

Started 2026-09-26. In progress — attention and the MoE are not yet verified, and the table
below says so.

## Status

| stage | per token | equation | evidence |
|---|---:|---|---|
| RMSNorm | 186 | written | **bit-exact, 2,976 / 2,976** |
| AttnRes | 185 (1,002 source passes) | written | **median 4.5e-08, 99% under 1e-6** |
| Router | 92 | written | **bit-exact, 100%** |
| MoE | 92 | written | not verified — needs the MXFP4 expert weights |
| Attention, MLA | 24 | written | cache footprint matches to the byte; arithmetic unverified |
| Attention, KDA | 69 | not written | — |

## Constants

Read from the released configuration, not assumed.

```
hidden E            7168        rms_eps        1e-5
layers              93          attn_res_block 12
n_experts           896         topk           16        n_shared 2
latent L            3584        moe_inter I    3072
moe_renorm          1           latent_norm    1         routed_scale 1.0
situ_b1             4.0         situ_b2        25.0
first_dense         1           dense_inter    33792
MLA  n_heads 96  q_lora 1536  kv_lora 512  qk_nope 128  qk_rope 64  v_head 128  out_gate 1
KDA  heads 96    head_dim 128  conv_k 4      gate_lb -5.0
MLA at one-based layers 4, 8, ... 92, and 93; every other layer is KDA
```

## The primitives

### RMSNorm — verified bit-exact

$$\mathrm{N}(x;w) \;=\; w \odot x \cdot \Big(\overline{x^2}+\epsilon\Big)^{-1/2}$$

The engine accumulates $\sum x_i^2$ in **double** and computes the reciprocal square root in
float32, then multiplies `w[i] * x[i] * inv` in that order. Reproducing those three choices
gives bit equality; departing from any of them does not.

*Checked:* 93 layers x 16 positions x 2 sites = 2,976 vectors, **all bit-identical**, worst
relative difference `0.000e+00`.

### SiTU-GLU

$$\mathrm{SiTU}(g,v)\;=\;\Big[b_1\tanh(g/b_1)\,\sigma(g)\Big]\;\odot\;\Big[b_2\tanh(v/b_2)\Big],
\qquad b_1=4,\; b_2=25$$

The sigmoid takes the **uncapped** gate. Feeding it the capped value yields a bounded,
plausible, wrong function — the engine's source flags this explicitly.

### AttnRes — verified

$$\mathrm{AR}(V;f)\;=\;\sum_{v\in V}\pi_v\,v,
\qquad
\pi \;=\; \operatorname*{softmax}_{v\in V}\Big(\big\langle\, v\,(\overline{v^2}+\epsilon)^{-1/2},\; f \,\big\rangle\Big)$$

with $f_A = w_{\text{ares\_norm}}\odot w_{\text{ares\_proj}}$ before attention and
$f_M = w_{\text{mres\_norm}}\odot w_{\text{mres\_proj}}$ before the MLP. The **key is the
normalized source; the value is the raw one.**

$V$ is the pushed snapshot stack followed by the running residual. Source counts follow
directly and match measurement:

```
pre-attention aggregation    nsrc = ceil(L/12) + 1     absent at L = 0
pre-MLP aggregation          nsrc = floor(L/12) + 2
```

which gives a mean of 3.5 at layer 24 and 9 at layer 92, both as measured.

*Checked in three parts:* the per-source reciprocal against the captured L2, `1.192e-07`
(exactly float32 epsilon); the softmax against the captured scores, `2.057e-07`; and the
output $\sum\pi_v v$ rebuilt from captured source vectors — **median `4.5e-08`, 99% under
`1e-6`** over 2,960 aggregations. The residual is float32 accumulation in the engine against
float64 in the check, not a difference of form.

### Router — verified bit-exact

$$\mathcal{T}\;=\;\operatorname*{top\text{-}16}_{e}\Big(\underbrace{\sigma\big(\langle W_e,x\rangle\big)}_{\text{score}} + b_e\Big),
\qquad
p_j\;=\;\frac{\sigma\big(\langle W_j,x\rangle\big)}{\sum_{k\in\mathcal{T}}\sigma\big(\langle W_k,x\rangle\big)}$$

$W$ is $[896,7168]$, one projection and an elementwise sigmoid. Selection uses
score **plus bias**; the returned weight is the **unbiased** score, renormalized over the
chosen 16. Ties break in first-index order.

*Checked:* reconstructed from the packed checkpoint and run on each layer's own captured
input — **100.0%** of traced picks reproduced, all 92 layers, 225 positions.

## The layer

For layer $L$ at one position, with running residual $r$, snapshot stack $S$, hidden $h$:

$$
\begin{aligned}
&h \leftarrow \mathrm{AR}\big(S\cup\{r\};\,f_A\big) && \text{if } S\neq\emptyset\\[2pt]
&\textbf{if } L\equiv 0 \bmod 12:\quad S\leftarrow S\cup\{r\},\quad r\ \text{is \textbf{replaced}, not added}\\[2pt]
&r \leftarrow r + \mathrm{Attn}_L\big(\mathrm{N}(h;w_{\text{in}})\big) && \textcolor{gray}{\textit{not yet written}}\\[2pt]
&h \leftarrow \mathrm{AR}\big(S\cup\{r\};\,f_M\big)\\[2pt]
&r \leftarrow r + \mathrm{MoE}_L\big(\mathrm{N}(h;w_{\text{post}})\big),\qquad h \leftarrow r
\end{aligned}
$$

The last line is the one that is easy to get wrong: **the running residual absorbs the FFN
output as well as the attention output.** A snapshot is therefore the previous layer's
`layer.out`, not its `resid.post_attn`. Assuming otherwise reconstructs the aggregation
correctly only 12% of the time.

## MoE — written, not verified

$$\mathrm{MoE}(x)\;=\;W_{\uparrow}\;\mathrm{N}\!\Big(\sum_{j\in\mathcal{T}} p_j\,E_j\big(W_{\downarrow}x\big);\,w_\ell\Big)\;+\;\mathrm{Sh}(x)$$

$$E_j(z)=W^{2}_j\,\mathrm{SiTU}\big(W^{1}_j z,\;W^{3}_j z\big),
\qquad
\mathrm{Sh}(x)=W^{\text{sh}2}\,\mathrm{SiTU}\big(W^{\text{sh}1}x,\;W^{\text{sh}3}x\big)$$

Three structural facts that a textbook MoE does not have:

- **Experts do not act on the 7168 state.** It is down-projected to a 3584 latent first, and
  every expert lives there. $W_\downarrow$ is $[7168\to 3584]$, $W_\uparrow$ is the reverse.
- **The aggregate is normalized, not each expert.** One RMSNorm after the weighted sum.
- **The shared expert reads the original full-width $x$ and is added unweighted**, outside
  the normalization and outside the routing.

Routing happens on the full width, before the down-projection.

## MLA — written, not yet numerically verified

24 of the 93 layers. One projection produces both the compressed key-value latent and a
shared positional channel.

**Query, low rank:**

$$\tilde q = \mathrm{N}\big(W_{qa}\,x_p;\,w_{qa}\big)\in\mathbb{R}^{1536},
\qquad q = W_{qb}\,\tilde q \in\mathbb{R}^{96\times192},
\qquad q_h=\big(q^{n}_h\Vert q^{r}_h\big),\; 128+64$$

**Key and value, one projection:**

$$c = W_{kva}\,x_p\in\mathbb{R}^{576},
\qquad \hat c=\mathrm{N}\big(c_{1:512};\,w_{kva}\big),
\qquad \rho_p = c_{513:576}\in\mathbb{R}^{64}$$

$$\big[k^{n}_h \,\Vert\, v_h\big] = W_{kvb}\,\hat c \in\mathbb{R}^{96\times256}$$

The norm covers **the latent only** — never $\rho$. And $\rho_p$ is a **single 64-value
channel shared by all 96 heads.**

**Scores, causal:**

$$s_{h,j}=\frac{\big\langle q^{n}_h,\,k^{n}_{h,j}\big\rangle+\big\langle q^{r}_h,\,\rho_j\big\rangle}{\sqrt{192}},\qquad j\le p$$

$$o_h=\sum_{j\le p}\operatorname{softmax}_j\big(s_{h,\cdot}\big)\,v_{h,j}$$

**Output, gated before projection:**

$$\mathrm{MLA}(x_p)=W_o\Big[\sigma\big(W_g\,x_p\big)\;\odot\;\textstyle\bigoplus_h o_h\Big]$$

### Four things that are easy to get wrong

- **The scale is $1/\sqrt{192}$, over $q_n+q_r$, not $1/\sqrt{128}$.**
- **There is no rotary embedding.** Despite the name, $\rho$ is never rotated — there is no
  `cos`, `sin` or `theta` anywhere in the engine. Position enters this model through the 69
  KDA layers; the 24 MLA layers are position-free. Dropping the $\langle q^r,\rho\rangle$
  term entirely is the silent bug the source warns about: it still runs and is wrong.
- **The gate is applied before $W_o$ and is not normalized**, unlike KDA which norms first.
- **The cache stores the expanded per-head $k,v$, not the latent.** That is a deliberate
  engine choice, not a property of the equation.

### A structural cross-check that already passes

The equation predicts the cache footprint, and it matches measurement to the byte:

```
expanded    24 layers x (96 x 256 + 64) x 4 B  =  2,365,440 B/position   measured 2,365,440
latent      24 layers x (512 + 64)     x 4 B  =      55,296 B/position   measured     55,296
```

Both were measured from a saved state file days before this equation was written. Shape
confirmed; the arithmetic itself is still unverified against captured vectors.

## KDA — not yet written

69 of the 93 layers, and the ones carrying positional information. Recorded as absent rather
than sketched.

---

**Evidence** — vectors from `f6.bin` (9 state sites x 93 layers x 64 positions) and
per-source scalars from `s6.src` (every AttnRes source, 225 positions), both produced by taps
that leave the engine bit-identical to its reference. Verification scripts `eq_norm.py`,
`eq_attnres.py`, `eq_attnres2.py`, `lookahead.py`.
