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
| SiTU-GLU + dense MLP | 1 | written | **median 2.0e-06** end to end at layer 0 |
| MoE | 92 | written | **all 6 links verified, 2 bit-exact** |
| Attention, MLA | 24 | written | **all 6 steps verified, 3 layers** |
| Attention, KDA | 69 | written | **all 10 steps verified, 3 layers** |

**Every stage of the model is now written in closed form and checked against the running
engine.** No stage is taken on trust.

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

## MoE — all six links verified

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

### Checked link by link, on one execution

Taps were added at the five MoE intermediates and a 5-position prompt run so a single chunk
fires the tap unambiguously. **All six links are checked against that same dump**, so they
are mutually consistent rather than six separate measurements. Layers 1, 12, 24, 48, 72, 92:

```
1  z = W_down . x                            median 2.99e-06   worst 3.99e-06
2  latent_normed = N(latent_sum; w_lat)      median 0.00e+00   bit-exact
3  routed_out = W_up . latent_normed         median 1.79e-06   worst 2.86e-06
4  shared = W_sh2 . SiTU(W_sh1 x, W_sh3 x)   median 1.87e-06   worst 3.16e-06
5  ffn.out = routed_out + shared_out         median 0.00e+00   bit-exact
6  latent_sum = sum_j p_j E_j(z)             median 1.37e-07   worst 1.97e-07
```

Link 6 required decoding the experts from the checkpoint: **OCP MX E2M1** nibbles with an
**E8M0** per-group scale, group 32, where the **low nibble is the even element** and a scale
byte of 255 means zero rather than NaN. Reversing the nibble order gives right values in
wrong places, which every statistical check would pass.

### The residuals sort by kernel, which is a check in itself

The expert sum is the **tightest** link at 1.4e-07, an order of magnitude below the int8
links at about 2e-06. That is what the two kernels predict: `k3_matmul_q8` accumulates in
**float32** and applies the row scale once at the end, while `k3_matmul_mxfp4` accumulates in
**double**. The errors partition by which kernel ran, which is independent evidence that both
were read correctly rather than a tolerance chosen to fit.

### The dense layer, verified end to end

Layer 0 is a dense MLP rather than an MoE, and both its input and output are captured, so it
can be checked whole — which also verifies SiTU-GLU and the int8 matmul form:

$$y = W_{\text{down}}\;\mathrm{SiTU}\big(W_{\text{gate}}x,\;W_{\text{up}}x\big),
\qquad \text{inner width } 33{,}792$$

**Median relative error 2.03e-06, worst 4.22e-06** over 8 positions.

### Prefill takes a different function, same mathematics

A prompt longer than one token with a streamed expert source runs `moe_prefill_chunk`, not
`k3_moe`: it batches in chunks of 64, fetches each unique expert once and applies it to every
token that selected it. The tail is line-for-line identical to the per-token path, and
`K3_NO_BATCH_PREFILL` exists to A/B them for bit-identity. **The equation is the same; only
the loop order differs.** This is the second time in this investigation that the function
that looks like the main path was not the one running.

## MLA — all six steps verified

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

Both were measured from a saved state file days before this equation was written.

### Checked step by step

Taps at `mla.q`, `mla.ct`, `mla.kv`, `mla.acc` and `mla.gated`, so a residual localizes to a
stage rather than to "MLA". Layers 3, 47 and 92, positions 0-4, one execution:

```
                                                  layer 3     layer 47    layer 92
1  q   = W_qb . N(W_qa x)                        1.30e-06    1.52e-06    1.71e-06
2  ct  = [N(latent) ; rope], norm on latent only 1.49e-06    1.76e-06    1.47e-06
3  kv  = W_kvb . ct[:512]                        5.29e-07    7.36e-07    8.81e-07
4  acc = sum_j softmax(s) v,  s over sqrt(192)   6.40e-08    6.65e-08    7.19e-08
5  gated = acc * sigmoid(W_g x)                  2.19e-06    7.16e-07    3.42e-07
6  attn.out = W_o . gated                        1.31e-06    2.24e-06    2.19e-06
```

**Verified first attempt, no residual to fix.** Step 4 is again an order of magnitude tighter
than the projections around it, because the attention dot products accumulate in **double**
while the int8 projections accumulate in float32 — the same signature seen in the MoE, and a
second independent confirmation of that reading.

Step 4 passing is what settles the two claims flagged above as easy to get wrong: the
$1/\sqrt{192}$ scale and the **unrotated** shared rope channel. Either being wrong would have
broken that step specifically and left its neighbors intact.

## KDA — all ten steps verified

69 of the 93 layers, and the only place position enters the model. $H=96$ heads,
$D=128$, $P=HD=12288$, kernel $K=4$.

**Projections from $x_t$:**

$$q_t=W_qx_t,\quad k_t=W_kx_t,\quad v_t=W_vx_t\ \in\mathbb{R}^{12288},
\qquad \beta_t=\sigma\big(W_bx_t\big)\in\mathbb{R}^{96}$$

$$z_t=W_{fb}\big(W_{fa}x_t\big)\qquad\text{one shared low-rank pair, } 7168\to128\to12288$$

**ShortConv, causal depthwise, SiLU fused** — applied to $q,k,v$ with separate weights, and
the $K-1$ history carries across calls:

$$a_{t,c}=\sum_{j=0}^{3}w_{c,j}\,x_{t-3+j,\,c},\qquad y_{t,c}=a_{t,c}\,\sigma(a_{t,c})$$

Taps run oldest to newest, so $w_{c,3}$ multiplies the current input.

**L2 normalization on $q$ and $k$ only, per head.** $v$ is deliberately left alone:

$$q_{t,h}\leftarrow \frac{q_{t,h}}{\sqrt{\lVert q_{t,h}\rVert^2+10^{-6}}}$$

This is **not** RMSNorm — no division by $D$, no learned weight, and $\epsilon=10^{-6}$
rather than $10^{-5}$.

**Channel-wise forget gate:**

$$g_{t,i}=\ell\cdot\sigma\Big(e^{A_h}\big(z_{t,i}+\text{dt\_bias}_i\big)\Big),
\qquad \alpha_{t,i}=e^{g_{t,i}},\qquad \ell=-5$$

so $g\in(-5,0]$ and $\alpha\in(e^{-5},1]$. **$A_{\log}$ is indexed per head, not per
channel** — the checkpoint stores $D$ floats but only the first $H$ are nonzero, and
indexing it per channel is silent and fatal.

**The recurrence** — a delta rule over a state $S_h\in\mathbb{R}^{D\times D}$, with
$\hat q=q/\sqrt{D}$:

$$S\leftarrow\operatorname{diag}(\alpha_t)\,S,
\qquad u=S^{\top}k_t,
\qquad S\leftarrow S+\beta_t\,k_t\,(v_t-u)^{\top},
\qquad o_t=S^{\top}\hat q_t$$

$(v-u)$ is the prediction error: that term is what makes this a **delta rule** rather than
plain accumulation. The output reads the **already updated** state.

**Output — norm, then gate, then project:**

$$o_{t,h}\leftarrow\mathrm{N}\big(o_{t,h};w_{o\text{-norm}}\big)\ \text{per head},
\qquad o_t\leftarrow o_t\odot\sigma\big(W_gx_t\big),
\qquad \mathrm{KDA}(x_t)=W_o\,o_t$$

### Where it differs from MLA, and why that matters

| | MLA | KDA |
|---|---|---|
| position | none | carried in $S$ |
| cost in context length | $O(T)$ cache, grows | $O(1)$ state, fixed |
| output order | gate, then project | **norm, then gate, then project** |
| normalization of $q,k$ | none | L2 per head |

The output ordering is the trap: MLA applies no norm before its gate, KDA does. Swapping
them runs and is wrong.

### A second structural cross-check that passes

$$\texttt{kper}=P\cdot D+3P(K-1)=1{,}572{,}864+110{,}592=1{,}683{,}456\ \text{floats}=6{,}733{,}824\ \text{B}$$

At 93 layers that is **626,245,632 B**, against **626,246,288 B** measured in a saved state
file — a 656-byte difference, the size of a file header.

**And it exposes waste.** The state is allocated for all 93 layers (`state_layers = NL`)
though only 69 are KDA, so **161.6 MB per stream — 25.8% of the recurrent state — is
allocated and never touched.** A KDA-only allocation would be 464.6 MB.

### Checked step by step

Taps at the post-conv $q,k,v$, the post-L2 $q,k$, $\beta$, $\alpha$, the recurrence output
and the gated result. Layers 1, 50 and 90, positions 0-4, one execution:

```
                                                  layer 1     layer 50    layer 90
1  q_conv = SiLU(ShortConv(W_q x))               1.92e-06    1.80e-06    1.92e-06
2  k_conv = SiLU(ShortConv(W_k x))               1.61e-06    1.63e-06    2.37e-06
3  v_conv = SiLU(ShortConv(W_v x))               1.04e-06    1.10e-06    1.79e-06
4  q_norm = L2 per head, sqrt(sum+1e-6)          0.00e+00    0.00e+00    0.00e+00
5  k_norm = L2 per head                          0.00e+00    0.00e+00    0.00e+00
6  beta = sigmoid(W_b x)                         6.60e-07    7.76e-07    3.58e-07
7  alpha = exp(-5 sig(e^A_h (z+dt)))             1.85e-06    4.90e-06    7.48e-06
8  o = delta-rule recurrence                     1.58e-07    1.37e-07    2.81e-07
9  gated = N(o per head) * sigmoid(W_g x)        8.91e-07    2.13e-07    6.54e-07
10 attn.out = W_o . gated                        1.68e-06    1.67e-06    3.24e-06
```

**Verified first attempt, no residual to fix.** Three traps are settled by which steps
passed rather than by argument:

- **Steps 4 and 5 are bit-exact**, which confirms the L2 form is $\sqrt{\sum v^2+\epsilon}$
  and not $\sqrt{\overline{v^2}+\epsilon}$. The mean form would rescale every $q$ and $k$ by
  $\sqrt{128}$ and quietly change the attention temperature.
- **Step 7 confirms $A_{\log}$ is per head.** Indexing it per channel would give order-1
  error, not $10^{-6}$.
- **Step 8 confirms the delta rule** — the $(v-u)$ error term, and reading the
  already-updated state.

**Step 7 is the loosest link and it grows with depth**: 1.85e-06, 4.90e-06, 7.48e-06 across
the three layers, worst 1.23e-05. That was worth explaining rather than excusing, so it was
taken apart.

### Why the alpha residual grows — measured, not assumed

**The first explanation given was wrong.** It was attributed to "exponential amplification",
implying $a=e^{A_h}$ grows with depth. Read from the checkpoint, it does the opposite:

```
layer      1     20     50     70     90        correlation with depth
e^A max  3.41   2.36   2.28   1.96   1.63              -0.431
```

Largest at layer 1, smallest at layer 90. The stated mechanism predicted the reverse of the
observation. (`A_log`'s tail is zero at every layer, which independently confirms the
per-head indexing.)

Splitting the step with taps on $z$ before the decay and $g$ after it gives the real account:

```
layer  |x|max  |z|max   dz_abs    dz/|z|   alpha from    alpha from   first-order
                                            TAPPED z       my z        prediction
  1     0.947   8.529  8.82e-06  1.03e-06    1.79e-07     1.85e-06     1.88e-06
 20     1.090   5.929  6.77e-06  1.14e-06    1.79e-07     1.49e-06     1.41e-06
 50     2.169   6.546  1.98e-05  3.03e-06    1.79e-07     4.90e-06     4.92e-06
 70     4.362  10.192  2.64e-05  2.59e-06    2.38e-07     7.69e-06     7.74e-06
 90     7.297  10.767  2.17e-05  2.02e-06    2.38e-07     7.48e-06     7.44e-06
```

**The decay equation is exact.** Fed the tapped $z$, it reproduces $\alpha$ to 1.79e-07 and
that figure is flat across all depths — float32 epsilon. None of the residual is in the
equation.

**All of it is the two chained int8 matmuls that produce $z$**, propagated first order:

$$|\Delta\alpha| \;=\; \alpha\cdot 5\cdot\sigma'(u)\cdot a\cdot|\Delta z|$$

which **predicts the observed error to within 1-6% at every layer**. Nothing else is acting.

Two factors drive the growth, and the one that dominates is not the obvious one:

```
layer   |x|max   |u| mean   sigma'(u) mean   alpha mean
  1      0.947      5.048           0.0284       0.8747
 50      2.169      4.568           0.0329       0.8575
 70      4.362      2.960           0.0900       0.6323
 90      7.297      3.403           0.0805       0.6681
```

- **Activations grow 7.7x** through the stack, so the absolute error in $z$ grows with them.
- **The gate desaturates.** $|u|$ falls from 5.05 to 2.96, so $\sigma'(u)$ rises **3.2x**.
  This outweighs $e^{A}$ falling 2.1x and $\alpha$ falling 1.4x, which is why the effective
  Jacobian rose from about 0.21 to 0.34 while the term originally blamed was shrinking.

**That is a fact about the model, not only about the arithmetic: deeper KDA layers forget
more** — mean $\alpha$ drops from 0.87 to 0.63 — **and are correspondingly more sensitive to
their own input.** The numerical residual was the thing that pointed at it.

---

## Composing the whole layer

Every check above feeds a stage the engine's own input. That measures each equation but says
nothing about whether the errors **compound**. So the layer was run end to end through the
verified equations, each stage fed **my previous output**, carrying my own KDA recurrent
state and ShortConv history across the five positions.

### One thing had to be measured first: the snapshot is not zero

The layer needs the snapshot stack, and snapshot 0 is `pref` entering layer 0 — a vector
never captured. The obvious guess is that it is zero. **Tested and false:** assuming zero
reproduces layer 0's pre-MLP aggregation only to 6.7e-03 - 1.2e-01, orders too large to be
numerical. It was tapped at the push instead of solved by division, which would have been
circular since the softmax weights depend on the snapshot's own score.

A loader bug surfaced here too: `mlp_res_proj` and `self_attention_res_proj` are stored
**I8R**, not BF16 — 7172 bytes is one f32 scale plus 7168 int8, dequantized into the widen
buffer at bind. Reading them as BF16 silently yields 3586 values. No earlier result was
affected, because the AttnRes checks used tapped scores and never needed the fold.

### Layer 1 and layer 90, eight stages each

```
                                        layer 1                 layer 90
stage                              median      worst       median      worst
1  h  = AR(snapshot, r ; foldA)   6.23e-08   1.03e-07     7.56e-08   8.74e-08
2  x  = N(h ; w_in)               8.03e-08   1.26e-07     1.63e-08   1.30e-07
3  a  = KDA(x)                    1.67e-06   2.38e-06     3.73e-06   5.82e-06
4  r  = r + a                     7.32e-07   1.35e-06     7.12e-07   2.11e-06
5  h  = AR(snapshot, r ; foldM)   7.45e-07   1.42e-06     6.13e-07   1.60e-06
6  x  = N(h ; w_post)             3.48e-07   4.67e-07     2.87e-07   4.21e-07
7  y  = MoE(x)                    1.47e-06   2.58e-06     2.82e-06   4.42e-06
8  out = r + y                    1.26e-06   2.43e-06     6.63e-07   2.63e-06
```

Layer 90 carries **eight snapshots**, so step 1 passing at 7.56e-08 independently confirms
the snapshot tap, the source-count formula, and the aggregation over a full stack.

### What propagation actually does

**The errors do not compound.** Composed KDA at layer 1 is 1.667e-06 against 1.676e-06
measured link-by-link from the engine's own input — indistinguishable. Each stage's own
arithmetic dominates what it inherits by roughly an order of magnitude, so feeding my values
forward changes nothing measurable.

**The residual additions reduce error.** Adding a small perturbed quantity to a large clean
residual dilutes the relative error:

```
step 3 -> 4     layer 1  1.67e-06 -> 7.32e-07   x0.44
                layer 90 3.73e-06 -> 7.12e-07   x0.19
step 7 -> 8     layer 1  1.47e-06 -> 1.26e-06   x0.85
                layer 90 2.82e-06 -> 6.63e-07   x0.24
```

**And the dilution strengthens with depth.** Per-stage errors are larger at layer 90 — KDA
more than doubles, 1.67e-06 to 3.73e-06 — yet the **layer output error is lower**, 1.26e-06
at layer 1 against 6.63e-07 at layer 90. The residual grows faster than the error does. This
is the same activation growth that drives the alpha residual, acting in the opposite
direction: it hurts the gate and helps the stream.

**No routing decision flipped.** 0 of 5 positions at both layers selected a different top-16
than the engine. Accumulated error is nowhere near the margin between the 16th and 17th
expert.

### What this does not establish

Two layers, five positions, one prompt, and a single layer composed at a time rather than 93
chained. Whether error accumulates across **layers** — where the residual is carried forward
rather than re-derived — is a different question and is not answered here.

**Evidence** — vectors from `f6.bin` (9 state sites x 93 layers x 64 positions) and
per-source scalars from `s6.src` (every AttnRes source, 225 positions), both produced by taps
that leave the engine bit-identical to its reference. Verification scripts `eq_norm.py`,
`eq_attnres.py`, `eq_attnres2.py`, `lookahead.py`.
