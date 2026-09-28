# The model as one equation

`k3-stages.md` establishes the model one kernel at a time, 42 numbered stages across two
layers and then a walk over all 93. This file does the opposite: it states the whole thing
once, as a single composed expression, with nothing left as prose.

The test of this document is narrow and mechanical. **Supply the inputs listed in section 1,
evaluate section 5, and you get the token the engine emits.** Nothing else is needed, and
nothing here is a simplification of something more complicated underneath.

Every operator below is transcribed from an equation already recorded in `k3-stages.md` or
from the walker that reproduced the model bit-exactly. Where the source is the walker rather
than the stages document, it is marked. Section 6 records exactly what is verified and what
is not.

---

## 1. What has to be supplied

### 1.1 The input

| symbol | meaning | this prompt |
|---|---|---|
| $T$ | number of positions | 5 |
| $t_0 \dots t_{T-1}$ | token ids | 1008, 10484, 318, 15383, 387 |

That is the entire input. Everything else is a constant of the model.

### 1.2 The shape constants

$$
\begin{aligned}
E &= 7168 & H &= 96 & D &= 128 & P &= HD = 12288 \\
V &= 163840 & N_L &= 93 & K_c &= 4 & N_{\text{exp}} &= 896 \\
k &= 16 & \text{LAT} &= 3584 & I &= 3072 & \text{SI} &= 6144 \\
D_I &= 33792 & Q_N &= 128 & Q_R &= 64 & Q_H &= 192 \\
V_H &= 128 & \text{KVL} &= 512 & \text{KVW} &= 576 & \text{KVD} &= 256 \\
Q_{\text{lora}} &= 1536 & \text{GRP} &= 32
\end{aligned}
$$

### 1.3 The scalar constants

$$
\begin{aligned}
\epsilon_5 &= \mathrm{fl}_{32}(10^{-5}) = 9.9999997473787516\times10^{-6}
&\qquad \epsilon_6 &= \mathrm{fl}_{32}(10^{-6}) \\
\lambda &= -5 & b_1 &= 4 \\
b_2 &= 25 & \rho &= 1 \ \ (\texttt{routed\_scale}) \\
q_{\text{sc}} &= 1/\sqrt{128} & m_{\text{sc}} &= 1/\sqrt{192}
\end{aligned}
$$

### 1.4 The layer type sets

$$
\begin{aligned}
\mathbb{M} &= \{3,7,11,\dots,87,91\} \cup \{92\} && \text{24 layers using MLA} \\
\mathbb{K} &= \{0,\dots,92\} \setminus \mathbb{M} && \text{69 layers using KDA} \\
\mathbb{E} &= \{1,\dots,92\} && \text{92 layers using MoE; layer 0 is dense} \\
\mathbb{P} &= \{L : L \equiv 0 \bmod 12\} && \text{8 layers pushing a snapshot}
\end{aligned}
$$

### 1.5 The weights

Per layer $L$, and then one model-level group. Dtype matters, because it selects which
projection operator applies.

| symbol | tensor | dtype | used by |
|---|---|---|---|
| $W_E$ | `embed_tokens` | BF16 | $\S5$ |
| $w^{\text{in}}_L$ | `input_layernorm` | BF16 | $\mathcal{N}$ |
| $w^{\text{post}}_L$ | `post_attention_layernorm` | BF16 | $\mathcal{N}$ |
| $n^{A}_L,\ p^{A}_L$ | `self_attention_res_norm`, `self_attention_res_proj` | BF16 | $\mathrm{AR}$ |
| $n^{M}_L,\ p^{M}_L$ | `mlp_res_norm`, `mlp_res_proj` | I8R | $\mathrm{AR}$ |
| $W^{q}_L, W^{k}_L, W^{v}_L$ | `q_proj`, `k_proj`, `v_proj` | I8R | KDA |
| $W^{b}_L$ | `b_proj` | I8R | KDA |
| $W^{fa}_L, W^{fb}_L$ | `f_a_proj`, `f_b_proj` | I8R | KDA |
| $C^{q}_L, C^{k}_L, C^{v}_L$ | `{q,k,v}_conv1d` | F32 | $\mathcal{C}$ |
| $A^{\log}_L$ | `A_log` | F32 | KDA |
| $\tau_L$ | `dt_bias` | F32 | KDA |
| $w^{o}_L$ | `o_norm` | F32 | KDA || $W^{qa}_L, w^{qan}_L, W^{qb}_L$ | `q_a_proj`, `q_a_layernorm`, `q_b_proj` | I8R / BF16 / I8R | MLA |
| $W^{ka}_L, w^{kan}_L, W^{kb}_L$ | `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj` | I8R / BF16 / I8R | MLA |
| $W^{g}_L$ | `g_proj` | I8R | both |
| $W^{o}_L$ | `o_proj` | I8R | both |
| $W^{\text{gate}}_L, W^{\text{up}}_L, W^{\text{down}}_L$ | dense `mlp.*` | I8R | layer 0 |
| $G_L,\ \gamma_L$ | `block_sparse_moe.gate`, `e_score_correction_bias` | I8R / F32 | router |
| $W^{\downarrow}_L$ | `routed_expert_down_proj` | I8R | MoE |
| $W^{\uparrow}_L$ | `routed_expert_up_proj` | I8R | MoE |
| $w^{\ell}_L$ | `routed_expert_norm` | F32 | MoE |
| $W^{(e)}_1, W^{(e)}_3, W^{(e)}_2$ | `experts.e.{w1,w3,w2}` | MXFP4 | MoE |
| $W^{s}_1, W^{s}_3, W^{s}_2$ | `shared_experts.{gate,up,down}` | I8R | MoE |
| $n^{O}, p^{O}$ | `output_attn_res_norm`, `output_attn_res_proj` | BF16 | tail |
| $w^{F}$ | `model.norm` | BF16 | tail |
| $W_{\text{lm}}$ | `lm_head` | BF16 | tail |

Two of these are **folded before use**, and the fold is elementwise, in float32:

$$f^{A}_L = n^{A}_L \odot p^{A}_L, \qquad f^{M}_L = n^{M}_L \odot p^{M}_L, \qquad f^{O} = n^{O} \odot p^{O}$$

---

## 2. The primitive operators

Ten operators. Everything in the model is a composition of these.

Throughout, $\mathrm{fl}_{32}(\cdot)$ is round-to-nearest float32, $\odot$ is elementwise
product, and $\exp$, $\tanh$ are the glibc single-precision functions. **Association order is
part of each definition**, not a formatting choice — `k3-stages.md` records three separate
cases where the mathematically equivalent regrouping drops the match to between 74% and 80%.

### $\mathbb{Q}$ — int8 projection with a fixed 8-lane reduction tree

For an I8R weight of $m$ rows, int8 entries $w_{o,i}$ and per-row float32 scale $s_o$:

$$\big(\mathbb{Q}[W,s]\,x\big)_o \;=\; s_o \cdot \Big[\big((A_0{+}A_4)+(A_2{+}A_6)\big) + \big((A_1{+}A_5)+(A_3{+}A_7)\big)\Big]$$

$$B_c \;=\; \sum_{\substack{i \,<\, 16\lfloor n/16\rfloor \\ i \,\equiv\, c \ (\mathrm{mod}\ 16)}} w_{o,i}\,x_i, \qquad A_j \;=\; B_j + B_{j+8}$$

with $B_0 \dots B_{15}$ **sixteen** independent float32 lanes, each accumulated by
single-rounded FMA over inputs taken in blocks of 16, and the pairing $B_j + B_{j+8}$
performed **once, after the loop**. There is no running total, and there are not eight lanes:
collapsing the sixteen into eight during the loop is arithmetically different and reproduces
16.4% of values rather than 100%.

There is also a scalar tail over $i \geq 16\lfloor n/16\rfloor$, which no shape in this model
reaches. This is `k3_matmul_q8`, and it carries every projection in the model except the
experts, the router and the lm_head.

### $\mathbb{X}$ — MXFP4 projection with double accumulators

$$\big(\mathbb{X}[W,s]\,x\big)_r \;=\; \mathrm{fl}_{32}\Big[\big(Q_0{+}Q_2\big)+\big(Q_1{+}Q_3\big)\Big],
\qquad Q_c = \big(V_0[c]+V_2[c]\big)+\big(V_1[c]+V_3[c]\big)$$

with $V_0..V_3$ four **double** accumulator vectors of 4 lanes, lane $c$ of $V_m$ taking
inputs $i \equiv 4m+c \pmod{16}$. Each 4-bit E2M1 code is decoded and multiplied by its
group's E8M0 scale **in float32 before** entering the double accumulation:

$$\tilde{w}_{r,i} = \mathrm{fl}_{32}\big(\mathrm{E2M1}[c_{r,i}] \cdot 2^{\,s_{r,g(i)}-127}\big), \qquad g(i) = \lfloor i/32 \rfloor$$

The accumulation is a **separate multiply and add, not a fused multiply-add**: each product
is rounded to double before it is added to its lane, so every term is rounded twice. This
is the opposite of $\mathbb{Q}$, which fuses. The natural SIMD reading of "accumulate into
a double vector" is `fmadd`, and that is a different function.

### $\mathbb{B}$ — BF16 projection with a 16-lane double tree

$$\big(\mathbb{B}[W]\,x\big)_r = \mathrm{fl}_{32}\Big[\big(U_0{+}U_1\big)+\big(U_2{+}U_3\big)\Big],
\qquad U_c = \big(C_c + C_{c+4}\big) + \big(C_{c+8} + C_{c+12}\big)$$

with $C_0..C_{15}$ double lanes, lane $c$ taking inputs $i \equiv c \pmod{16}$, and the BF16
weight widened to float32 exactly (a shift, not a rounding). Used only by the lm_head.

### $\mathcal{N}_n$ — RMSNorm

$$\mathcal{N}_n(x; w, \epsilon)_i \;=\; \big(w_i x_i\big) \cdot \mathrm{fl}_{32}\!\left(\frac{1}{\sqrt{\dfrac{1}{n}\displaystyle\sum_{j=0}^{n-1} x_j^2 \;+\; \epsilon}}\right)$$

The sum accumulates in **double**, the reciprocal square root is rounded to float32 **once**,
and the outer product is float32 **left to right**: $(w_i x_i)$ first, then $\times\,\mathrm{inv}$.
Default $\epsilon = \epsilon_5$.

$\mathcal{N}^{\text{blk}}_{n,b}$ applies the same thing independently to each of $n/b$
contiguous blocks of width $b$, reusing one $b$-wide weight.

### $\mathcal{L}$ — per-head L2 normalization

$$\mathcal{L}(v)_i \;=\; v_i \cdot \mathrm{fl}_{32}\!\left(\frac{1}{\sqrt{\displaystyle\sum_{j=0}^{127} v_j^2 \;+\; \epsilon_6}}\right)$$

Not RMSNorm. No division by $n$, no learned weight, $\epsilon_6$ not $\epsilon_5$, applied in
place, and only to $q$ and $k$ — never to $v$.

### $\sigma$, $\mathrm{SiTU}$ — the pointwise nonlinearities

$$\sigma(x) = \frac{1}{1+\exp(-x)}$$

$$\mathrm{SiTU}(g,u) \;=\; \Big[\big(b_1 \tanh(g/b_1)\big)\cdot \sigma(g)\Big] \cdot \Big[b_2 \tanh(u/b_2)\Big]$$

**The sigmoid takes the uncapped $g$**, not $b_1\tanh(g/b_1)$. The capped form is bounded and
plausible and wrong; `k3-stages.md` stage 18 records that it loses 35% of the activation
magnitude and leaves 1 to 19 of 7168 values correct.

### $\mathcal{C}$ — depthwise causal ShortConv with fused SiLU

For channel $c$ at position $t$, taps $w_{c,0..3}$, history $b_0,b_1,b_2$ oldest first:

$$a \;=\; \Big(\big((w_{c,3}\,x_t) + w_{c,0} b_0\big) + w_{c,1} b_1\Big) + w_{c,2} b_2, \qquad \mathcal{C}(x)_{c,t} = a\,\sigma(a)$$

**The current input is the first term**, then history oldest to newest. The history is the
carried convolution state, zero on the first position of a fresh sequence and non-zero on
every decode call.

### $\mathrm{AR}$ — the snapshot aggregation

For an ordered source list $V = \{s_1,\dots,s_d, r\}$ — snapshots first, live residual last —
and a folded direction $f$:

$$\mathrm{AR}(V; f) \;=\; \sum_{v \in V} \pi_v\, v, \qquad
\pi \;=\; \operatorname*{softmax}_{v \in V} \Big\langle \frac{v}{\sqrt{\overline{v^2}+\epsilon_5}},\; f \Big\rangle$$

The arithmetic detail is load-bearing: the sum of squares accumulates in **double**,
sequentially; the scored product $v_i \cdot \mathrm{inv}$ is formed in **float32** before
widening to double; $z$ accumulates in double; each $\pi$ is a double division rounded once
to float32; and the weighted sum accumulates in **float32, source-major**. Accumulating that
last sum in double instead takes the kernel from 100% to 54.99%.

### $\Delta$ — the KDA delta-rule recurrence

Per head, carrying $S \in \mathbb{R}^{128\times128}$ across positions:

$$S \leftarrow \operatorname{diag}(\alpha)\,S, \qquad u = S^\top k, \qquad
S \leftarrow S + \beta\,k\,(v-u)^\top, \qquad o = S^\top q$$

All float32, every sum accumulated **sequentially in $i$ ascending**, no FMA. The output is
read from the **already updated** state, and $q$ arrives pre-scaled by $q_{\text{sc}}$.

### $\mathrm{SA}$ — scaled softmax attention *(source: the verified walker)*

Causal, per head, over positions $s = 0 \dots t$:

$$\mathrm{score}_s = \mathrm{fl}_{32}\big(\mathrm{fl}_{32}\langle q, [\,k^{\text{lat}}_s;\ k^{\text{rope}}_s\,]\rangle_{64} \cdot m_{\text{sc}}\big), \qquad
o = \sum_{s \le t} \mathrm{fl}_{32}\!\left(\frac{e_s}{z}\right) v_s$$

with $e_s = \exp(\mathrm{score}_s - \max_{s'} \mathrm{score}_{s'})$ and $z = \sum_s e_s$
accumulated in double. The dot product is **double, sequential**, over the full $Q_H = 192$
width — the 128 latent components and the 64 rope components scored together — and
$m_{\text{sc}} = 1/\sqrt{192}$ is over that full width, not over the 128.

---

## 3. The attention block

Both variants take the normalized layer input $x_1 \in \mathbb{R}^{7168}$ and return
$\mathrm{attn.out} \in \mathbb{R}^{7168}$. Both end with the same output projection. They
agree on nothing else.

### 3.1 KDA, for $L \in \mathbb{K}$

$$
\begin{aligned}
q,k,v &= \mathcal{C}\big(\mathbb{Q}[W^{q}_L]\,x_1\big),\ \ \mathcal{C}\big(\mathbb{Q}[W^{k}_L]\,x_1\big),\ \ \mathcal{C}\big(\mathbb{Q}[W^{v}_L]\,x_1\big) && \in \mathbb{R}^{12288} \\[4pt]
q,k &\leftarrow \mathcal{L}(q),\ \ \mathcal{L}(k) && \text{per 128-wide head} \\[4pt]
\beta_h &= \sigma\big((\mathbb{Q}[W^{b}_L]\,x_1)_h\big) && h = 0\dots95 \\[4pt]
z &= \mathbb{Q}[W^{fb}_L]\,\big(\mathbb{Q}[W^{fa}_L]\,x_1\big) && E \to 128 \to 12288 \\[4pt]
\alpha_{hD+d} &= \exp\Big(\lambda\,\sigma\big(\exp(A^{\log}_L[h])\cdot(z_{hD+d} + \tau_{L,hD+d})\big)\Big) \\[4pt]
o &= \Delta\big(q\,q_{\text{sc}},\,k,\,v,\,\alpha,\,\beta\big) \\[4pt]
\mathrm{attn.out} &= \mathbb{Q}[W^{o}_L]\ \Big(\mathcal{N}^{\text{blk}}_{12288,128}(o; w^{o}_L) \odot \sigma\big(\mathbb{Q}[W^{g}_L]\,x_1\big)\Big)
\end{aligned}
$$

Note the two indexings in the $\alpha$ line: $A^{\log}$ is **per head**, $\tau$ is **per
channel**. And note that the gate reads $x_1$, not anything attention produced — the shape
forces it, since $W^{g}$ is $[12288, 7168]$ and the attention output is 12288 wide.

### 3.2 MLA, for $L \in \mathbb{M}$ *(source: the verified walker)*

$$
\begin{aligned}
q &= \mathbb{Q}[W^{qb}_L]\ \mathcal{N}_{1536}\big(\mathbb{Q}[W^{qa}_L]\,x_1;\ w^{qan}_L\big) && \in \mathbb{R}^{96\times192} \\[4pt]
c &= \mathbb{Q}[W^{ka}_L]\,x_1 \in \mathbb{R}^{576}, \qquad
c \leftarrow \big[\,\mathcal{N}_{512}(c_{0:512};\ w^{kan}_L)\ ;\ c_{512:576}\,\big] \\[4pt]
\mathrm{kv} &= \mathbb{Q}[W^{kb}_L]\ c_{0:512} \in \mathbb{R}^{96\times256}, \qquad
k^{\text{rope}} = c_{512:576} \in \mathbb{R}^{64} \\[4pt]
k^{\text{lat}}_h &= \mathrm{kv}_{h,0:128}, \qquad v_h = \mathrm{kv}_{h,128:256} \\[4pt]
\mathrm{acc}_h &= \mathrm{SA}\big(q_h,\ k^{\text{lat}}_h,\ k^{\text{rope}},\ v_h\big) \\[4pt]
\mathrm{attn.out} &= \mathbb{Q}[W^{o}_L]\ \Big(\mathrm{acc} \odot \sigma\big(\mathbb{Q}[W^{g}_L]\,x_1\big)\Big)
\end{aligned}
$$

Four details that are each silently wrong if missed, and all four are forced by the shapes or
the source rather than by a name:

- the norm on $c$ covers the **latent 512 only**; the 64 rope components pass through unnormalized
- the rope slot is **not rotated** and is nonetheless **scored**, and the same 64 values serve all 96 heads
- the softmax scale is over the **full 192**, not over the 128 latent part
- the gate is applied **before** the output projection and with **no norm** — the exact opposite
  order from KDA, which norms first and then gates

$k^{\text{lat}}$, $k^{\text{rope}}$ and $v$ for all $s \le t$ are the KV cache. On a decode
call they are read for $s < \texttt{cached}$ from the previous call's writes.

---

## 4. The MLP block

Takes the normalized aggregated residual $x_2 \in \mathbb{R}^{7168}$, returns
$\mathrm{ffn.out} \in \mathbb{R}^{7168}$.

### 4.1 Dense, layer 0 only

$$\mathrm{ffn.out} \;=\; \mathbb{Q}[W^{\text{down}}_0]\ \mathrm{SiTU}\big(\mathbb{Q}[W^{\text{gate}}_0]\,x_2,\ \ \mathbb{Q}[W^{\text{up}}_0]\,x_2\big)$$

### 4.2 MoE, layers 1 to 92

**Route.** Not through $\mathbb{Q}$ — the int8 gate is widened per row and the dot product
accumulates in **double, sequentially**:

$$s_e = \sigma\Big(\mathrm{fl}_{32}\big(\langle G_{L,e},\, x_2 \rangle_{64}\big)\Big), \qquad e = 0 \dots 895$$

$$\mathcal{J} = \operatorname*{top-}k_{\ e}\ \big(s_e + \gamma_{L,e}\big), \qquad
\pi_j = \mathrm{fl}_{32}\!\left(\frac{s_{\mathcal{J}_j}}{\sum_{j'} s_{\mathcal{J}_{j'}} + 10^{-20}}\right)\cdot \rho$$

The bias participates **only in selection**; the weight kept is the unbiased $s_e$. Top-$k$ is
by repeated maximum with a strict `>`, so ties go to the lowest index.

**Compute.** The router read the full width; the experts read a compressed latent:

$$\zeta = \mathbb{Q}[W^{\downarrow}_L]\,x_2 \in \mathbb{R}^{3584}$$

$$\mathrm{accL} \;=\; \sum_{j=0}^{15} \pi_j\ \mathbb{X}\big[W^{(\mathcal{J}_j)}_2\big]\ \mathrm{SiTU}\Big(\mathbb{X}\big[W^{(\mathcal{J}_j)}_1\big]\zeta,\ \ \mathbb{X}\big[W^{(\mathcal{J}_j)}_3\big]\zeta\Big)$$

accumulated in **float32, expert-major** — one expert fully summed in before the next starts.

**Combine.** The norm is applied to the aggregate **once**, not per expert, and the shared
expert reads the **full-width** $x_2$ and is added **unweighted**, bypassing the router:

$$\mathrm{ffn.out} \;=\; \underbrace{\mathbb{Q}[W^{\uparrow}_L]\ \mathcal{N}_{3584}\big(\mathrm{accL};\ w^{\ell}_L\big)}_{\text{routed}}
\;+\; \underbrace{\mathbb{Q}[W^{s}_2]\ \mathrm{SiTU}\big(\mathbb{Q}[W^{s}_1]\,x_2,\ \mathbb{Q}[W^{s}_3]\,x_2\big)}_{\text{shared}}$$

---

## 5. The whole model

State carried into layer $L$: the residual $r$ and the snapshot stack $\mathcal{S}$, both per
position; and per layer, the recurrent state $S_L$, the convolution history, and the KV cache.

### Initial conditions

$$r^{(0)}_p \;=\; W_E\big[t_p\big]\ \ \text{widened BF16}\to\text{F32}, \qquad
\mathcal{S}^{(0)} = \varnothing, \qquad S_L = 0, \qquad \texttt{cached} = 0$$

### The layer, $\mathrm{Layer}_L$

$$
\begin{aligned}
h &= \begin{cases}
\mathrm{AR}\big(\mathcal{S} \cup \{r\};\ f^{A}_L\big) & \mathcal{S} \neq \varnothing \\
r & \mathcal{S} = \varnothing
\end{cases}
&& \text{(1) pre-attention aggregation, guarded}
\\[6pt]
\mathcal{S} &\leftarrow \mathcal{S} \cup \{r\} \quad \text{if } L \in \mathbb{P}
&& \text{(2) snapshot push}
\\[6pt]
x_1 &= \mathcal{N}_{7168}\big(h;\ w^{\text{in}}_L\big)
&& \text{(3) pre-attention norm}
\\[6pt]
r &\leftarrow \begin{cases}
\mathrm{ATTN}_L(x_1) & L \in \mathbb{P} \\
r + \mathrm{ATTN}_L(x_1) & L \notin \mathbb{P}
\end{cases}
&& \text{(4) residual: replace or add}
\\[6pt]
x_2 &= \mathcal{N}_{7168}\Big(\mathrm{AR}\big(\mathcal{S} \cup \{r\};\ f^{M}_L\big);\ w^{\text{post}}_L\Big)
&& \text{(5) pre-MLP aggregation, unguarded, then norm}
\\[6pt]
r &\leftarrow r + \mathrm{MLP}_L(x_2)
&& \text{(6) MLP residual, unconditional}
\end{aligned}
$$

Step (4) is the one that does not look like a residual stream. When a snapshot is pushed, the
incoming residual is **discarded** from the stream and survives only inside $\mathcal{S}$;
the stream restarts from the attention output alone. The engine expresses this as a
`have_prefix` flag cleared by the push and consumed by the add, which reduces exactly to the
condition written above.

### The composition

$$\big(r^{(L+1)}, \mathcal{S}^{(L+1)}\big) \;=\; \mathrm{Layer}_L\big(r^{(L)}, \mathcal{S}^{(L)}\big), \qquad L = 0, 1, \dots, 92$$

### The tail and the emitted token

$$
\boxed{\;
\hat{t} \;=\; \operatorname*{arg\,max}_{0 \le w < 163840}\;
\Big(\ \mathbb{B}\big[W_{\text{lm}}\big]\ \ \mathcal{N}_{7168}\Big(\ \mathrm{AR}\big(\mathcal{S}^{(93)} \cup \{r^{(93)}\};\ f^{O}\big)_{p = T-1}\ ;\ w^{F}\Big)\Big)_w
\;}
$$

with $|\mathcal{S}^{(93)}| = 8$, so the final aggregation runs over **nine** sources. The
aggregation is evaluated at every position; the norm, the lm_head and the argmax are
evaluated at the **last position only**.

### Generation

Prefill is the case $\texttt{cached} = 0$. Decode is the same function called again with one
position and $\texttt{cached} = T$, so $p = \texttt{cached} + t$ and the attention loop runs
$s = 0 \dots p$:

$$\hat{t}_{n+1} \;=\; \mathrm{Model}\big(\hat{t}_n \mid r, \mathcal{S}\!\downarrow, S_L, \text{conv}, \text{KV}\big)$$

$S_L$, the convolution history and the KV cache **carry**. The snapshot stack $\mathcal{S}$
is **rebuilt** from scratch on every call — `int nb = 0;` at the top of `forward` — so each
call aggregates over its own eight snapshots, not the previous call's.

---

## 6. What this reproduces, and what is not covered

### This document was executed

The equation above is not a description of code that exists elsewhere. It was transcribed
into a program and run, and the program reproduces the model:

```
ALL LAYERS 0..92 IDENTICAL, floats checked 79,535,968
  final.aggregate  site 32  nsrc=9   35840/35840
  final.norm       site 33            7168/7168
  logits           site 34          163840/163840
  argmax token     17374   engine emitted 17374   MATCH
  WHOLE MODEL, embedding to emitted token: 79,742,816 floats identical
```

Two defects in this document were found by doing that, and both are now fixed above:

- **$\mathbb{Q}$ had eight lanes where the kernel has sixteen.** As originally written the
  operator reproduced 2,012 of 12,288 values. This was the first operator tested and it
  failed immediately.
- **$A^{\log}$ is stored at width 128 while only the first 96 entries are used**, one per
  head. $a_h = \exp(A^{\log}[h])$ was correct; the storage width was simply not stated, and
  implementing from the document alone gives a shape error.

A third was found later, by `k3-equation-solution.md` step 10, and is also fixed above:

- **$\mathbb{X}$ did not say whether its multiply and add are fused.** Every other operator
  is explicit - $\mathbb{Q}$ says "single-rounded FMA", $\Delta$ says "no FMA" - and
  $\mathbb{X}$ said nothing, while being the one operator whose obvious SIMD translation is
  `fmadd`. The reference is separate multiply and add; the verified build carries
  `-ffp-contract=off`, which compiles the accumulate to `vmulsd` + `vaddsd` rather than
  `vfmadd213sd`.

The other seven operators were transcribed as written and needed no correction.

### Verified

Evaluated against the engine's own tap points, from the token ids, with nothing re-seeded at
any point:

| | floats compared | identical |
|---|---|---|
| prefill, 93 layers + tail | 79,742,816 | 79,742,816 |
| prefill + one decode step | 96,587,584 | 96,587,584 |

Zero mismatches, max ulp 0. Prefill emitted **17374**; fed back, decode emitted **20829**;
both are the tokens the engine emitted. Run-to-run determinism was measured separately:
9,092 records, 83,441,504 float32 values, zero differing across two invocations.

### Where each operator was established

Sections 2, 3.1, 4 and 5 are transcriptions of equations recorded in `k3-stages.md` stages 1
to 42. Section 3.2 ($\mathrm{SA}$ and the MLA block) and the tail equation in section 5 are
transcribed from the walker instead, because `k3-stages.md` records MLA only as four prose
observations and the tail only as a results block. The walker is evidence rather than
inference — it is the code that produced the totals above — but it is a different kind of
source and is marked as such wherever it is used.

### Not covered

- **One prompt.** Five tokens, `The capital of France is`. Different routing, longer contexts
  and different expert sets are untested.
- **One decode step.** Two forward calls, not twenty. Nothing here establishes that the
  carried state stays correct at $\texttt{cached} = 50$.
- **$W^{fa}$'s 128-dim intermediate** is confirmed by inference through $z$, never measured.
- **The $b$ projection** is never directly observed; it is closed through $\beta$.
- **The decode pass's MoE latent internals** are not tapped and are closed indirectly through
  `moe.routed_out`.
- **The scalar tail** of $\mathbb{Q}$ never executes at any shape in this model.
- **The full-recompute branch** of the generation loop, $\texttt{incremental} = \text{false}$,
  is not exercised.
