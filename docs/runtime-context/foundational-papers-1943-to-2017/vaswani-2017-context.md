# Ashish Vaswani et al. (2017) — *Attention Is All You Need* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Inspect original arXiv TeX source files & cloned reference repositories on k3 CPU: `harvardnlp/annotated-transformer`, `jadore801120/attention-is-all-you-need-pytorch`, `hyunwoongko/transformer`)**: DONE
- **E2 (Scaled Dot-Product Attention $1/\sqrt{d_k}$ variance & softmax Jacobian stabilization Eq. 1 & Footnote 4 p. 4, plus exact `base=65M` / `big=213M` parameter accounting Table 3 p. 9)**: DONE
- **E3 (Sinusoidal Positional Encoding $2\times 2$ linear relative-offset operator $M_k$ & shift-invariant dot product Section 3.5 p. 6, and Multi-Head subspace resolution vs single-head averaging Section 3.2.2 pp. 4–5)**: DONE
- **E4 (Causal masking prefix invariance & leakage control Section 3.2.3 p. 5, Encoder-Decoder sequence transduction with Eq. 3 warmup & label smoothing $\epsilon_{ls}=0.1$ Section 5 pp. 7–8, and Table 1 $O(n^2 d)$ vs $O(n d^2)$ crossover)**: DONE

## 2. What this is

Authors say (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, "Attention Is All You Need", *31st Conference on Neural Information Processing Systems (NIPS 2017)* / `arXiv:1706.03762`):
- **Core Thesis (Sections 1–2, Table 1, pp. 1–2, 6–7)**: Replaces recurrence ($O(n)$ sequential operations, $O(n)$ maximum path length, $O(n d^2)$ per-layer complexity) and convolutions ($O(\log_k n)$ or $O(n/k)$ path length, $O(k n d^2)$ complexity) with **Self-Attention**, which connects any two sequence positions in **$O(1)$ sequential operations** and **$O(1)$ path length** with **$O(n^2 d)$ complexity per layer** (faster than recurrent layers whenever $n < d$, which holds for typical BPE/word-piece sentence lengths $n \sim 30\text{–}50 \ll d_{\text{model}} = 512$).
- **Encoder-Decoder Architecture (Section 3.1, Fig. 1, p. 3)**:
  - **Encoder**: Stack of $N = 6$ identical layers, each with two sub-layers: (1) Multi-Head Self-Attention and (2) Position-wise Feed-Forward Network $\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$ ($d_{\text{model}} = 512, d_{\text{ff}} = 2048$, Eq. 2), wrapped in residual connections and Post-LayerNorm: $\text{LayerNorm}(x + \text{Sublayer}(x))$.
  - **Decoder**: Stack of $N = 6$ identical layers with three sub-layers: (1) Masked Multi-Head Self-Attention (setting upper-triangular positions $j > i$ to $-\infty$ prior to softmax to prevent leftward information flow), (2) Multi-Head Encoder-Decoder Cross-Attention ($Q$ from decoder, $K, V$ from encoder output), and (3) Position-wise FFN.
  - **Shared Embeddings & $\sqrt{d_{\text{model}}}$ Scaling (Section 3.4, p. 5)**: Shares the same weight matrix between source embedding, target embedding, and pre-softmax linear projection, multiplying embedding lookups by $\sqrt{d_{\text{model}}}$.
- **Scaled Dot-Product & Multi-Head Attention (Section 3.2, Fig. 2, Eq. 1, pp. 3–5)**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  Footnote 4 (p. 4) explains why scaling by $1/\sqrt{d_k}$ is critical: for $q_i, k_i \sim \mathcal{N}(0, 1)$ independent, $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean $0$ and variance $d_k$; for large $d_k$, unscaled dot products grow large in magnitude, pushing softmax into saturation where gradients vanish. Multi-Head Attention projects $Q, K, V$ into $h = 8$ subspaces of dimension $d_k = d_v = d_{\text{model}}/h = 64$ so the model can jointly attend to information from different representation subspaces at different positions without single-head averaging.
- **Sinusoidal Positional Encoding (Section 3.5, p. 6)**:
  $$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  Chosen because for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$.
- **Training & Results (Sections 5–6, Eq. 3, Tables 2–4, pp. 7–10)**: Trained with Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$), inverse-sqrt warmup schedule $\text{lrate} = d_{\text{model}}^{-0.5} \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$ ($\text{warmup}=4000$, Eq. 3), residual dropout $P_{\text{drop}}=0.1$, and label smoothing $\epsilon_{ls}=0.1$ (which hurts perplexity as the model becomes less confident, but improves accuracy and BLEU), reaching `28.4` BLEU on WMT14 EN-DE (`213M` params) and `27.3` BLEU (`65M` params).

## 3. Experiments

### E1: Inspect original arXiv TeX source files and cloned reference repositories (`harvardnlp/annotated-transformer`, `jadore801120/attention-is-all-you-need-pytorch`, `hyunwoongko/transformer`)
- **Question**: What author comments or hidden notes exist in the original arXiv `1706.03762` `.tex` files, and how do the three most widely cited PyTorch Transformer repositories compare to the 2017 paper's exact specification (Post-LN vs Pre-LN, $\sqrt{d_{\text{model}}}$ embedding scaling, 3-way weight tying, and Eq. 3 learning rate schedule)?
- **Prediction** (logged in `work/predictions.txt` at 2026-10-26 19:05 UTC):
  - `harvardnlp/annotated-transformer` implements Pre-LN (`x + self.dropout(sublayer(self.norm(x)))`) rather than the paper's Post-LN (`LayerNorm(x + Sublayer(x))`).
  - `hyunwoongko/transformer` omits $\sqrt{d_{\text{model}}}$ embedding scaling and 3-way embedding/output weight tying.
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **Original arXiv `1706.03762` TeX Source Audit (`/root/dm-study/foundations/vaswani-2017/work/tex/`)**:
    - In `ms.tex`, the authors commented out two appendix files that remain inside the arXiv source archive:
      1. `%\input{parameter_attention}` (`parameter_attention.tex`: **"Two Feed-Forward Layers = Attention over Parameters"**): Explicitly writes out the structural equivalence between a 2-layer ReLU FFN and dot-product attention:
         $$\text{FFN}(x, W_1, W_2) = \text{ReLU}(x W_1) W_2 \quad \longleftrightarrow \quad A(q, K, V) = \text{Softmax}(q K^T) V$$
         and reports two unpublished experiments on WMT14 EN-DE (`newstest2013` dev) replacing every Position-wise FFN in the Transformer with **Multi-Head Attention over Parameters (`AOP`)** (`2,097,152` parameters per sublayer, matching `2 * 512 * 2048`): `AOP_1` ($h_p=8, d_{pk}=d_{pv}=64, n_p=1536$: `PPL = 4.92, BLEU = 25.5`) and `AOP_2` ($h_p=16, d_{pk}=d_{pv}=64, n_p=512$: **`PPL = 4.86, BLEU = 25.9`**, outperforming the base model's `4.92 / 25.8`, though $30\%$ slower in step time).
      2. `%\input{sqrt_d_trick}` (`sqrt_d_trick.tex`): Contains a commented-out `\iffalse ... \fi` proof noting that *"The repeated layer norms at each transformer layer encourage $\vec{u}$ and $\vec{v}$ to be normalized"* so $E[\langle \vec{u}, \vec{v}\rangle] = 0$ and $\operatorname{Var}(\langle \vec{u}, \vec{v}\rangle) = d_k$.
  - **Cloned Reference Repositories (`/root/dm-study/foundations/vaswani-2017/repo/`)**:
    1. `harvardnlp/annotated-transformer` (`debc9fd`): In `SublayerConnection` (`the_annotated_transformer.py` lines 344–357), the docstring notes *"Note for code simplicity the norm is first as opposed to last"*, implementing Pre-LN `x + self.dropout(sublayer(self.norm(x)))` rather than the 2017 paper's Post-LN `LayerNorm(x + Sublayer(x))`.
    2. `jadore801120/attention-is-all-you-need-pytorch` (`132907d`): Executed on `k3` CPU (`output shape = (16, 100)`, `44,154,880` parameters for $V=100$). Implements Post-LN and optional 3-way weight sharing, but defaults to `scale_emb=False` unless `scale_emb_or_prj='emb'` is passed, and adds an extra `LayerNorm` immediately after positional encoding.
    3. `hyunwoongko/transformer` (`6328654`): Executed on `k3` CPU (`output shape = (2, 8, 100)`, `44,292,196` parameters for $V=100$). Implements Post-LN (`self.norm1(x + _x)`), but omits $\sqrt{d_{\text{model}}}$ embedding scaling in `TokenEmbedding` and does not tie the source/target/pre-softmax embedding matrices.

### E2: Scaled Dot-Product Attention $1/\sqrt{d_k}$ Variance & Softmax Jacobian Stabilization (Section 3.2.1, Eq. 1 & Footnote 4, p. 4) & Exact Base/Big Parameter Accounting (Table 3, p. 9)
- **Question**:
  1. As $d_k$ increases across $\{16, 64, 256, 512, 1024\}$ for unit-variance queries and keys $q, k \sim \mathcal{N}(0, I_{d_k})$, how do dot-product variance, maximum softmax attention weight, and the Frobenius norm of the softmax Jacobian $\|\frac{\partial \text{softmax}(z)}{\partial z}\|_F$ behave with vs without $1/\sqrt{d_k}$ scaling?
  2. What are the exact parameter counts of `Transformer (base)` and `Transformer (big)` with a shared $V = 37,000$ BPE vocabulary (Section 3.4 & 5.1, Table 3 p. 9)?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Without scaling, $\operatorname{Var}(q \cdot k) \approx d_k$, max attention approaches $1.0$, and $\|\text{J}_{\text{softmax}}\|_F$ collapses by $> 3.5\times$ by $d_k = 1024$, whereas $1/\sqrt{d_k}$ scaling keeps variance $\approx 1.0$, max attention $\approx 0.165$, and $\|\text{J}_{\text{softmax}}\|_F \approx 0.241$ invariant to $d_k$.
  - `Transformer (base)` has $\approx 63.1\text{M}$ parameters (`65M` in Table 3) and `Transformer (big)` has $\approx 214.2\text{M}$ parameters (`213M` in Table 3).
- **Run**: `work/vaswani2017_experiments.py` (`run_e2_attention_and_params()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E2.1 (Scaled vs Unscaled Dot-Product Attention across $n=32$ keys, 500 trials)**:
    | $d_k$ | $\operatorname{Var}(q \cdot k)$ | $\operatorname{Var}(q \cdot k / \sqrt{d_k})$ | Unscaled Max Attn | Scaled Max Attn | Unscaled $\|J_{\text{softmax}}\|_F$ | Scaled $\|J_{\text{softmax}}\|_F$ |
    | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | `16` | `16.55` | `1.0346` | `0.6334` | `0.1663` | `0.28707` | `0.24150` |
    | `64` | `63.87` | `0.9979` | `0.8214` | `0.1649` | `0.19522` | `0.24126` |
    | `256` | `257.89` | `1.0074` | `0.9063` | `0.1642` | `0.11453` | `0.24095` |
    | `512` | `513.78` | `1.0035` | `0.9369` | `0.1626` | `0.08215` | `0.24160` |
    | `1024` | `1023.96` | `1.0000` | **`0.9574`** | **`0.1660`** | **`0.05750`** | **`0.24115`** |
  - **E2.2 (Exact Parameter Accounting, Table 3, p. 9)**:
    - `Transformer (base)` ($N=6, d_{\text{model}}=512, d_{\text{ff}}=2048, h=8, V=37,000$): 6 Encoder layers $= 18,902,016$, 6 Decoder layers $= 25,199,616$, Shared Embedding $= 18,944,000 \implies$ **Total $= 63,045,632$** ($\approx 65\text{M}$ in Table 3).
    - `Transformer (big)` ($N=6, d_{\text{model}}=1024, d_{\text{ff}}=4096, h=16, V=37,000$): 6 Encoder layers $= 75,552,768$, 6 Decoder layers $= 100,730,880$, Shared Embedding $= 37,888,000 \implies$ **Total $= 214,171,648$** ($\approx 213\text{M}$ in Table 3).

### E3: Sinusoidal Positional Encoding Linear Relative-Offset Operator $M_k$ (Section 3.5, p. 6) & Multi-Head Subspace Resolution vs Single-Head Averaging (Section 2 & 3.2.2, pp. 2, 4–5)
- **Question**:
  1. Does the block-diagonal $2 \times 2$ rotation matrix $M_k = \operatorname{diag}_i \begin{bmatrix} \cos(k \omega_i) & \sin(k \omega_i) \\ -\sin(k \omega_i) & \cos(k \omega_i) \end{bmatrix}$ satisfy $PE_{pos+k} = M_k PE_{pos}$ for all positions $pos$ and offsets $k$ (Section 3.5, p. 6), and is the inner product $PE_{pos}^T PE_{pos+k}$ invariant to absolute position $pos$?
  2. When a query must simultaneously retrieve subspace $A$ from position $j_A$ and orthogonal subspace $B$ from position $j_B$ amidst distractor noise, how does Multi-Head Attention ($h=2, d_k=d_v=8$) compare against Single-Head Attention ($h=1, d_k=d_v=16$) with the exact same parameter count?
- **Prediction** (logged in `work/predictions.txt` before running):
  - $\max \|PE_{pos+k} - M_k PE_{pos}\|_\infty < 10^{-13}$ and $\max_k \operatorname{std}_{pos}(PE_{pos}^T PE_{pos+k}) < 10^{-12}$.
  - Single-Head Attention ($h=1$) suffers averaging dilution (`MSE > 0.35`), whereas Multi-Head Attention ($h=2$) achieves `MSE < 0.005` ($> 100\times$ lower MSE).
- **Run**: `work/vaswani2017_experiments.py` (`run_e3_positional_and_multihead()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E3.1 (Sinusoidal Positional Encoding Linear Operator & Shift Invariance, p. 6)**:
    - Across all offsets $k \in \{-13, -1, 0, 1, 7, 29\}$ and positions $pos \in [20, 120]$, $\max \|PE_{pos+k} - M_k PE_{pos}\|_\infty = \mathbf{2.05 \times 10^{-14}}$.
    - Because $\sin(pos \cdot \omega_i)\sin((pos+k)\omega_i) + \cos(pos \cdot \omega_i)\cos((pos+k)\omega_i) = \cos(k \omega_i)$, the inner product $PE_{pos}^T PE_{pos+k} = \sum_{i=0}^{d_{\text{model}}/2-1} \cos(k \omega_i)$ is **strictly invariant to absolute position $pos$** ($\max \operatorname{std}_{pos}(PE_{pos}^T PE_{pos+k}) = \mathbf{8.78 \times 10^{-14}}$), decaying smoothly with distance $k$ (`k=0 -> 256.00`, `k=1 -> 249.10`, `k=5 -> 189.60`, `k=20 -> 157.57`).
  - **E3.2 (Multi-Head Subspace Resolution vs Single-Head Averaging, pp. 2, 4–5)**:
    - Single-Head ($h=1, d_k=d_v=16$): **`MSE = 0.37236`** (a single scalar attention distribution across positions cannot unmix two distinct source positions $j_A, j_B$ carrying independent subspaces).
    - Multi-Head ($h=2, d_k=d_v=8$, identical parameter count): **`MSE = 0.00048`** — a **`782.1x` lower reconstruction MSE**!

### E4: Causal Masking Prefix Invariance (Section 3.1 & 3.2.3, pp. 3, 5), Sequence Transduction Training with Eq. 3 Warmup & Label Smoothing $\epsilon_{ls}=0.1$ (Section 5, pp. 7–8), and Table 1 Complexity Crossover (p. 6)
- **Question**:
  1. Does upper-triangular $-\infty$ causal masking in the Decoder self-attention stack guarantee exact zero leftward information flow when future tokens mutate, and what happens at autoregressive inference when a model is trained without the causal mask?
  2. Does label smoothing $\epsilon_{ls} = 0.1$ (Section 5.4, p. 8) increase nominal perplexity while preserving `100.0%` autoregressive sequence transduction accuracy?
  3. At what sequence length $n$ does core Self-Attention complexity ($2 n^2 d$ MACs) cross core Recurrent complexity ($n d^2$ MACs) for $d = 512$ (Table 1, p. 6)?
- **Prediction** (logged in `work/predictions.txt` before running):
  - With causal masking, prefix logit change under future token mutation is `0.00e+00`; without causal masking, teacher-forced training reaches `100.0%` by looking ahead, but autoregressive greedy decoding collapses to $< 10\%$.
  - With causal masking, both $\epsilon_{ls} = 0.0$ and $\epsilon_{ls} = 0.1$ achieve `100.0%` autoregressive sequence accuracy, with $\text{PPL}(\epsilon_{ls}=0.1) > \text{PPL}(\epsilon_{ls}=0.0)$.
  - Self-Attention vs Recurrent core MAC crossover occurs at $n = d / 2 = 256$.
- **Run**: `work/vaswani2017_experiments.py` (`run_e4_causal_and_transduction()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E4.1 (Causal Masking Prefix Invariance)**: Mutating future tokens at indices $3, 4$ changes prefix logits at indices $0\dots 2$ by **`0.00e+00`** with causal masking vs **`0.3420`** without causal masking.
  - **E4.2 (End-to-End Sequence Transduction Training & Autoregressive Decoding)**:
    - **Causal Mask + $\epsilon_{ls} = 0.0$**: teacher-forced accuracy $= \mathbf{100.0\%}$, autoregressive greedy sequence accuracy $= \mathbf{100.0\%}$, $\text{PPL} = \mathbf{1.0016}$.
    - **Causal Mask + $\epsilon_{ls} = 0.1$ (Section 5.4, p. 8)**: teacher-forced accuracy $= \mathbf{100.0\%}$, autoregressive greedy sequence accuracy $= \mathbf{100.0\%}$, $\text{PPL} = \mathbf{1.1014}$ (confirming p. 8: *"This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score"*).
    - **No Causal Mask (Negative Control)**: teacher-forced accuracy $= \mathbf{100.0\%}$ (cheating via leftward lookahead), but autoregressive greedy sequence accuracy collapses to **`8.6%`**!
  - **E4.3 (Table 1 Complexity Crossover for $d = 512$, p. 6)**: At typical translation sentence length $n = 32$, Self-Attention uses **`1,048,576`** core MACs vs **`8,388,608`** for a Recurrent layer (**`8.0x` fewer MACs**), crossing over at $n = 256$ (`67,108,864` MACs each).

## 4. Findings

1. **Unpublished "Attention over Parameters (`AOP`)" section discovered in the original arXiv `1706.03762` TeX tarball (`parameter_attention.tex`)**: Reading the raw `.tex` files in the arXiv source archive uncovered `parameter_attention.tex` (commented out in `ms.tex`), in which Vaswani et al. showed that $\text{FFN}(x, W_1, W_2) = \text{ReLU}(x W_1) W_2$ is Attention over Parameters and replaced all FFNs in the Transformer with 16-head parameter attention (`AOP_2`), achieving `4.86` PPL and `25.9` BLEU on `newstest2013` (beating the base model's `4.92` PPL and `25.8` BLEU).
2. **Mathematical & empirical verification of $1/\sqrt{d_k}$ scaling (Eq. 1 & Footnote 4), Sinusoidal Positional Encoding ($M_k$ & shift invariance, Section 3.5), and Multi-Head Subspace Resolution ($782\times$ lower MSE, Section 3.2.2)**:
   - Scaling dot products by $1/\sqrt{d_k}$ stabilizes variance at $1.0000$ and preserves the softmax Jacobian norm (`0.24115` vs `0.05750` unscaled at $d_k=1024$).
   - Sinusoidal positional encodings satisfy $PE_{pos+k} = M_k PE_{pos}$ to $2.05 \times 10^{-14}$ and have strictly shift-invariant inner products $PE_{pos}^T PE_{pos+k} = \sum_i \cos(k \omega_i)$ ($\operatorname{std}_{pos} = 8.78 \times 10^{-14}$).
   - Two attention heads ($h=2, d_k=d_v=8$) achieve $782.1\times$ lower MSE (`0.00048` vs `0.37236`) than a single head ($h=1, d_k=d_v=16$) of identical parameter count when retrieving two subspaces from two positions.
3. **Why $\sqrt{d_{\text{model}}}$ embedding scaling is coupled to 3-way weight tying (Section 3.4, p. 5)**: Because the shared matrix $W \in \mathbb{R}^{V \times d_{\text{model}}}$ serves as the pre-softmax linear projection $z = y W^T$, its weights must have fan-in variance $1/d_{\text{model}}$ (`std = d_model ** -0.5`); multiplying token embedding lookups by $\sqrt{d_{\text{model}}}$ restores embedding variance from $1/d_{\text{model}}$ back to $1.0$ so it matches the $O(1)$ sinusoidal positional encoding $PE$. Using PyTorch's default `nn.Embedding` (`std = 1.0`) with `* sqrt(d_model)` makes the initial token embeddings $d_{\text{model}}\times$ larger in variance than $PE$, drowning out positional information!

## 5. Surprises / bugs hit during the work

- **How default `nn.Embedding` (`std=1.0`) + `* sqrt(d_model)` drowns out Positional Encoding (`E4.2`)**: In our first run of `E4.2`, PyTorch's default `nn.Embedding` initialized `shared_emb.weight` with $\mathcal{N}(0, 1.0)$ instead of fan-in $\mathcal{N}(0, d_{\text{model}}^{-1})$. Multiplying `shared_emb(x)` by $\sqrt{d_{\text{model}}}$ gave embeddings variance $d_{\text{model}} = 32$ while `self.pe` had variance $0.5$, drowning out positional order so the sequence-reversal transducer stalled at `14.6%`. Initializing `shared_emb.weight` with `std = d_model ** -0.5` (matching the pre-softmax linear fan-in) immediately brought teacher-forced and autoregressive sequence accuracy to **`100.0%`**.

## 6. READ-LOG

- `research/vaswani-2017/work/vaswani2017.txt` (lines 1–1077: full 15-page NIPS 2017 / arXiv:1706.03762 paper)
- `research/vaswani-2017/work/tex/`: `ms.tex`, `model_architecture.tex`, `sqrt_d_trick.tex`, `parameter_attention.tex`, `background.tex`, `introduction.tex`, `training.tex`, `results.tex`
- `harvardnlp/annotated-transformer`: `the_annotated_transformer.py`
- `jadore801120/attention-is-all-you-need-pytorch`: `transformer/Models.py`, `transformer/SubLayers.py`, `transformer/Modules.py`
- `hyunwoongko/transformer`: `models/blocks/encoder_layer.py`, `models/embedding/token_embeddings.py`, `models/model/transformer.py`

## 7. Files produced

- `research/vaswani-2017/work/predictions.txt`
- `research/vaswani-2017/work/vaswani2017.txt`
- `research/vaswani-2017/work/vaswani2017_experiments.py`
- `research/vaswani-2017/scripts/setup_vaswani2017.sh`
- `research/vaswani-2017/scripts/e1_repos.sh`
- `research/vaswani-2017/scripts/e2_e3_e4_run.sh`
- `research/vaswani-2017/results/setup_vaswani2017.out`
- `research/vaswani-2017/results/e1_repos.out`
- `research/vaswani-2017/results/e2_e3_e4_run.out`
