# Upstream context: MoonshotAI/Kimi-K3 (official repository)

Analysis of the official repository, read directly rather than from summaries.
Clone at `c:\personal\oss\k3-official\Kimi-K3` (shallow, `--depth 1`).
One of a set; see also the vLLM execution analysis in this folder.

## Intended outcome

Establish what the *model* is, as its authors define it, separately from what
`kimi-k3-in-c` does — because every measurement in `clover-intelegence` was
taken against that C reimplementation and had been described as "the model".

## What the repository actually contains

Four files. **No model code and no weights.**

| file | size |
|---|---|
| `README.md` | 44.6 KB — model card, architecture table, evaluation, deployment |
| `k3_tech_report.pdf` | 1.75 MB — **not yet read** |
| `LICENSE` | 3.0 KB — Kimi K3 License |
| `assets/kimi-logo.png` | 85.9 KB |

The implementation lives on HuggingFace, not here. We already hold it:
`/root/k3model/modeling_kimi_linear.py` on `k3`, copied to
`c:\personal\oss\k3-official\modeling_kimi_linear.py`. Deployment is delegated
to third-party engines, named below.

## Architecture, as the authors state it

| | official |
|---|---|
| total parameters | **2.8T** |
| **activated parameters** | **104B** |
| layers | 93 |
| dense layers | 1 |
| attention composition | **69 KDA + 24 Gated MLA** |
| attention hidden dimension | 7168 |
| attention heads | 96 |
| latent MoE dimension | 3584 |
| MoE hidden dimension per expert | 3072 |
| experts | 896 |
| selected experts per token | 16 |
| shared experts | 2 |
| vocabulary | 160K |
| context length | 1,048,576 |
| activation | SiTU-GLU |
| vision encoder | MoonViT-V2, 401M |
| **quantization** | **MXFP4 weights / MXFP8 activations, QAT from SFT onward** |

Described as "the world's first open 3T-class model", built on **Kimi Delta
Attention (KDA)** and **Attention Residuals (AttnRes)**, with a "Stable
LatentMoE framework" claimed at ~2.5x the scaling efficiency of Kimi K2.

## Every structural constant matches our engine

Checked against `clover-one.c` and the dataset on `k3`:

| | official | ours | |
|---|---|---|---|
| layers | 93 | `NLAY 93` | ✓ |
| attention split | 69 KDA / 24 MLA | 69 / 24 | ✓ |
| hidden | 7168 | `E 7168` | ✓ |
| heads | 96 | `H 96` | ✓ |
| latent MoE | 3584 | `LAT 3584` | ✓ |
| expert hidden | 3072 | `I_ 3072` | ✓ |
| experts / top-k | 896 / 16 | `NEXP 896` / `TOPK 16` | ✓ |
| shared experts | 2 | 6144 = 2 x 3072 | ✓ |

**The two parameter counts reconcile against our own byte measurements**
(derived, closes to ~1%):

```
routed experts  896 x 92 x 33,030,144             = 2.72T   -> the "2.8T"
  as stored     896 x 92 x 17,547,264 B           = 1.446 TB = experts.direct
activated       16 x 92 x 33,030,144              = 48.6B
  plus trunk    trunk.bin 54.47 GB int8           ~ 54.5B
                                            total ~ 103B   -> the "104B"
```

So our `Q` (53.8 GB a token, read every token) **is** the always-activated
~55B, and our `X` (25.8 GB a token) **is** the 48.6B of routed experts. The
official activated-parameter figure and our measured per-token byte split are
the same fact in two units.

## What this corrects

1. **MXFP8 activations, not FP32/FP64.** The model is quantization-aware
   trained for MXFP4 weights *and* MXFP8 activations. Our engine's
   `-ffp-contract=off -fno-fast-math` FP64-accumulate path is a deliberately
   numerically-exact reimplementation — it is not the arithmetic the model was
   trained for or is deployed with. Any statement about arithmetic cost must
   say which of the two it describes.
2. **"1T MoE" was wrong in our notes.** It is 2.8T total, 104B activated.
3. **Deployment is explicitly delegated**, to vLLM, SGLang and TokenSpeed. The
   authors publish no serving implementation of their own, so "how Kimi K3
   runs" is answered in those engines, not here. That is the next context file.

## The tech report — read in full

47 pages, extracted to `c:\personal\oss\k3-official\k3_tech_report.txt`. Method
is in `extract-pdf-text.py` beside it: fetch the `pypdf` wheel, verify its
sha256, `zipimport` it from a temp directory, delete it. Nothing installed.

### §2.1.1 KDA — the recurrence, and its parallel form

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top, \qquad \tilde o_t = S_t^\top q_t$$

> "**Chunkwise parallel form** — Following Kimi Linear, KDA is recurrent across
> chunks and **parallel within each chunk**."

Block structure is confirmed as `3 x (KDA + LatentMoE)` then
`1 x (Gated MLA + LatentMoE)`, repeated, "with an additional Gated MLA layer
placed at the end of the backbone, ensuring that the final layer always
performs global attention". 23 x (3+1) + 1 = **69 KDA + 24 MLA**.

### §5.1 — the serial recurrence is attacked at four levels

The report names the problem in our own terms:

> "The serial dependence of the KDA state **is at odds with the GPU's
> preference for wide, uniform parallelism**, and it manifests as a different
> bottleneck in each execution regime. We design a dedicated kernel for each."

| level | mechanism |
|---|---|
| within a chunk | token-parallel |
| across chunks | **FlashKDA**, CUTLASS, *overlaps* intra-chunk compute with cross-chunk state propagation; "decomposes the work into **token-parallel stages and a head-parallel recurrence**" |
| within a rank | SM-level **context-parallel planner** splits the sequence across SMs of one rank, no cross-device traffic |
| across ranks | **KCP** (§5.1.2) |

FlashKDA "serves both training and inference prefill and is auto-dispatched as
a backend of flash-linear-attention" — which is the `chunk_kda` our HuggingFace
read bottomed out at, and `chunk_intra_token_parallel.py` in vLLM.

**§5.1.2 KCP is the key idea.** Each segment's effect decomposes into a
cumulative transition $M^{t\leftarrow 1}_{[i+1]}$ acting on the incoming state,
plus a state generated locally from zero — both computable from local tokens
*before* the incoming state exists. Those compose **associatively**, so:

> "the incoming state of each rank can be recovered by a **prefix scan** ...
> KCP requires only a **fixed-size all-gather** and achieves **linear compute
> scaling**."

**So the sequence dependence is not worked around, it is dissolved by
associativity.** A serial recurrence with an associative composition is a scan,
and scans parallelise. This is the piece our engine does not have at all.

### §5.4.2 — but decode is explicitly a different regime, and it is ours

> "Compared with KDA prefill, **KDA decoding presents a distinct set of
> challenges: the primary bottleneck shifts from exploiting parallelism to
> efficiently managing the evolving recurrent state**, which is updated in
> place at every decoding step."

And on the routed experts, in the authors' own words:

> "For routed experts, **at small batch sizes, the group GEMMs reduce to
> memory-bound streaming of weight matrices** — a regime for which conventional
> tile-centric kernels are poorly suited due to their compute-oriented design."

Their answer is not more parallelism. It is **WarpDecode**, a *token-centric*
kernel where "each warp is responsible for one output neuron and **streams the
associated weights directly from memory**", plus an **offline weight-layout
permutation** to cut runtime dequantization cost.

**This independently confirms our decode measurement.** We measured 2.11
FLOP/byte, bandwidth-bound, ~82% of the arithmetic idle, and concluded the
machine should be built around streaming rather than around arithmetic. Moonshot
reached the same conclusion for the same regime and wrote a streaming kernel.
Our offline `experts.direct` repack is the same move as their layout permutation.

Also in §5.4.2: MTP rollback is solved by caching only the *projected inputs*
(far smaller than the state), replaying accepted tokens on-chip in one fused
kernel — "independently proposed in the concurrent work ReplaySSM", which is
`nvidia/ops/recoverssm.py` in vLLM.

### §5.4.1 and §5.4.3 — serving

KDA states are packed into the **same paged pool** as MLA KV at equal page size.
Prefix hashing runs on fine **512-token hash blocks** while the physical block
stays coarse (1024-6144); KDA checkpoints are persisted only at a sparse subset
of hash-aligned endpoints, typically conversation-turn boundaries. Result: "any
shared prefix is reusable at any 512-token boundary". Fleet level adds
cache-aware affinity scheduling (consistent hashing to a primary and secondary
cluster) and budget-based admission control.

## What this confirms, and what it corrects

**Confirms:** the 93-layer depth is real and sequential; decode is
state-management- and bandwidth-bound, not parallelism-bound; small-batch expert
GEMMs are memory-bound weight streaming; offline weight relayout is the right
move.

**Corrects:** the sequence dimension inside a layer is *not* inherently serial.
Our engine's strict per-position scan is one implementation; the model's own
formulation is a chunkwise, token-parallel, associatively-composable scan. Any
claim we make about prefill cost is a claim about `kimi-k3-in-c`, not about
Kimi K3.

## What could not be established

- The report gives no absolute latency or throughput figures we can compare
  against, and no hardware configuration for the serving numbers. Nothing here
  is a measurement we can reproduce.
- FlashKDA [14], WarpDecode [12] and ReplaySSM [25] are cited works, not read.
- Whether vLLM implements all of this, or a subset, is the next context file.

