# Context — building Clover ground up

The record for a new line of work: understand K3 as it actually is, then decide a
Direction, execute, observe, and keep what the cycle taught. No outcome is being
predicted here. This file is Context only; where it states a conclusion it is marked.

Offered, not imposed. Delete or correct anything here.

## Where this Context came from

The distilled understanding lives in its own document,
[understanding-k3.md](understanding-k3.md), built in order: the C code first, then the
weights, then everything else. It is the summary that should hold the hours of reading.
This file records the cycle; that one records what was learned.

`include/k3/k3.h` (572 lines), read in full on 2026-09-24. It is the authoritative
description because it is the header of a working engine whose oracle gate reports
`ENGINE MATCHES THE REFERENCE EXACTLY` against the released checkpoint, and it states
that every architectural value was verified against the released `config.json`.

Since then, in `src/core/k3_ops.c`: the full function map, plus `k3_kda_step`, `k3_moe`,
`k3_decoder_layer` and the matmul commentary read line by line.

Cross-checked against the live `config.json` on the machine: 93 layers, hidden 7168,
vocab 163,840, 896 experts, moe_intermediate 3072, 96 heads, kv_lora_rank 512,
first_k_dense_replace 1. Both agree.

**Not yet read:** the MXFP4 matmul body (~400 lines), `k3_mla_cached`, `k3_kda_layer`,
`k3_bind.c`, `k3_trunk.c`, `k3_cache.c`, the tokenizer, the chat layer. No weight
statistics have been computed. Nothing recorded rests on reading the weights themselves.

## Read in the C that the header did not say

- **KDA is an online error-correcting update, not just retrieval.** The recurrence writes
  `S += k · beta(v - u)` where `u = S^T k` is what the state already predicted. The state
  is corrected by its own prediction error, once per head per token — 6,624 times per
  token at K3 scale. `beta` is one scalar per head from a small projection of the input.
- **The routed experts never see the full hidden width.** `k3_moe` down-projects 7168 to
  a latent 3584, runs the 16 chosen experts entirely in that narrower space, sums them
  there, normalizes the sum, and only then projects back up. Only the router and the two
  shared experts read the full 7168.
- Three implementations of the matmul agree **bit for bit** because the four-accumulator
  partition and reduction order are written out by hand and reproduced in each; the AVX2
  path avoids FMA precisely because it would round once where the scalar path rounds
  twice.

## What K3 is

93 decoder layers: **one dense layer (layer 0), 92 MoE layers**. Attention is split
**69 Kimi Delta Attention + 24 Gated MLA**. Hidden 7168, 96 heads.

MLA sits at zero-based layers 3, 7, 11, … 87, then **91 and 92, which are both MLA** —
the every-fourth pattern breaks at the end. The released config lists these one-based
and the reference tests `layer_idx + 1`, which is where an off-by-one would hide.

### The layer recipe, and the part that is easy to get wrong

```
prefix_sum = h
if snapshots non-empty:  h = attn_res([snapshots…, prefix_sum], self_attention_res)
if layer_idx % 12 == 0:  push prefix_sum onto snapshots;  prefix_sum = NONE
h = input_layernorm(h);  h = attention(h)
prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
h = attn_res([snapshots…, prefix_sum], mlp_res)        # unconditional
h = post_attention_layernorm(h);  h = moe(h) or dense_mlp(h)
prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
```

**Block Attention Residuals**: every 12 layers the running residual is snapshotted and
then *cleared*, so on a boundary layer it does not also survive as a separate softmax
source. On every other layer it does. Aggregation is
`softmax(dot(RMSNorm(sources), fold)) @ RAW sources` — the scores come from normalized
keys, the output is a mixture of the *unnormalized* sources. `fold` is one vector,
the norm weight times the projection weight, folded at load time.

This is the most unusual thing in the architecture and the least like a standard
transformer: the residual stream is not a single accumulating path but a small stack of
snapshots that each layer attends over.

### KDA, the 69 layers

A delta-rule linear attention with per-head decay. Order is load bearing:

1. q, k, v projections
2. causal depthwise short conv, width 4, **SiLU fused inside**
3. **L2Norm on q and k only, never v**
4. `beta = sigmoid(b_proj(x))` — a **per-head scalar**, not per channel
5. `z = f_b(f_a(x)) + dt_bias`, one shared low-rank pair for all heads;
   `alpha = exp(lb · sigmoid(exp(A_log[h]) · z))`, **A_log indexed per head**
6. recurrence on state `S[d_k][d_v]`: decay, read `u = Sᵀk`, write `S += k(β(v−u))ᵀ`,
   then output `o = Sᵀq` **from the already-updated state**
7. head-wise RMSNorm over d_v
8. multiply by `sigmoid(g_proj(x))` — **norm first, then gate**
9. o_proj

The engine runs the naive sequential O(T) recurrence, not the chunked parallel form.

### MLA, the 24 layers

**NoPE — no rotation is applied anywhere** — yet the 64 rope dimensions still exist, are
still concatenated onto query and key, and are still cached. The softmax scale is over
the **full** 192 width, not 128. Dropping the unused slots silently changes the model.

The output gate multiplies **before** o_proj with no norm — the opposite order to KDA.
The header is explicit that sharing one code path between them is wrong.

### MoE, the 92 layers

- `scores = sigmoid(logits)` — **independent, they do not sum to 1**
- a frozen bias steers **selection only**; combining weights are gathered from the
  **unbiased** scores, then renormalized
- top-16 of 896 routed experts, plus **2 shared experts at full width added unweighted**
- routed experts work in a **latent width of 3584**, with down/up projections around
  them; RMSNorm is applied to the **aggregate**, not per expert
- the router reads the **full hidden width**, before the latent down-projection

### Numerics

SiTU-GLU: `a = b1·tanh(gate/b1)·sigmoid(gate)`, `u = b2·tanh(up/b2)`, `y = a·u`, with
b1=4, b2=25, so **|y| ≤ 100**. The sigmoid sees the *uncapped* gate.

Routed experts ship in **OCP MX FP4** and are multiplied straight out of it: 0.53125
bytes per parameter (0.5 for the nibble, 1/32 for a shared E8M0 scale per 32 elements).
One 33,030,144-parameter expert is exactly 17,547,264 bytes. Dequantized it would be
132 MB, and a token touches 1,472 of them — 194 GB per token if anything widened them.

## Three invariants the header says must hold

Each is a place where a plausible implementation runs, emits fluent text, and is wrong.
Each is gated by a fixture chosen so that getting it wrong changes the output.

1. `A_log` is indexed **per head**, not per channel. The checkpoint ships head_dim floats
   but only the first num_heads are meaningful; the rest is padding.
2. MLA is NoPE, but the rope slots still exist and the softmax scale covers the full
   head width.
3. The routing bias steers **selection** only; weights come from the unbiased scores.

## What is worth noticing, as observation not conclusion

- The model spends most of its depth on **linear attention** (69 of 93) and uses
  quadratic attention sparingly (24), with two of those at the very end.
- **Sigmoid routing rather than softmax** means expert scores are independent; nothing
  forces them to compete for a probability budget.
- Shared experts are added **unweighted** — every token gets them at full strength.
- The residual path is a **stack of snapshots attended over**, not a single sum.
- Several choices exist only to be *cheap to stream*: MXFP4 held packed through the
  matmul, a fixed layer order so the next read is always known.

## Already measured on this machine (from the benchmark campaign)

- Expert reuse is **strongly depth dependent**: token-to-token persistence at the same
  layer averages 36.84%, but runs 12–17% in early layers and 63–72% in late ones.
- 1,472 expert reads per token (92 layers × 16).
- Trace on disk: 33,469 requests over 15 clean decode tokens, plus `expert_hist.json`.

## Open questions this Context raises

None of these have been investigated.

1. How many of the 896 experts per layer are ever selected at all, over a varied corpus?
2. Does the depth-dependence of expert reuse correspond to anything in the weights, or
   only to the routing?
3. What does the snapshot-stack residual buy that a plain residual does not?
4. Is the 69/24 KDA-to-MLA ratio load bearing, or convention?
5. What is the smallest architecture that keeps whatever matters here?

## State of the work

Direction boundary built into the engine and verified (branch
`clover/direction-boundary`): the engine refuses to generate without a human-owned
Direction and binds the Outcome to its SHA-256. That is enforcement, not a note.

Next stage is **Direction**, and it is the human's. No architecture decision has been
made and none should be inferred from this file.
