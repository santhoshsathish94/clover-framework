# Observed architecture: the traces against the official sources

A reference for what the distributed-hosting engine actually does, each claim tied to
both a measurement and an official source. Written because a pile of observations had
accumulated with no map onto what Kimi says the model is, and because one behaviour
survives every explanation available and should be recorded as open rather than guessed
at.

Companion to [CONTEXT.md](CONTEXT.md), which is the chronological cycle log. This file
is the settled picture; that file is how it was reached.

## Provenance

| | |
|---|---|
| engine | `pipeline/bin/pipeline-serve`, fleet resident, server plus 92 layers |
| traces | `CLOVER_VECTORS` with no `CLOVER_VECTORS_OWNER`, so all 92 pods write, about 1.5 GB per five-token prompt |
| prompts | france, japan, germany, once, thethe, hello, question, nonsense, french, genesis, plus a nested prefix ladder |
| official sources | `k3-official/k3_tech_report.txt`, `k3tok/config.json`, `k3-official/vllm/vllm/models/kimi_k3/nvidia/model.py` |

The vllm tree contains a full reference implementation. Where it and the report
disagree in emphasis, the code is taken as authoritative and the report as the
explanation of intent.

## Indexing

`num_hidden_layers` is **93**. `first_k_dense_replace` is **1**, so model layer 1 is
dense: that is the server. Therefore:

```
pod N  =  model layer N+1  (1-indexed)  =  layer_idx N  (0-indexed, as in vllm)
```

Verified, not assumed. `full_attn_layers` is `4 8 12 ... 88 92 93`, which under this
mapping predicts MLA at pods `3 7 11 ... 91 92`. Asking the traces which pods emit a KDA
attention record returns exactly those 24 and 68 KDA pods; the config lists 69 KDA
layers, one of which is layer 1, the server.

## Structure

| observed | source | status |
|---|---|---|
| 12-layer segments | `attn_res_block_size: 12` | confirmed |
| 8 segments, last one short | "8 blocks with 12-layer size, giving a partial final block" | confirmed |
| tail folds 9 sources | "9 total blocks when counting the embedding layer" | confirmed |
| snapshot 0 unchanged across 91 layers | `b0 = h1`, the token embedding is a permanent source | confirmed |
| 3 KDA then 1 MLA | "3:1 mixing ratio" | confirmed |
| pods 91 and 92 both MLA | "An additional Gated MLA layer is placed at the end of the backbone" | confirmed |
| 16 of 896 experts, 2 shared | `num_experts_per_token: 16`, `num_shared_experts: 2` | confirmed |
| router weights sum to 1 | `moe_renormalize: true` | confirmed |

## The AttnRes fold, and what the "reset" actually is

From `model.py`:

```python
self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
self.prev_valid_blocks   = cdiv(layer_idx, self.attn_res_block_size)
...
if self.is_block_write_layer:
    prefix_sum = hidden_states          # replaced, not accumulated
mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
```

So the residual "resetting to near zero" every twelfth layer is not a reset. At a block
write layer the running prefix sum is **replaced** by that layer's own output, the
previous sum having already been written out as a block representation. Nothing is
discarded. What earlier notes described as "stage 20 does not add the carried residual
back" is this, seen from inside the engine.

Each layer performs two folds, one before attention and one before the MoE. The source
counts follow:

```
attention sources = cdiv(layer_idx, 12) + 1
MoE sources       = attention sources + is_block_write_layer
```

Tested against the traced fold-weight vectors at 21 pods spanning every block boundary.
**Twenty match exactly** — pod 12 at 2 and 3, pod 24 at 3 and 4, pod 84 at 8 and 9. The
exception is pod 1, which records no fold weights at all: `transformer-1` contains four
`clover_vectors_write` calls where every other pod has six. An instrumentation gap, not
an architectural difference; the fold outputs are present.

## The nine tail sources, cross-checked between implementations

The reference harness and the production fleet are different binaries on different code
paths. For france at position 0 they agree to four decimal places on every source:

| source | origin | tail (reference) | pod (engine) |
|---|---|---|---|
| b0 | token embedding | 2.0446 | 2.0446 |
| b1 | pod 11 = model layer 12 | 137.7729 | 137.7729 |
| b2 | pod 23 = model layer 24 | 332.9793 | 332.9793 |
| b3 | pod 35 = model layer 36 | 136.5090 | 136.5090 |
| b4 | pod 47 = model layer 48 | 272.0080 | 272.0080 |
| b5 | pod 59 = model layer 60 | 399.7753 | 399.7753 |
| b6 | pod 71 = model layer 72 | 57.0181 | 57.0181 |
| b7 | pod 83 = model layer 84 | 64.5007 | 64.5007 |
| b8 | pod 92 = model layer 93 | 88.4760 | 88.4760 |

Every `b1` to `b7` is the output of the **last** layer of a complete block.

## Behaviour, and where it comes from

**The tail scorer ignores source magnitude.** At position 0 the two largest sources,
399.78 and 332.98, receive weights 0.0164 and 0.0023, while `b0` at 2.04, the smallest,
receives 0.0047. The report gives the reason: the kernel is
`ϕ(q,k) = exp(qᵀ RMSNorm(k))`, and "the RMSNorm prevents layers with large-magnitude
outputs from dominating the weights".

**Fold weights barely move across prompts** because the query is a layer-specific
*learnable* pseudo-query `ql = wl`, not a function of the token.

**Causal masking is exact.** Prompts sharing a prefix produce cos 1.000000 with a maximum
coordinate gap of 0.0000 at every shared position, at every layer. A nested prefix ladder
from one to five tokens gives identical position-0 state at every length.

**Attention contracts prompt differences.** Across 22 layers the attention output is more
prompt-similar than its own input in 21 of them, typically about twice. At a normal layer
this is invisible because the large prompt-specific prefix sum is folded back in; at a
block write layer it is not, so the contracted part is what shows.

**Early positions converge to position-determined states.** By pod 48, position 0's norm
across five unrelated prompts spans 0.15 percent — 272.01, 272.18, 272.24, 272.27,
272.43 — at cos 0.99998 or better, while positions 2 and 4 sit at cos 0.28 to 0.74. An
earlier note treated the large position-1 value at pod 48 as a separate anomaly. It is
not; it is this.

**The MoE can undo the contraction.** At pod 12 every prompt pair involving the nonsense
string falls to about 0.72 after the experts while the attention still agreed at 0.995;
every pair without it stays above 0.99.

## Open: pod 12

Pod 12, model layer 13, is the first block write layer. Its attention output is close to
a **fixed direction**, independent of both position and prompt:

- Between positions within one prompt: cos 0.9964 to 0.9996, in all five prompts tested
  to three decimal places, and out to position 15 on a sixteen-token prompt with norms
  clamped between 3.81 and 3.90.
- Across prompts: cos 0.9949 to 0.9998 for pairs as unlike as biblical English, French
  and `aaaa bbbb cccc`, from inputs that disagree at cos 0.14.

Measured across **all 92 pods** on a sixteen-token prompt, cos between the attention
output at positions 2 and 8, pod 12 is the **only** layer above 0.99, at 0.998938. The
next highest are MLA pods 15 and 31 at 0.970 and 0.968. The other block write pods are
unremarkable: 0.537, 0.719, 0.593, 0.728, 0.835, 0.710.

### What has been ruled out

- **Not a boundary property.** Pods 24, 36, 48, 60, 72 and 84 are the same kind of layer
  and do not do it.
- **Not an artefact of position 0 having no context.** It holds at every position out to 15.
- **Not the early-position convergence.** That affects positions 0 and 1 only; this
  affects all of them.
- **Not an instrumentation artefact.** The traced local is the residual buffer after the
  attention stage, which is the attention output at both boundary and normal layers.
- **Not stated anywhere in the sources.** The report describes all blocks identically and
  the vllm implementation special-cases `is_block_write_layer` alone, with no per-layer
  distinction.

### What would be needed

The anomaly is therefore a property of that layer's learned weights rather than of the
design. Deciding it would need either the KDA internals — the recurrent state `S`, the
channel-wise decay `α`, the write strength `β` of Eq. 1 — which the trace does not
record, or the layer's weights read directly from
`operators/qkv-all/layer-12/operator.bin`. The first means adding tracing inside the
attention, a change to production code rather than a read. The second means parsing a
raw weight layout, where a mis-parse would produce a confident wrong answer.

Neither has been done. The behaviour is recorded as measured and unexplained.

## Caveats

Ten prompts, all short, mostly English, one position per prompt for the depth
trajectories unless stated. Request times printed by the fleet include up to 1.5 GB of
trace writes and measure nothing. `transformer-1` under-reports as described above.
