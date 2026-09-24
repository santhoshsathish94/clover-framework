# i — inside the blocks

**Probes:** how the machine actually works, rather than one anomaly. The layer spine was
already observed; the interiors of the three big blocks were not. This opens them.

Coverage before this cycle — observed: the layer spine, both norms, routing decisions,
the aggregation, and MLA's attention distribution. Black boxes: everything inside the
mixture of experts (92 layers), and everything inside KDA (69 layers).

## Part 1 — the mixture of experts

Taps placed on the latent path: the down-projection in, each chosen expert's raw output,
the weighted sum, the effect of the aggregate norm, the up-projection out, and the shared
expert separately. Sampled at the last position of each batch, which is the position the
output token comes from.

### The instrument went in the wrong function first

The taps were placed in `k3_moe` and produced **zero records**, while routing fired 460
times. Prefill with a streamed source does not run `k3_moe` at all — it runs
`moe_prefill_chunk`, which fetches each unique expert once for the whole batch.

**This is the second time the same mistake has been made in this investigation.** The
routing tap was first placed at the layer call site, which the batched path never writes.
Both times the function that *looks* like the main path is not the one that runs during
prefill. Recorded so the third time is avoided: in this engine, check which MoE path a
configuration takes before instrumenting it.

### Then my sampling aliased with the architecture

The first reading suggested the MoE was nearly idle through the middle of the stack —
expert sums of 0.01 to 0.14 at layers 12, 24, 36, 48. Those are exactly the **snapshot
layers**, where the residual has just been reset to near zero. Sampling every 12th layer
aliased perfectly with the model's 12-layer block period.

Measured across all layers: mean expert sum **4.35** off the boundaries against **1.53**
on them. The MoE is not idle mid-stack; the sample was.

### What the interior shows

Off the boundaries, last position, prompt "The capital of France is":

| layer | latent in | expert sum | after norm | routed out | shared out | shared/routed |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 17.21 | 4.87 | 10.01 | 22.34 | 25.25 | 1.13× |
| 22 | 12.65 | 5.27 | 4.08 | 9.85 | 5.20 | 0.53× |
| 33 | 11.89 | 2.46 | 1.37 | 3.64 | 4.42 | 1.21× |
| 44 | 11.83 | 2.25 | 2.19 | 4.97 | 4.70 | 0.95× |
| 55 | 15.61 | 4.60 | 6.66 | 14.98 | 8.85 | 0.59× |
| 66 | 15.11 | 5.50 | 5.11 | 5.47 | 6.29 | 1.15× |
| 77 | 16.30 | 2.71 | 3.99 | 8.63 | 4.51 | 0.52× |
| 88 | 25.41 | 13.02 | 4.40 | 8.60 | 4.62 | 0.54× |

**Two shared experts do comparable work to all sixteen routed ones.** Mean ratio 0.78×
across 92 layers, and in **22 of them the shared expert is larger**. The part that is
always on, costs nothing to route and is never streamed from disk carries close to half
the feed-forward contribution.

**The sixteen chosen experts contribute remarkably evenly.** The largest single expert
accounts for a mean of **9.5%** of total magnitude where a perfectly even split would be
6.25%; the range is 6.6% to 23.9%, and max/min within a layer is typically 1.1–2.5×.

That is worth putting beside the routing weights, which run from 0.003 to 0.645. **The
experts produce similar-sized outputs; the router's weights do the differentiating.** A
weight of 0.645 and a weight of 0.003 are applied to contributions of comparable
magnitude.

**The aggregate norm does real work.** It rescales the summed experts by a mean of 1.30×
and by as much as **26×** — rescuing aggregates that arrive near zero before the
up-projection sees them.

## Part 2 — KDA

Not yet instrumented.
