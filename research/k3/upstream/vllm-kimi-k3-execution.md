# Upstream context: vLLM's Kimi K3 implementation

How Kimi K3 is actually *executed*, read from source rather than from the docs
site. Full clone at `c:\personal\oss\k3-official\vllm` (7,660 files).
Companion to [moonshot-kimi-k3-official.md](moonshot-kimi-k3-official.md),
which covers the model definition and the tech report.

## Where it lives

**`vllm/models/kimi_k3/`** — *not* under `model_executor/models/`, which is
where a stale sparse-checkout pattern sent me first. 53 Python files:

| | |
|---|---|
| `nvidia/` | the primary backend: `model.py` (2,157 lines), `kda.py`, `mla.py`, `dspark_mla.py`, `latent_moe_runner.py`, `mtp.py`, `low_latency_gemm.py` |
| `amd/` | a parallel backend with its own `kda.py`, `mla.py`, `mtp.py`, `latent_moe_runner.py` |
| `*/ops/third_party/kda/` | `chunk.py`, `chunk_intra.py`, **`chunk_intra_token_parallel.py`**, `fused_recurrent.py` |
| `nvidia/ops/` | `attn_res.py`, `recoverssm.py`, `latent_moe_tail.py`, `fused_mla_key_concat_kv_cache.py` |
| `nvidia/ops/cute_dsl/` | CUTLASS-DSL kernels incl. `allreduce_rmsnorm_reduce_scatter_early_exit.py`, `fused_add_multicast_gemm.py`, `lamport_copy.py` |
| `common/` | `mtp.py`, `mm_preprocess.py` |

Plus `csrc/libtorch_stable/kimi_k3/*.cu`, 6 dedicated benchmarks under
`benchmarks/kernels/`, and ~20 tests under `tests/models/kimi_k3/`.

## The layer loop is still strictly sequential

`KimiLinearModel.forward`:

```python
for layer_idx, layer in enumerate(
    self.layers[self.start_layer : self.end_layer], start=self.start_layer
):
    hidden_states, prefix_sum, residual = layer(...)
```

No layer parallelism, in the production engine, on either backend. **The
93-deep chain is real.** What `start_layer:end_layer` *does* give is
**pipeline parallelism** — each PP rank owns a contiguous slice of layers and
passes `IntermediateTensors` (`hidden_states`, `residual`) to the next.

## What *is* parallel inside a layer

`KimiDecoderLayer.forward`, with sequence parallelism on:

```python
hidden_states, prefix_sum, residual = self._pre_attn_norm(...)   # AttnRes fold + norm
if self.use_sequence_parallel:
    hidden_states = sp_all_gather(hidden_states)                 # need whole sequence
    hidden_states = hidden_states[: positions.shape[0]]
hidden_states = self._run_self_attn(positions, hidden_states)
if self.use_sequence_parallel and hidden_states.shape[0] == M:
    hidden_states = sp_reduce_scatter(hidden_states)             # back to sharded
hidden_states, prefix_sum, residual = self._post_attn_norm(...)
hidden_states = self.mlp(hidden_states)
```

This is tech-report §5.4.2 realised: "the TP all-reduce is decomposed into a
reduce-scatter and an all-gather, with the intra-block kernel inserted between
the two collectives, operating on the sequence-sharded hidden states". The model
enters and leaves sharded (`sp_shard` / `sp_all_gather` in the model forward)
and is only full-sequence for the duration of attention.

**So "processes a layer in parallel" means the token dimension is split across
ranks** and every non-attention part of the layer runs on all ranks at once on
different tokens.

### The full parallelism inventory

| axis | where | note |
|---|---|---|
| pipeline (PP) | `start_layer:end_layer`, `IntermediateTensors` | layer slices across ranks |
| tensor (TP) | throughout | heads and FFN split |
| sequence (SP) | `sp_shard` / `sp_all_gather` / `sp_reduce_scatter` | tokens split; SP and PP are mutually exclusive here — asserted in the code |
| expert (EP) | `KimiK3MegaMoEExperts` | released HF reference has `ep_size = 1`; vLLM does not |
| context (CP) | `tests/distributed/test_kimi_linear_context_parallel.py` | KCP from §5.1.2 |
| within KDA | `chunk_intra_token_parallel.py` | token-parallel inside a chunk |

## MoE: overlap rather than fusion

`KimiMoE._maybe_overlap_router_and_down_proj` — the router gate and the latent
down-projection both read the same `hidden_states`, so they run **on two CUDA
streams** joined by `maybe_execute_in_parallel`, with the side stream used only
below a token-count threshold (i.e. in decode).

Note this differs from the report, which says they "fuse the latent
down-projection with the MoE router into a single GEMM". vLLM overlaps them on
streams instead. **Report and implementation are not the same artefact** — worth
remembering before quoting either as "how K3 works".

Routing is `fused_grouped_topk` with `e_score_correction_bias` — the §2.3.3
Quantile Balancing bias, frozen at inference. The shared-expert output is
"folded into the up-projection GEMM's beta-add epilogue, so combining the two
branches costs no extra kernel".

## AttnRes and speculative decoding

`block_residual` is carried as `(tokens, num_attn_res_blocks, hidden)` — the
same snapshot stack our engine keeps, materialised as one tensor. The folds are
fused into the surrounding norms (`_pre_attn_norm` / `_post_attn_norm` call
`attn_res(...)` with the norm weights), and aux captures run on a side stream
(`_aux_attn_res_stream`), matching §5.4.2.

`KimiLinearModel` inherits `EagleModelMixin`; there is `mtp.py` on both backends,
`get_spec_layer_idx_from_weight_name`, and `ops/recoverssm.py` (the ReplaySSM
rollback). A comment in the model forward records the coupling:

> "the final norm is applied in `compute_logits` instead of here, so the MTP
> draft model receives the pre-norm hidden states."

**None of this is usable with the open weights** — the released index has no
`nextn`/`mtp`/`draft`/`eagle` tensors (checked). The machinery ships; the
draft weights do not.

## What this says about our architecture proposal

vLLM runs Kimi K3 as **pipeline-parallel slices of layers, each rank holding its
own layers' weights, passing only activations between ranks.** That is the chain
in [AI-PROCESSING-UNIT.md](../../../clover-intelegence/code/standalone-hosting/AI-PROCESSING-UNIT.md),
at coarser granularity — several layers per device instead of one.

So the architecture is not speculative; it is how the model is served today. Our
contribution is narrower than first written: **one layer per unit, with the unit
sized to the workload's balance point**, rather than many layers per
general-purpose GPU. The pipeline half of the claim is settled by existing
practice. The bandwidth-provisioning half is still the open question.

## From the launch blog (published figures, **not reproduced by us**)

[vllm.ai/blog/2026-07-27-k3](https://vllm.ai/blog/2026-07-27-k3). These are the
vLLM team's numbers on their hardware. Quote them as theirs.

**Scale floor:** "The entire model can barely fit in a single NVIDIA DGX B300
and requires a minimum of 16 NVIDIA B200/GB200 GPUs to serve on that hardware
generation."

**Decode throughput, batch size 1, GB300 NVL72:**

| | TP8 | TP16 |
|---|---|---|
| no speculation | 111 tok/s | **118 tok/s** |
| with DSpark | 331 tok/s | **370 tok/s** (~3.14x) |

### A draft model now exists — this corrects us

We checked the Moonshot weight index, found no MTP/draft tensors, and concluded
speculative decoding was out of reach. That was right about *Moonshot's* release
and wrong about the world: **Inferact have trained and open-sourced
`Inferact/Kimi-K3-DSpark`**, a block-diffusion speculator, used with
`num_speculative_tokens: 7`.

Published acceptance: **4.73 accepted tokens per step** on low-entropy tasks
(coding), **2.61** on high-entropy (creative writing).

Our own measured break-even is **3.1 of 8**. Their low-entropy figure clears it;
their high-entropy figure does not. So the lever is real, and the open item
should read "integrate a draft model" rather than "none exists".

### At batch 1 they bypass the Tensor Cores — corroboration again

Their low-latency BF16 `skinnyGEMM`:

> "Generic cuBLAS kernels do not achieve the best performance here because they
> are optimized for more general shapes. In the kernel, we bypass shared-memory
> data staging, load activations and weights directly into registers, and **use
> CUDA Core FMA instructions** to perform the math. **This avoids the heavy TMA
> and Tensor Core setup phase** used to achieve maximum throughput."

8-100% kernel speedup, ~10% end-to-end in small-batch settings. **At batch 1 the
tensor cores are the wrong instrument and they route around them** — the same
conclusion our FLOP/byte measurement reached, reached independently, on hardware
three generations newer.

### Other optimisations with figures

| | effect |
|---|---|
| custom reduce-scatter / all-gather kernels | 1.7-4.5x faster than NCCL at small-to-medium messages |
| LatentMoE tail fusion | ~20% on that step, 7-8% end-to-end |
| KDA metadata builder | 870 us -> 34 us at batch 1 (96%), 6% end-to-end |
| Decode Context Parallelism (roadmap) | ~40% higher throughput than TP8 in early experiments |

Cache retention has two policies: **interval-based** (checkpoint every N tokens,
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, prompt ends always kept) and
**Marconi-style selective retention** — "cache on the second hit", so one-off
prefixes never consume capacity.

### Calibration against our own projection

118 tok/s is **8.47 ms per token**, i.e. **~0.091 ms per layer** across 93
layers. My chain sizing in `AI-PROCESSING-UNIT.md` targeted 1.08 ms per layer.
**Production is already ~12x faster per layer than the unit I proposed.**

That is worth sitting with. At 16 GPUs with HBM3e and the layer sharded 16 ways,
each rank reads ~54 MB of a layer, so per-rank bandwidth stops being the binding
term and collectives and launch overhead take over — which is exactly why their
optimisation list is about fusing kernels and beating NCCL rather than about
streaming weights faster. **Our bandwidth-bound analysis describes the
single-box regime, not theirs.**

## From the preview post (2026-07-22) — the cache design

The hard problem was prefix caching, and the fix was conceptual before it was
technical. vLLM **separated three things that used to move together**:

| | |
|---|---|
| physical block size | how KDA state and full-attention KV are allocated on the GPU |
| scheduler alignment | where execution must stop so all cache groups stay consistent |
| prefix-match unit | the finer interval at which a shared prefix is hashed and matched |

Before that, the physical block size also constrained where a hit could land, so
"two requests sharing almost the entire prompt could still miss the reusable
prefix because their common boundary did not fill the same physical block".
Partial hits now use **copy-on-write** and **chained fine-grained hashes** so a
boundary certifies the whole prefix, not just its tail.

Also noted there: Kimi K3's MLA has "a gate projection that can execute in
parallel with the main attention path" — multi-stream in decode, fused into the
gate-projection epilogue in prefill. And on depth: "a small per-layer launch or
memory penalty quickly becomes a large TPOT penalty" across 93 layers.

## From the optimisation post (2026-09-13) — 2.8x, and where it came from

**Published, not reproduced.** 8K/1K, TP8, 8-token DSpark, B300, v0.27.1 to
commit `82a85dc1`:

| concurrency | latency s | throughput tok/s | TTFT ms |
|---|---|---|---|
| 1 | 12.37 -> **5.30** (-57.2%) | 83.3 -> **183.3** (+120%) | 2262.9 -> **376.3** (-83.4%) |
| 4 | 23.67 -> 10.50 (-55.6%) | 166.7 -> 416.7 (+150%) | 2314.9 -> 640.5 (-72.3%) |
| 16 | 55.90 -> 22.17 (-60.3%) | 258.3 -> 725.0 (+180.6%) | 7601.1 -> 1121.0 (-85.3%) |

**None of the 2.8x came from more parallelism or faster matmuls.** Their own
framing: "scheduler limits and small tensor copies could matter as much as a
large GEMM."

| change | effect |
|---|---|
| adaptive scheduling budget (stop splitting one request across forwards) | TTFT -55-65%, throughput +41.5% |
| internal KDA prefix checkpoints — export the checkpoint *inside* one prefill pass instead of splitting into two model forwards | TTFT -9-25%; "avoids a second full-model pass through attention, MoE, routing, and TP collectives" |
| zero-copy mixed KDA batches — removed 6 `index_select` + 2 `index_copy_` per layer | +5.2-7.7% at concurrency 4/16, **flat at batch 1** |
| deferred MXFP4 finalization fused into the latent tail | ~5% end-to-end |
| **ReplaySSM** — buffer SSM inputs, reconstruct accepted state at commit; rollback moves a pointer | +10.97% effective cache capacity at the same 46.48 GiB |
| **decode context parallelism** — shard MLA latent KV along the sequence | on 120k tokens: KV capacity 1.93M -> 19.75M; TPOT p50 13.8 -> 10.5 ms |

That list is almost entirely **scheduling, memory layout, kernel fusion and
avoiding redundant passes.** It is worth holding next to our own history, where
the wins were the same kind of thing: removing a blocked `Qm` path that re-read
weights, pre-faulting an allocation, partitioning a cache per layer rather than
globally.

### Updated calibration

At concurrency 1 they now reach 183.3 tok/s on 8 B300 — **~5.5 ms per token,
~0.059 ms per layer**. Our engine is 3.07 s per token. The gap is ~560x, on
eight datacentre GPUs against one desktop part, so it is a statement about
hardware and engineering investment rather than about either design.

## What I did not read

- `kda.py`, `mla.py`, `dspark_mla.py` internals; the `cute_dsl` CUTLASS kernels;
  the `.cu` sources; the entire `amd/` backend beyond its file list.
- `chunk.py` / `chunk_intra_token_parallel.py` bodies. I have read the *call
  sites* and the dispatch rule, not the kernels.
- The linked PRs behind each optimisation, and the tracking issue #50587.
- **No vLLM run, no profile, no number of our own.** Source reading plus
  published figures; nothing here is measured by us.
