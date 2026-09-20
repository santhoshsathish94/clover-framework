# Fareed Khan's Kimi K3 in C — Clover Technical Explanation

> A systems-level explanation of Fareed Khan's `kimi-k3-in-c` work and why it matters to the Clover AI infrastructure direction.
>
> This document explains the upstream project and its measured claims. It does not claim those results as Clover's own. Where Clover has independently reproduced only parts of the work, those limits are stated explicitly.

## Source

Upstream repository:
https://github.com/FareedKhan-dev/kimi-k3-in-c

Author: Fareed Khan

Upstream license: Apache-2.0 for the repository. Kimi K3 weights remain Moonshot AI's work and are not redistributed here.

This explanation was prepared against the current upstream `main` available on 2026-09-20 and checked against Clover's latest local evidence record.

## 1. The central idea

Kimi K3 is a 2.78-trillion-parameter model. The shipped checkpoint is about 1.56 TB.

The obvious assumption is:

> A 1.56 TB model needs something close to 1.56 TB of fast memory.

Fareed's implementation challenges that assumption by separating:

- how much model data exists,
- how much data must be reachable,
- how much data must be resident at one moment,
- and how much computation is active for a token.

The model is not made smaller.

Instead, the inference runtime changes where the model's bytes live and when they are fetched.

The resulting architecture is approximately:

```
Model checkpoint
    ↓
Indexed tensors
    ↓
Resident working set + local storage
    ↓
Routing / layer execution
    ↓
Exact computation
```

The headline upstream result is a measured peak RSS of about 8.24 GB for full-model inference, with no GPU, while the same implementation can use more memory to reduce storage traffic and improve speed.

The 8 GB figure is therefore a memory-residency result, not a claim that the whole model occupies 8 GB.

## 2. Follow the parameter ledger

Fareed's README presents a useful reduction ladder:

```
~5,560 GB   hypothetical BF16 residency for every parameter
    ↓
~1,560 GB   shipped checkpoint
    ↓
~113.49 GB  resident set after exploiting MoE routing
    ↓
~8.24 GB    measured peak RSS after streaming the dense trunk
```

Each step comes from a different property.

### 2.1 The experts are already low-bit

The routed experts arrive in MXFP4 rather than BF16.

That lowers their storage footprint substantially before the inference engine does anything.

### 2.2 Most expert weights do not need to be resident

K3 has 896 experts in its routed MoE layers, but only 16 are selected per routed layer for a token.

So most expert weights can remain on storage and be loaded only when selected.

Fareed's census reports roughly:

- 82,432 routed experts
- about 1.447 TB of routed expert data
- roughly 93% of the checkpoint represented by routed experts
- about 113.49 GB left in the always-resident set once the routed expert pool is treated as streamable

The key conceptual distinction is:

> The whole model must remain reachable, but the whole model does not have to remain resident.

## 3. Architecture-specific reductions

The implementation exploits properties already present in Kimi K3.

### Reduction 1 — MXFP4 expert weights

Each expert weight is represented with a 4-bit value plus a shared scale for groups of 32 weights.

Fareed avoids first expanding those weights to ordinary floating point.

Instead:

```
packed MXFP4
     ↓
decode during matrix multiplication
     ↓
accumulate result
```

That matters because a packed expert is much smaller than its FP32-expanded form. Fully dequantizing every selected expert would create large temporary memory traffic.

So the quantized representation is not merely a storage format. It remains part of the compute path.

### Reduction 2 — KDA state

Most layers use Kimi Delta Attention (KDA), whose recurrent formulation allows the relevant state to be carried forward rather than growing like a conventional full-history attention representation.

### Reduction 3 — MLA compression

The model also uses Gated Multi-Head Latent Attention (MLA) in selected layers.

The runtime follows the model's actual MLA structure and does not simplify it into a generic attention implementation.

### Reduction 4 — stream the dense trunk

The dense trunk is about 108.81 GB.

Instead of requiring all of it to be resident, Fareed packs the dense layers into a 109 GB `trunk.bin` with known offsets.

A selected layer can then be fetched directly, while a configurable subset of layers remains pinned in RAM.

This turns memory from a fixed minimum into a tunable system parameter.

## 4. Storage is part of the inference engine

The project treats storage as an active part of the compute architecture.

The checkpoint is split into SafeTensors shards. The runtime reads the headers, indexes tensor locations, and retrieves exact byte ranges as needed.

The conceptual pipeline is:

```
tensor name
   ↓
tensor index
   ↓
shard + byte offset
   ↓
aligned direct read
   ↓
kernel
```

Fareed opens the storage paths with `O_DIRECT`, bypassing the page cache entirely. The measurement behind that ran against the usual expectation: **3.2 GB/s cold with `O_DIRECT` against 2.3 GB/s buffered and warm**. The README's own summary is that "that single measurement decided the whole I/O design."

It is not a trunk-specific choice. `O_DIRECT` appears in the trunk streamer (`k3_trunk.c`), the routed-expert cache (`k3_cache.c`) and the SafeTensors reader (`k3_st.c`) alike. The access patterns are why it pays: the engine walks trunk layers in a fixed cyclic order, which the README shows drives an LRU to a hit rate of exactly zero, and routed-expert usage is deliberately flattened by Quantile Balancing. Under both patterns the page cache costs a copy and retains nothing worth keeping.

This leads to an important implication:

> For storage-streamed inference, storage is not passive capacity. Its latency, throughput, access pattern and locality are part of model performance.

The repository explicitly warns that network storage can behave substantially worse than local NVMe for the workload.

## 5. Memory is an allocation problem, not only a capacity problem

Fareed's measurements show that adding memory to an expert cache is not automatically useful.

He compares different memory splits while holding total memory fixed.

The striking observation is that below a threshold, increasing the routed-expert cache changes little because the access pattern does not create enough reusable locality.

By contrast, allocating more memory to the dense trunk removes repeated reads that are guaranteed to occur.

At a fixed 128 GB budget, the upstream measurements report about a 1.69× speed difference between tested allocations, with the faster configuration giving substantially more memory to the trunk.

The broader systems lesson is:

> Measure where the memory is allocated and what work it removes. Do not optimize a proxy metric simply because it sounds like the right metric.

This becomes especially clear in the cache analysis.

A reported raw cache hit can include data that was prefetched from disk moments before. Fareed therefore distinguishes that from a true resident hit that actually avoided storage traffic.

## 6. The cache result is a measurement lesson

The project exposes a subtle but important measurement problem.

A naive metric can say:

```
cache hit rate ≈ 100%
```

while the real resident hit rate is much lower, because the cache was populated by a disk read immediately before the request.

The useful question is not:

> Did the lookup find the item in the cache structure?

It is:

> Did this access avoid the expensive operation we are trying to optimize?

This is a direct Clover-style Outcome lesson:

**measure the causal effect, not merely the convenient counter.**

## 7. Correctness is treated as an invariant

Fareed repeatedly avoids the pattern:

> optimize first, then check whether the output still looks plausible.

The implementation aims for exact reproducibility across execution paths.

Examples include:

- scalar versus AVX2 kernels
- different memory budgets
- incremental versus recomputed decoding
- tokenizer behavior
- model configuration
- packed expert decoding
- tensor indexing

The repository intentionally avoids certain compiler transformations that could alter floating-point rounding, because the goal is bit-level equivalence between paths.

The practical rule is:

> An optimization that changes the output is not automatically an optimization.

## 8. The configuration reader refuses to guess

One of the strongest details in the project is the configuration handling.

The model's `config.json` defines architecture-specific facts such as:

- 93 layers
- 24 MLA layers
- 69 KDA layers
- 896 routed experts
- top-16 routing
- 2 shared experts

The reader rejects missing required fields rather than inserting defaults.

Why?

Because a wrong default can produce a system that:

- starts successfully,
- generates fluent text,
- and is nevertheless executing a different architecture.

That is a dangerous failure mode for AI infrastructure because the output can look reasonable enough to escape notice.

Fareed therefore treats configuration correctness as part of model correctness.

## 9. Tokenization is also a correctness boundary

Kimi K3 uses a byte-level BPE tokenizer.

The project explicitly validates byte-oriented behavior and tests non-ASCII cases rather than assuming that ASCII success is enough.

This matters because a tokenizer can appear perfectly functional while silently diverging on:

- CJK text
- emoji
- accented text
- raw byte sequences

The repository's validation philosophy is consistent:

> Every boundary that can silently change the model receives an explicit test.

## 10. The inference engine is tiny relative to the model

Fareed's runtime is intentionally small.

The README describes a roughly 176 KB inference binary with no PyTorch, no ONNX Runtime, no BLAS and no GPU library.

Its responsibilities are focused:

- model configuration
- tensor indexing
- storage reads
- model kernels
- routing
- caching
- decoding
- verification/reporting

This produces a striking ratio:

```
~176 KB inference engine
        ↓
~1.56 TB model checkpoint
```

The important point is not the exact binary size.

It is that a huge model does not necessarily require a huge serving software stack.

The runtime is small because the engineering is specialized around the model's actual execution structure.

## 11. Presets show the real trade-off

The upstream project exposes several memory operating points for the same model.

The current README lists approximately:

| Preset | Trunk budget | Expert cache | Reported peak |
|---|---:|---:|---:|
| ultra | 2.50 GB | 0.31 GB | ~3 GB planned |
| laptop | 3.00 GB | 1.00 GB | ~8.2 GB |
| desktop | 16.00 GB | 10.00 GB | ~31.9 GB |
| workstation | 60.00 GB | 30.00 GB | ~95.5 GB |
| server | 110.00 GB | 13.00 GB | ~128 GB |
| max | 110.00 GB | 109.00 GB | ~224 GB |

The measurements show that the largest memory configuration is not necessarily the fastest.

The `server` configuration keeps most of the dense trunk resident, while `max` adds a much larger expert cache without a demonstrated proportional benefit.

So memory should be treated as a resource allocation problem.

## 12. Incremental decoding

Generation can either recompute the whole prefix repeatedly or carry forward the relevant state.

Fareed's incremental path preserves the state needed for the next token, including the KV/cache state and KDA state.

The important part is not simply that incremental decoding is faster.

It is that the project checks the incremental path against the reference path.

Thus the optimization is:

```
same model state
    +
less repeated work
    =
faster equivalent execution
```

This is another example of preserving useful state rather than discarding it and recomputing from scratch.

## 13. The project measures the measurement

Fareed does not treat every timing difference as real.

He runs repeated measurements and documents the observed noise.

One reported set of identical runs had enough variation that small timing differences could not be trusted.

The project therefore distinguishes:

- deterministic counts
- byte totals
- exact output equality
- stopwatch measurements

This is valuable because it prevents a benchmark from turning ordinary device variance into a false optimization claim.

The broader lesson is:

> A measurement system needs its own evidence about how noisy its measurements are.

## 14. What the project actually proves

The upstream work establishes a strong engineering result:

A very large MoE model can be represented and executed through a small custom runtime with a much smaller resident working set by exploiting model sparsity, low-bit expert storage, direct storage access, state reuse and dense-layer streaming.

But the project also states important limits.

The README notes:

- no chat template
- greedy decoding only
- no chunked prefill
- no vision implementation
- no quality benchmark
- no broad downstream task evaluation
- very low speed at the smallest memory budgets

Therefore:

> “It fits” and “it is useful for a production workload” are different claims.

The second requires workload-level evidence.

## 15. Clover's independent verification

Clover has independently built the upstream engine and passed its weightless test suite on a local Windows machine.

The latest Clover evidence record reports:

- source build succeeded
- the weightless gate ladder passed
- the published 100,096-request expert-cache trace was reproduced
- no Kimi K3 weights were downloaded
- no Kimi K3 token was generated locally

The local environment had about 15.7 GB RAM and only about 215 GB free disk, so the full checkpoint plus packed trunk could not be run.

This distinction is important:

**Clover has verified parts of the engineering and reproduced the recorded cache analysis, but it has not independently reproduced the full-model inference measurements.**

The current local evidence therefore supports the methodology and some measurements, not the complete upstream performance claim.

## 16. Why this matters to Clover

The most important contribution to Clover is not “use Kimi K3.”

It is the architecture question Fareed's work exposes:

> How much of an AI infrastructure requirement is a property of the model, and how much is a property of how we organize memory, storage, routing and execution around that model?

That question naturally fits Clover's infrastructure direction.

Instead of starting with:

```
big model
   ↓
big GPU cluster
   ↓
centralized serving
```

Clover can investigate:

```
required outcome
   ↓
smallest model capable of the task
   ↓
model routing
   ↓
appropriate execution engine
   ↓
appropriate memory hierarchy
   ↓
local / cloud / hybrid infrastructure
   ↓
measured verified outcome
```

Fareed's project provides evidence that at least one very large model can cross a boundary that parameter-count reasoning makes look impossible.

## 17. The deeper systems principle

The strongest general lesson from the project is:

> **Model size is not the same thing as machine size.**

A model can be huge while its active computation is sparse.

A checkpoint can be huge while its resident working set is small.

A storage format can be compressed while remaining executable without full dequantization.

A cache can be large while providing little useful benefit.

A server can have more RAM while producing no proportional increase in useful work.

Therefore the right engineering question is not:

> “How big is the model?”

It is:

> “What bytes, computation and state are actually needed to produce the required outcome, and where should each one live?”

That is exactly the kind of systems question Clover is designed to investigate.

## 18. What we should test next

For Clover, the next experiment should not begin by renting a generic AI GPU instance.

The better sequence is:

1. Establish the task and measurable success criteria.
2. Run the smallest model that can plausibly perform that task.
3. Compare CPU, GPU and storage-streamed execution where relevant.
4. Hold total memory fixed while changing memory allocation.
5. Measure actual storage behavior, not provider marketing bandwidth.
6. Measure verified task completion, not only tokens per second.
7. Compare cost per verified successful task.
8. Keep successful and failed measurements as Growth inputs for the next architecture decision.

## 19. The first machine, and why the work is not finished when it runs

The machine is an AX102-3-LTD at Hetzner: 128 GiB of RAM, two 1.92 TB NVMe drives, sixteen cores, **no GPU**, about €50 for a day's work. It is deliberately ordinary. A machine nobody would describe as AI infrastructure is the point of the exercise, not a compromise forced on it.

Three things will be measured there, in order of what they are for:

1. **The workload shape.** Long prompts and 64-token generations, speculative decode, and a second turn resumed from saved state. Both upstream harnesses run a five-token prompt at eight tokens, which is the operating point least like real work. Nobody has measured the shape real work has.
2. **Thread scaling.** `OMP_NUM_THREADS` has never been swept on this engine. Sixteen cores against the reference machine's 124 is the open question, and it costs one loop.
3. **The published campaign, replicated.** Three repetitions per point against a measured 33 percent noise floor, which is upstream `ROADMAP.md` item 2.

A good deal is already settled without spending anything. The checkpoint is public and its 96 shards total exactly 1,560,936,091,448 bytes, so the download will verify. The weightless gate ladder passes, the released configuration parses and the tokenizer round-trips byte for byte. The kernel benchmark has produced a compute baseline and two bit-exactness hashes that the rented machine must reproduce.

What the machine cannot settle is worth stating as plainly. Kimi K3 through this engine has no chat template, no chunked prefill and no quality benchmark. It completes text; it does not follow instructions. It is the demonstration that model size and machine size are separable. It is not the working assistant, and the claim that one ordinary box is enough for most of the work has to be carried by smaller models doing real tasks, measured separately.

**And the work will not be finished when the numbers come back.** An Outcome is where the next cycle starts, not where this one ends. Whatever the machine shows becomes the Context for the following question, and there is always a following question: a better allocation, a faster device, a smaller model that does the same job, a measurement that turns out to have been asking the wrong thing. Reaching what was intended is not the same as running out of things to improve.

There is a particular reason for confidence in that improvement. **The mistakes of today's AI models are not mysterious.** They are observable, repeatable and, when someone bothers to write them down, correctable. The record beside this work lists eleven of them from a single session: figures quoted from a superseded measurement campaign, a gigabyte-versus-gibibyte confusion that produced a confident wrong answer, a prediction extrapolated from an image caption, advice that was precisely backwards about which preset to use. None of those were subtle once they were named. Each was caught because the cycle validates against the system rather than against the model's confidence, and each is now written down where the next cycle will read it.

A failure that is understood is a failure that can be designed out. That is the whole argument for recording them.

## Status

This document describes upstream work by Fareed Khan and the current Clover understanding of it.

It is an explanation and research note, not a claim that Clover created the Kimi K3 runtime or independently reproduced the full-model result.

The current Clover evidence record remains explicit about what has and has not been verified locally.

---

### References

- Fareed Khan, `kimi-k3-in-c`: https://github.com/FareedKhan-dev/kimi-k3-in-c
- Clover Framework: https://github.com/santhoshsathish94/clover-framework
- Clover AI implementation direction: `work-in-progress/clover-ai.md`
- Clover local Kimi evidence: `work-in-progress/kimi-k3-local-evidence.json`
