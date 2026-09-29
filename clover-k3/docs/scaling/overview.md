# Scaling overview

**Status: proposed design; distributed behavior is not verified.** [Reading path](README.md). Section numbers are retained for existing references.

## 1. From computation reduction to the data problem

Clover-K3 began with a computation problem.

K3 is a very large model with a complex sequence of transformations. Instead of treating the model as an opaque collection of kernels, Clover reconstructed the computation from its individual stages and expressed those stages as a mathematical model.

The principle is similar to the difference between calculating:

```text
1 + 2 + 3 + ... + n
```

by performing every addition and using:

```text
n(n + 1) / 2
```

as its mathematical equivalent.

K3 is vastly more complicated, but the principle is the same.

Its computation is a composition of well-defined transformations: embedding, normalization, attention, recurrent state updates, routing, expert computation, residual updates, and the final projection to logits. Each stage has a mathematical definition, and the stages can therefore be composed into one model equation.

The full derivation is documented in [`k3-analysis/k3-model-equation.md`](../../../research/k3/model/k3-model-equation.md). That document does not describe the model only conceptually. It defines the inputs, constants, weights, primitive operators, KDA and MLA attention paths, dense and MoE paths, carried state, the layer composition, and the final token-selection equation.

At the highest level, the computation becomes:

```text
input tokens
    ↓
embedding
    ↓
Layer₀ → Layer₁ → ... → Layer₉₂
    ↓
final aggregation → normalization → lm_head → argmax
    ↓
output token
```

with the important difference that each layer is itself a precise mathematical function with state:

```text
(r_next, state_next) = Layer_L(r, state)
```

and the whole model is the composition of those functions:

```text
Model = Tail ∘ Layer₉₂ ∘ ... ∘ Layer₁ ∘ Layer₀ ∘ Embedding
```

This is what solves the **computation description problem**. Instead of needing to reason about the model as an opaque program made from thousands of individual operations, the complete forward pass can be expressed as one mathematical object whose inputs, transformations, state and output are explicit.

The equation was then executed against the reference implementation. It reproduced the verified computation bit-for-bit for the tested paths: 79,742,816 checked float values were identical for the prefill path, and the prefill-plus-decode comparison reached 96,587,584 identical values, with zero mismatches and maximum ULP difference of zero.

So the important result is not that the equation magically removes arithmetic.

It is that the computation has been **made explicit, reproducible and mathematically tractable**. That makes it possible to ask a deeper question: which computation is actually required, which representations are sufficient, and which work can be reduced or eliminated without changing the result?

That distinction matters for the rest of Clover-K3.

The equation is the foundation from which the computation can be analyzed and reduced; the scaling architecture described in this document addresses the separate problem that remains after the computation has been understood: **where the required model data should live and how it should move**.

This changed the question.

The problem was no longer simply:

> How much computation does K3 contain?

The computation could now be described, decomposed, measured and investigated mathematically.

The next problem became:

> **If the computation is understood, where does the cost actually come from when the model runs?**

The measurements exposed a different constraint.

The model contains roughly 1.5 TB of inference weights, but the computation does not require all of those weights simultaneously. The model moves through layers, and each layer has its own trunk and expert data. A single prompt therefore causes substantial movement of model data even though only a fraction of the total model is required at any individual point.

The computation had been made explicit.

The next problem was therefore **data movement**.

That distinction is the foundation of the scaling architecture described in this document.

The goal is not to claim that the mathematical equation makes K3 require less arithmetic.

The goal is to understand the model well enough to determine **what must remain stationary, what must move, and where each part of the computation should physically live**.

---

## 2. The original problem

Once the computation was expressed as an equation, the next question was not how to make the model choose fewer weights. The measurements showed that the model was already doing that.

The original problem was more fundamental:

> **Why does a model that processes one layer at a time need to keep and move a model-sized data set as though the whole model were one computational unit?**

K3 contains a very large expert pool. Across its 92 MoE layers there are **82,432 experts**, occupying roughly **1.45 TB** of expert weights.

But a single five-token prefill does not need that entire pool.

The router selects 16 experts at each MoE layer. Across the 92 layers and five positions, that produces 7,360 expert draws. After the model's existing per-layer reuse is accounted for, those draws correspond to **5,683 distinct (layer, expert) pairs**, or **99.72 GB of distinct expert data**.

So the relationship is approximately:

```text
entire expert pool        82,432 experts    ≈ 1.45 TB
one 5-token prefill        5,683 experts    ≈ 99.72 GB
                                             ≈ 6.89% of the pool
```

This is important because it rules out the simplest explanation.

The problem is **not** that the existing implementation blindly reads the entire 1.45 TB model for every request. It already follows the router and fetches the experts that the computation actually selects. Within a layer it also avoids refetching the same expert when multiple positions request it.

The measured data problem is therefore different:

> **The model's useful computation is local to a layer, but the model's storage is organized as one enormous global pool.**

That creates a mismatch between the **logical flow of the model** and the **physical location of its data**.

The equation makes the logical flow explicit:

```text
Layer 0 → Layer 1 → Layer 2 → ... → Layer 92
```

At each layer, the computation needs that layer's trunk and the experts selected for that layer. It does not simultaneously need the expert weights belonging to the other 91 MoE layers.

Yet the conventional single-machine arrangement places those weights together and makes the inference system repeatedly retrieve the required expert data from that large shared storage system.

That is the problem this scaling work addresses.

The evidence for this conclusion is recorded separately in:

- [k3-analysis/k3-data-problem.md](../../../research/k3/experiments/k3-data-problem.md), which establishes the amount of distinct data actually required and separates genuine model requirements from implementation waste.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures the layer-level storage footprint, single-layer access behavior, multi-pod contention and the physical consequences of separating the layers.

Those measurements lead to the architectural question for the next section:

> **If the model itself is organized as a sequence of layers, should the infrastructure be organized around those same layers rather than around the total size of the model?**

That is the point where scaling stops being a question of buying a machine large enough to hold K3 and becomes a question of **placing each part of the model where the model actually uses it**.
