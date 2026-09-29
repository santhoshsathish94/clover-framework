# Clover-K3 Scaling

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

The full derivation is documented in [`k3-analysis/k3-model-equation.md`](../k3-analysis/k3-model-equation.md). That document does not describe the model only conceptually. It defines the inputs, constants, weights, primitive operators, KDA and MLA attention paths, dense and MoE paths, carried state, the layer composition, and the final token-selection equation.

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

- [k3-analysis/k3-data-problem.md](../k3-analysis/k3-data-problem.md), which establishes the amount of distinct data actually required and separates genuine model requirements from implementation waste.
- [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md), which measures the layer-level storage footprint, single-layer access behavior, multi-pod contention and the physical consequences of separating the layers.

Those measurements lead to the architectural question for the next section:

> **If the model itself is organized as a sequence of layers, should the infrastructure be organized around those same layers rather than around the total size of the model?**

That is the point where scaling stops being a question of buying a machine large enough to hold K3 and becomes a question of **placing each part of the model where the model actually uses it**.

## 3. Why the model flow determines the architecture

The previous section established the mismatch: the model is computed layer by layer, while its weights are physically organized as one large model-wide data set.

The next step is therefore not to invent a distributed architecture first and then fit K3 into it.

It is to follow the model itself.

The reason for this section is to establish the architectural rule:

> **The physical scaling boundary should follow the model's computational boundary.**

The final model equation makes that boundary explicit. K3 is a composition of Layer 0 through Layer 92, with state carried from one layer to the next:

```text
input
  ↓
Layer 0
  ↓
Layer 1
  ↓
Layer 2
  ↓
...
  ↓
Layer 92
  ↓
tail
  ↓
output
```

Each layer consumes the representation produced by the preceding computation and applies its own layer-specific parameters and state transitions. The model does not require the weights of all 93 layers to participate in the same operation at the same time.

That observation determines the first architectural boundary.

### One layer is one computational unit

For scaling purposes, a layer is the natural unit because it has a complete local computation:

- its own trunk weights;
- its own attention path;
- its own layer-specific state;
- its router;
- its 896 experts where applicable;
- its expert computation;
- and its output representation for the next layer.

The layer therefore has a clear input and a clear output.

That makes it possible to place the layer's data and computation together rather than distributing one layer's weights across unrelated infrastructure.

The architecture consequently follows this shape:

```text
server-side model entry
        ↓
Layer 1
        ↓
Layer 2
        ↓
...
        ↓
Layer 92
        ↓
server-side model exit
```

The end user does **not** communicate directly with the layer machines. The client communicates with the server. The server is the inference boundary and coordinates the internal layer pipeline; the layer units perform the token transformation.

This distinction is important because the architecture is not simply "93 servers exposed to clients." It is a server-coordinated model pipeline whose internal stages correspond to the model's actual computational stages.

### Why this follows from the equation

The equation gives us more than the ordering of the layers. It shows that the model carries representations and state between them.

The residual stream changes from layer to layer. Snapshot state is accumulated at defined points. KDA carries recurrent state. ShortConv carries history. MLA carries its KV cache. The MoE router selects experts from the representation available at that layer.

Therefore the useful boundary between two layers is not the model weights themselves.

It is the **runtime representation and state produced by one layer and consumed by the next**.

That leads to a simple physical rule:

```text
weights stay with the layer
runtime state moves between layers
```

The weights are relatively stationary because they belong to a specific layer. The runtime representation is the part that must travel because the next layer needs the result of the previous layer.

This is the fundamental reason the architecture can scale by layers instead of by total model size.

### What the measurements support

This is not only a conceptual decomposition.

The scaling experiments separately measured individual layer stores, their expert access behavior, the memory required by a layer process, contention when multiple layer processes share a machine, and the payload that crosses a layer boundary.

Those measurements are recorded in [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md).

They show that a layer can be represented as an independently measurable unit: its expert store is approximately 15.77 GB, its trunk slice is approximately 423–635 MB depending on layer type, and the runtime boundary consists of residual and snapshot data rather than token IDs or the entire model.

The measurements therefore support the architectural decomposition that the equation already implies.

They do **not** yet establish cluster-wide throughput. That is deliberately a later experiment. At this stage the evidence establishes the shape of the units and the data boundary between them.

### The resulting scaling principle

Traditional model serving often starts with the question:

> **How large must the machine be to hold and execute the whole model?**

Clover-K3 starts with a different question:

> **What resources does each part of the model actually require while it is executing?**

That change is the basis of the scaling architecture.

Once the model is divided according to its own computational flow, the next question becomes concrete:

> **What exactly belongs inside one layer unit, and how large is it?**

That is the purpose of the next section.

