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

## 4. What belongs inside one layer unit

Section 3 established that the scaling boundary should follow the model's layer boundary. This section answers the practical question that follows: **what must actually be placed inside that boundary?**

The answer is not simply "the layer's weights."

A layer unit must contain everything required to transform the representation it receives into the representation required by the next layer, without requiring the entire model to be present locally.

For K3, that means the unit contains:

```text
Layer N
├── layer trunk
├── attention computation
├── router
├── 896 experts          (MoE layers 1–92)
├── layer-local state
└── working memory
        │
        ↓
   transformed state
```

Layer 0 is different. It is dense rather than MoE, so it does not need the 896-expert store. It is kept on the server-side model entry rather than treated as one of the distributed MoE layer units.

### The expert store is layer-local

Each MoE layer has exactly 896 experts. The measured packed expert size is approximately **17.55 MB per expert**, giving a layer-level expert store of approximately:

```text
896 × 17.55 MB ≈ 15.72 GB
```

The measured on-disk layer store is approximately **15.77 GB** once the store structure and associated data are included.

This is the key physical transformation from the original model-wide arrangement:

```text
one global expert pool
        ↓
92 independent layer stores
```

The total amount of expert data has not been made to disappear. It has been **partitioned according to ownership**.

Each layer unit owns the data that only that layer can use.

### The trunk is small relative to the expert store

The expert store is not the whole layer.

The layer also needs its non-expert trunk and working state. The measured trunk slices range from approximately **423 MB to 635 MB**, depending on the layer.

That means the persistent layer footprint is dominated by its experts, while the computation itself requires only a relatively small local trunk plus runtime buffers.

This distinction matters because it means the layer unit does not require a machine sized for the entire 1.5 TB model.

Its storage requirement is approximately the size of **one layer**, not the size of K3.

### Runtime memory is different from persistent storage

The next distinction is between what must be stored and what must be resident.

The layer's expert store can remain on local storage and be read as required. The layer process needs resident memory for:

- its trunk slice;
- residual and snapshot working buffers;
- MoE intermediates;
- KDA recurrent state where applicable;
- convolution history;
- MLA cache state where applicable;
- and other execution buffers.

The measured architecture work therefore does not treat the full 15.77 GB expert store as mandatory private process memory.

The relevant measurements are recorded in [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md), including the single-layer store measurements, cold versus warm access behavior and the runtime working-set observations.

This distinction is important for efficiency.

A machine does not need 15.77 GB of private RAM merely because the layer owns 15.77 GB of expert weights. Storage capacity and resident execution memory are different resources and should be sized separately.

### Why the layer is independently scalable

Once the layer owns its own persistent data and computation, it becomes an independently addressable scaling unit.

For example:

```text
Layer 37
   ├── its trunk
   ├── its 896 experts
   └── its runtime state
```

can be replicated without replicating the other 91 layers.

That is a fundamentally different scaling primitive from adding another complete copy of the 1.5 TB model.

The architecture therefore makes the following resources independently placeable:

- layer storage;
- layer CPU/GPU resources;
- layer memory;
- layer replicas.

The exact resource allocation still has to be validated with the real layer implementation. The current measurements establish the storage and access shape; they are not a claim that the final production pod specification has already been proven.

### Evidence behind this decomposition

The reason for writing this section is to turn the abstract "one layer is one unit" rule into a concrete resource boundary.

The evidence comes from two independent parts of the repository:

- [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md) defines which operations, weights and state belong to each layer.
- [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md) measures the resulting layer stores, trunk sizes, working memory behavior and access characteristics.

Together they establish what can be owned by a layer unit today.

They do not yet establish the final hardware configuration for every deployment. That remains an implementation and deployment measurement.

The next section can therefore address the special case at the beginning of the pipeline: **why Layer 0 belongs on the server-side model entry rather than being treated as another distributed MoE pod.**

## 5. Server-side model entry: why Layer 0 is different

The previous section defined the contents of a distributed layer unit. This section handles the one deliberate exception at the beginning of the model: **Layer 0**.

The reason for writing this section is to make the pipeline boundary precise. Layer 0 is part of the model computation, but it is not equivalent to Layers 1–92 because its computation and data shape are different.

### Layer 0 is dense, not MoE

The model equation defines Layer 0 as the only dense MLP layer:

```text
Layer 0
├── attention path
├── dense MLP
└── no 896-expert store
```

Layers 1–92 belong to the MoE set:

```text
MoE layers = {1, 2, ..., 92}
experts per MoE layer = 896
```

Layer 0 therefore does not have the approximately 15.77 GB layer-local expert store that defines the persistent footprint of the distributed MoE units.

This is not an arbitrary placement decision. It follows directly from the model definition in [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md), where the dense MLP is explicitly defined for layer 0 and the MoE computation is defined only for layers 1 through 92.

### Layer 0 belongs to the server-side model entry

The external architecture is:

```text
client
   │
   │ inference request
   ▼
server-side model entry
   │
   ├── embedding
   ├── Layer 0
   │
   ▼
Layer 1
   ▼
Layer 2
   ▼
...
   ▼
Layer 92
   │
   ▼
server-side model exit
   │
   ▼
client
```

The client therefore does **not** connect directly to a Layer 0 machine or to the distributed layer units.

The server is the inference boundary. It accepts the client request, performs the model-entry work including embedding and Layer 0, and then passes the resulting runtime representation into the internal layer pipeline.

The distributed layer units are an internal implementation detail of the server-side inference system.

This distinction matters because the architecture is not:

```text
client → Pod 0 → Pod 1 → ... → Pod 92
```

where every pod is an externally addressable inference endpoint.

It is:

```text
client → server
             │
             ├── model entry / Layer 0
             │
             └── internal layer pipeline
                    Layer 1 → ... → Layer 92
             │
             └── model exit
             │
             ▼
           client
```

The server therefore owns the external request lifecycle while the internal layer units own the model transformations.

### Why Layer 0 does not need its own distributed expert pod

There are two independent reasons.

First, Layer 0 has no MoE expert pool. There is therefore no large layer-local expert store whose physical placement motivates a dedicated distributed storage boundary.

Second, the model equation makes Layer 0 the first complete transformation after embedding. Its output is simply the runtime representation consumed by Layer 1.

That makes Layer 0 a natural part of the model-entry stage:

```text
token IDs
   ↓
embedding
   ↓
Layer 0
   ↓
runtime representation
   ↓
Layer 1
```

Keeping this stage at the server boundary also avoids introducing an unnecessary network or process boundary before the distributed portion of the model has any reason to begin.

### What the server sends into Layer 1

The server does not send the original token IDs or the entire model state to Layer 1.

It sends the runtime representation and state required by the next layer.

The exact state is determined by the model equation. The model carries:

- the residual representation;
- snapshot state where applicable;
- recurrent KDA state;
- convolution history;
- and the attention cache state where applicable.

The first distributed boundary therefore occurs after the server has completed the model-entry computation and has a valid state for Layer 1 to consume.

Conceptually:

```text
server-side model entry
        │
        │ runtime representation + required state
        ▼
     Layer 1
```

The layer units then repeat this pattern: consume the state produced by the previous stage, apply their own computation using their locally owned weights, and produce the state required by the next stage.

### What the measurements support

The purpose of this section is not to claim that Layer 0 is intrinsically faster or that a particular server specification has already been proven.

The evidence establishes the structural distinction.

[k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md) defines Layer 0 as dense and defines the MoE set as layers 1–92. [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md) separately measures the Layer 0/trunk arrangement and the layer-level storage and access behavior used for the distributed portion.

The scaling measurements also show why the distinction should remain explicit: the large persistent storage footprint is created by the 92 MoE layer stores, while Layer 0 does not carry an equivalent 896-expert store.

What has been established here is therefore the **pipeline ownership boundary**, not a final hardware specification.

### The resulting boundary

The architecture now has a clear beginning:

```text
CLIENT
  │
  │ request
  ▼
SERVER
  │
  ├── embedding
  ├── Layer 0
  │
  ▼
INTERNAL AI FABRIC
  │
  ├── Layer 1
  ├── Layer 2
  ├── ...
  └── Layer 92
  │
  ▼
SERVER
  │
  │ response
  ▼
CLIENT
```

The important rule is:

> **The client communicates with the server. The server owns the model entry and exit. The distributed layer units are internal stages that own Layers 1–92.**

This gives the scaling architecture a clean boundary without pretending that every stage of the model has the same resource requirements.

The next section can therefore move into the distributed portion itself:

> **What does one of Pods 1–92 actually execute when it owns one complete K3 layer?**

