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

Section 4 defined the resource boundary of a distributed layer unit. This section establishes the beginning of that pipeline and, more importantly, what the model equation makes possible from that beginning.

Layer 0 is different from Layers 1–92 because it is dense. But its importance is not only that it has no expert store.

**Layer 0 is the model-entry transformation from which the composed equation establishes the deterministic computation and routing flow for the remaining layers.**

### Layer 0 is dense, not MoE

The model equation defines Layer 0 as the only dense MLP layer:

```text
Layer 0
├── attention path
├── dense MLP
└── no 896-expert store
```

Layers 1–92 are the MoE layers:

```text
MoE layers = {1, 2, ..., 92}
experts per MoE layer = 896
```

Layer 0 therefore does not have the approximately 15.77 GB expert store that defines the persistent footprint of the distributed MoE units.

This structural distinction comes directly from [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md), which defines the complete layer composition, including the dense Layer 0 path and the routing/expert computation used by Layers 1–92.

### The equation changes what "expert selection" means

The important result from expressing the model as an equation is that expert selection is no longer an opaque runtime event.

For each MoE layer, the equation explicitly defines the router:

```text
x2 at layer L
      ↓
routing scores
      ↓
top-16 expert selection
      ↓
selected expert computation
```

The route is therefore part of the deterministic model function.

For a position (t), the relevant representation is determined by the prompt prefix up to (t). The repository's routing-predictability experiment verified this relationship and then used it to build a routing cache: for a prompt that has already been seen, the expert IDs required by a later layer can be known while the preceding layer is still executing.

This is the key distinction:

```text
NOT: predict which expert might be useful

BUT: derive the exact routing result from the known model inputs,
     cache it, and verify it against the live router
```

The live router remains the correctness authority. The cached route is used to start the next layer's expert reads earlier; a disagreement is treated as a failure rather than silently changing the model result.

The evidence for this is recorded in [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md) and the Clover-K3 implementation. The measured cross-layer lookahead changed the tested warm execution from approximately **8.75 s to 7.25 s**, a **17.0% reduction**, while retaining the exact router check.

This is important to the scaling architecture because the equation does not merely tell us **where** computation occurs. It exposes enough of the dependency structure to determine **when the data required by a later layer can be prepared**.

### Layer 0 belongs to the server-side model entry

The external architecture is therefore:

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

The client does **not** connect directly to Layer 0 or to the distributed layer units.

The server is the inference boundary. It accepts the client request, performs the model-entry work including embedding and Layer 0, and then coordinates the internal layer pipeline.

The distributed layer units are therefore an internal AI-fabric implementation detail of the server-side inference system.

### Why Layer 0 does not need its own distributed expert pod

There are two direct reasons.

First, Layer 0 has no MoE expert pool. There is no 896-expert store whose physical placement motivates a dedicated distributed expert-storage boundary.

Second, Layer 0 is the first complete transformation after embedding. Its output establishes the runtime state from which the composed model computation proceeds into the MoE layers.

Conceptually:

```text
token IDs
   ↓
embedding
   ↓
Layer 0
   ↓
runtime representation + state
   ↓
Layer 1 → ... → Layer 92
```

Keeping this stage at the server boundary avoids introducing an unnecessary distributed boundary before the MoE portion of the model begins.

### What the server sends into Layer 1

The server does not send the entire model or simply forward raw token IDs to Layer 1.

It sends the runtime representation and state required by the next stage.

The equation defines the carried state, including:

- the residual representation;
- snapshot state where applicable;
- recurrent KDA state;
- convolution history;
- and attention cache state where applicable.

The first distributed boundary therefore occurs after the server has completed the model-entry computation and has a valid state for Layer 1 to consume.

```text
server-side model entry
        │
        │ runtime representation + required state
        ▼
     Layer 1
```

The layer units then repeat the same mathematical pattern: consume the state produced by the previous stage, apply the layer's locally owned computation, and produce the state required by the next stage.

### Why this matters for scaling

This gives the architecture two different kinds of boundaries.

The first is the **computational boundary**:

```text
Layer 0 → Layer 1 → ... → Layer 92
```

The second is the **data-preparation opportunity exposed by the equation**:

```text
known prompt/context
       ↓
deterministic routing information
       ↓
prepare later-layer expert data
       ↓
layer computation continues
```

The first boundary tells us where computation should live.

The second tells us when the data for that computation can begin moving.

Together they are what allow the infrastructure to follow the actual dependency graph of the model instead of treating the entire model as one indivisible machine image.

### What the measurements support

The purpose of this section is not to claim a final hardware specification or a universal throughput improvement.

The evidence establishes three concrete facts:

1. [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md) defines Layer 0 separately from the 92 MoE layers and makes routing an explicit part of the mathematical computation.
2. The routing-predictability work establishes that, for a known prompt prefix, later-layer routing can be derived ahead of the layer that consumes it and checked against the live router.
3. The measured lookahead experiment reduced the tested warm execution from 8.75 s to 7.25 s while preserving the exact routing check.

Those results establish the **dependency and data-preparation structure**. They do not by themselves establish cluster-wide throughput or the final production hardware configuration.

### The resulting boundary

The architecture now has a precise beginning:

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
  ├── derive/prepare deterministic routing information
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

> **The client communicates with the server. The server owns model entry and exit. The composed equation determines the model's layer and routing dependencies, allowing later-layer data to be prepared when those dependencies are already known. The distributed layer units own Layers 1–92.**

The next section can therefore move into the distributed portion itself:

> **What does one of Pods 1–92 actually execute when it owns one complete K3 layer?**

## 6. Pods 1–92: one layer, trunk + experts

Section 5 established why the model entry is different and how the equation exposes deterministic routing dependencies. This section defines the actual distributed unit: **one complete K3 MoE layer**.

The reason for writing it is simple:

> **If scaling follows the model's computational boundary, what exactly must one Layer 1–92 unit own so that it can execute its layer independently?**

The answer is the complete layer computation and the data required to perform it.

### One pod owns one layer

For every (L in {1,dots,92}), the distributed unit is:

```text
Pod L
├── Layer L trunk
├── Layer L attention path
├── Layer L router
├── Layer L: 896 experts
├── Layer L local execution state
└── working memory
```

The important point is **ownership**.

The pod does not own a fraction of the whole model.

It owns one complete layer:

```text
Layer L
   ↓
trunk + router + 896 experts + execution state
```

That means a request reaching Pod 37 does not require Pod 37 to know or hold the weights of Layers 0–36 or 38–92.

The only model information it needs from outside its boundary is the runtime state produced by the preceding stage and the control information required to execute its own layer.

### The trunk stays with the layer

The trunk contains the layer-specific attention and projection weights that are required regardless of which experts are selected.

The measured trunk slices are approximately:

```text
MLA layer: ~423 MB
KDA layer: ~635 MB
```

The exact footprint varies because K3 does not use one identical attention path in every layer. The model equation defines the layer-specific KDA and MLA branches, while the scaling measurements establish the corresponding physical trunk sizes.

Therefore the trunk is not a shared global resource.

It belongs to the layer that uses it and remains local to that pod.

### The router stays with the layer

The router is also part of the layer's computation.

For a MoE layer, the equation defines routing from the layer representation:

```text
x2
 ↓
router scores
 ↓
top-16 selection
 ↓
expert weights
 ↓
selected experts
```

The router therefore belongs to the same computational unit as the experts it selects.

This does not mean the router has to delay all data movement until the last possible moment. Section 5 established the opposite: once the deterministic routing dependency is known for a previously seen prompt, the system can prepare the next layer's expert reads ahead of execution while retaining the live router as the correctness check.

The distinction is:

```text
router = correctness authority
routing information = data-preparation input
```

This allows the storage system to begin moving expert data before the arithmetic reaches that layer, without changing which experts the model actually selects.

### The 896 experts remain layer-local

Each MoE layer has exactly 896 experts.

The measured packed expert size is approximately **17.55 MB per expert**, giving:

```text
896 × 17.55 MB ≈ 15.72 GB
```

The measured on-disk layer store is approximately **15.77 GB** including its storage structure and associated data.

The architecture therefore changes the physical organization from:

```text
82,432 experts
      │
      ▼
one global expert pool
```

to:

```text
Layer 1  → 896 experts
Layer 2  → 896 experts
...
Layer 92 → 896 experts
```

No expert is required to be duplicated across all layer units merely because the model contains many layers.

The expert weights stay where their computation lives.

### Experts are stored, not necessarily resident

The 15.77 GB figure is a **persistent layer-store size**, not a requirement that 15.77 GB of private RAM be allocated to every pod.

The current scaling measurements use a per-layer SQLite store and read expert blobs as required.

The measured design boundary is therefore:

| Resource | Per layer |
|---|---:|
| Expert store on local storage | ~15.77 GB |
| Trunk resident footprint | ~423–635 MB |
| KDA state where applicable | ~6.29 MB |
| Additional execution buffers | workload-dependent |
| Full expert store required in private RAM | **No** |

The scaling experiment measured approximately **1.1 GB of expert blobs per 64-expert stage**. Cold access was measured separately from warm page-cache behavior. This distinction matters because page cache depends on the memory limit and workload; warm-cache bandwidth is not a guaranteed pod specification.

The evidence and caveats are recorded in [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md).

### What crosses into a pod

A pod does not receive the whole prompt as its layer input.

It receives the runtime representation and state required by its layer.

The payload includes the residual and accumulated snapshots. As depth increases, the number of snapshot vectors can increase:

```text
L1  → 2 vectors      57,344 B / position
L25 → 4 vectors     114,688 B / position
L48 → 5 vectors     143,360 B / position
L85 → 9 vectors     258,048 B / position
```

For a five-position prefill, the measured total traffic across all 92 layer boundaries is approximately **70.53 MB**.

This is important because the inter-layer interface is a **runtime-state interface**, not a model-weight interface.

The large persistent expert stores stay with their layer. The relatively small runtime state moves between layers.

### What a pod actually executes

A pod is not merely a storage server.

For its layer (L), it performs the layer function defined by the equation:

```text
input state
    ↓
attention / recurrent state update
    ↓
snapshot aggregation where applicable
    ↓
input normalization
    ↓
router
    ↓
selected expert computation
    ↓
residual update
    ↓
output normalization / projection
    ↓
next-layer state
```

The exact attention branch depends on the layer:

- KDA layers execute the recurrent delta-rule path and carry KDA state.
- MLA layers execute the latent/rope attention path and carry the corresponding cache state.
- MoE layers execute the router, selected expert computation and residual update.
- Snapshot-push layers additionally update the accumulated snapshot state.

The equation is what makes this decomposition precise: it defines the operators, ordering, state transitions and layer-specific parameters instead of treating a pod as an arbitrary collection of kernels.

### Why this unit is useful for scaling

Once one pod owns one complete layer, scaling no longer means replicating the full model.

It becomes possible to reason independently about:

```text
Layer 37
├── storage
├── memory
├── CPU/GPU resources
└── replicas
```

without automatically replicating Layers 0–36 and 38–92.

That is the physical consequence of following the model equation.

It does **not** yet prove that every layer should receive identical hardware, nor that a particular number of pods gives a particular throughput. Those are deployment measurements that come later.

### Evidence behind this section

This section is written to define the concrete ownership boundary of Pods 1–92.

The evidence comes from:

- [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md), which defines the layer-specific attention paths, routing, MoE computation, state and weights.
- [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md), which measures the 15.77 GB layer store, 423–635 MB trunk range, runtime-memory behavior, expert-read behavior and inter-layer payload.
- The verified Clover-K3 implementation, which keeps the live router as the correctness authority while allowing deterministic routing information to be used for earlier expert reads.

These establish the **resource and data boundary** of a layer unit. They do not yet establish the final production hardware specification.

The next section can therefore address the next architectural question:

> **Once each pod owns one layer, what exactly moves from one layer to the next, and how can that movement be overlapped with computation?**

## 7. What moves between layers

Section 6 established that each distributed unit owns one complete layer. The next question is therefore the interface between those units:

> **If the weights stay with each layer, what data actually has to move from Layer (L) to Layer (L+1)?**

This section is written to make that boundary measurable.

The answer is not the model weights, and it is not simply the original token IDs. The next layer needs the **runtime representation and carried state produced by the previous computation**.

### The layer boundary is a state boundary

At the end of a layer, the result is not a new copy of the model.

It is a relatively small runtime state:

```text
Layer L
   │
   ├── residual representation
   ├── accumulated snapshots
   ├── recurrent state where applicable
   ├── convolution history where applicable
   └── attention cache state where applicable
   │
   ▼
Layer L+1
```

The exact contents depend on the model path and on the position being processed.

The important architectural distinction is:

```text
persistent model data      → stays with its layer
runtime computation state  → crosses the layer boundary
```

This is the boundary that makes layer-level distribution possible.

### The residual is only part of the payload

It would be tempting to describe the interface as one 7168-element residual vector.

That is incomplete for K3.

The implementation shows that each layer consumes the current residual **plus the accumulated snapshot vectors**. Snapshots are pushed at defined layers, so the amount of state carried forward grows with depth.

The measured payload therefore changes across the pipeline:

| Layer boundary | Vectors / position | Payload / position |
|---|---:|---:|
| Layer 1 | 2 | 57,344 B |
| Layer 25 | 4 | 114,688 B |
| Layer 48 | 5 | 143,360 B |
| Layer 85 | 9 | 258,048 B |

The growth is caused by the model's state structure, not by replication of model weights.

### Why snapshots matter

The equation defines snapshot aggregation as part of the layer computation.

When a snapshot-push layer is reached, the current representation is added to the carried snapshot set. Later layers can consume those snapshots together with the live residual.

Conceptually:

```text
residual
   │
   ├── continues forward
   │
   └── snapshot push
          │
          ▼
     accumulated state
          │
          └──────────────→ later layers
```

That means the infrastructure cannot treat the inter-layer message as a fixed-size tensor independent of depth.

The interface is determined by the model equation itself.

### What does not cross the boundary

The following do **not** need to be sent from one layer pod to the next merely to execute the next layer:

- the previous layer's expert weights;
- the next layer's expert weights;
- the entire 1.45 TB expert pool;
- the entire model checkpoint;
- or the original token IDs as the primary layer representation.

Each pod already owns the persistent data required for its own computation.

The next pod receives the state required to continue the mathematical composition.

This is the central data-locality property of the architecture.

### Measured traffic across the full pipeline

For the measured K3 state representation, the cumulative payload across all 92 layer boundaries is:

| Positions processed | Total across all 92 hops | Mean per hop |
|---|---:|---:|
| 1 | 14.11 MB | 0.15 MB |
| 5 | 70.53 MB | 0.77 MB |
| 16 | 225.71 MB | 2.45 MB |
| 64 | 902.82 MB | 9.81 MB |

These numbers describe **runtime-state movement**, not model-weight movement.

That distinction is critical.

The model may contain roughly 1.45 TB of expert weights, but a five-position prefill moves about 70.53 MB of runtime state across the complete 92-hop pipeline under this measured representation.

The architecture therefore separates two very different quantities:

```text
model capacity:
≈ 1.45 TB expert store

runtime communication:
≈ 70.53 MB for the measured 5-position pipeline
```

The large number stays distributed.

The smaller number moves.

### Why placement matters

The measured numbers also expose an important deployment constraint.

For a single position, the cumulative state crossing all layer boundaries is only about 14.11 MB. For 64 positions it grows to about 902.82 MB.

At small batches, this makes the runtime-state interface relatively small compared with the persistent expert data.

At larger batches, however, the communication becomes significant enough that placement and fabric bandwidth matter.

The scaling architecture therefore should not blindly distribute every layer across arbitrary distant machines.

The physical placement should preserve the model's pipeline while keeping the runtime-state path efficient.

The existing measurements found that at 64 positions, approximately 10 MB crosses an average hop. Across nodes connected at 10 Gb/s, that can represent roughly **0.7 seconds of communication for the complete prompt**, making co-scheduling and network topology a real deployment consideration.

This is not a claim that 10 Gb/s is the final required fabric. It is a measured example showing where communication can become a constraint as the workload grows.

### The equation determines the interface

This is the deeper reason for writing the section.

Without the equation, it is easy to think about distributed inference in terms of arbitrary tensor messages.

With the equation, the interface can be derived from the actual state transition:

```text
(r_next, state_next)
       =
Layer_L(r, state)
```

Therefore:

```text
input to Layer L+1
      =
output state of Layer L
```

The interface is not an infrastructure convention invented independently of the model.

It is the mathematical boundary between two composed functions.

That makes it possible to measure the payload, reason about its scaling with sequence length and snapshot depth, and choose physical placement based on actual communication requirements.

### Evidence behind this section

The evidence comes from:

- [k3-analysis/k3-model-equation.md](../k3-analysis/k3-model-equation.md), which defines the carried residual, snapshots, KDA state, convolution history and attention cache state.
- [k3-analysis/clover-scaling-architecture.md](../k3-analysis/clover-scaling-architecture.md), which measures the layer-boundary payload at multiple depths and sequence sizes.
- The verified Clover-K3 implementation, which shows that the layer input contains the residual and accumulated snapshots rather than simply a token-ID stream.

These sources establish the current runtime-state boundary and its measured size.

They do not yet establish the final inter-node fabric, compression scheme, serialization format or production network topology. Those remain deployment questions.

The next section can therefore address the key concurrency question:

> **If a token must pass through Layers 1–92 in order, how can multiple layers remain active without forcing the entire pipeline to wait on one token at a time?**
