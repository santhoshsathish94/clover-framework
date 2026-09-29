# Scaling architecture contract

**Status: proposed design; distributed behavior is not verified.** [Reading path](README.md). Section numbers are retained for existing references.

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

Those measurements are recorded in [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md).

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

The relevant measurements are recorded in [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), including the single-layer store measurements, cold versus warm access behavior and the runtime working-set observations.

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

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md) defines which operations, weights and state belong to each layer.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md) measures the resulting layer stores, trunk sizes, working memory behavior and access characteristics.

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

This structural distinction comes directly from [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the complete layer composition, including the dense Layer 0 path and the routing/expert computation used by Layers 1–92.

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

The evidence for this is recorded in [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md) and the Clover-K3 implementation. The measured cross-layer lookahead changed the tested warm execution from approximately **8.75 s to 7.25 s**, a **17.0% reduction**, while retaining the exact router check.

This is important to the scaling architecture because the equation does not merely tell us **where** computation occurs. It exposes enough of the dependency structure to determine **when the data required by a later layer can be prepared**.

### Layer 0 belongs to the server-side model entry

The external architecture is therefore:

```text
client
   │
        ├── embedding
        │ embedding vector + request metadata
   ▼
server-side model entry
   │
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
server-side pipeline exit
   │
        │ final representation
   ▼
client
        └── model exit: aggregation, normalization, lm_head, token selection
```

The client does **not** connect directly to Layer 0 or to the distributed layer units.

The server is the layer-pipeline boundary. It accepts the client's embedded representation, executes Layer 0, and coordinates the internal layer pipeline. The client owns embedding and the model exit, including final aggregation, normalization, vocabulary projection and token selection.

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

Keeping Layer 0 at the server boundary avoids a separate Layer 0 pod. The client-to-server boundary carries the embedding output, before the MoE portion of the model begins.

### What the server sends into Layer 1

The server does not send the entire model or simply forward raw token IDs to Layer 1.

It sends the runtime representation and state required by the next stage.

The equation defines the carried state, including:

- the residual representation;
- snapshot state where applicable;
- recurrent KDA state;
- convolution history;
- and attention cache state where applicable.

The first server-to-layer-unit boundary occurs after the server has completed Layer 0 and has a valid state for Layer 1 to consume. The client-to-server boundary precedes it.

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

1. [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md) defines Layer 0 separately from the 92 MoE layers and makes routing an explicit part of the mathematical computation.
2. The routing-predictability work establishes that, for a known prompt prefix, later-layer routing can be derived ahead of the layer that consumes it and checked against the live router.
3. The measured lookahead experiment reduced the tested warm execution from 8.75 s to 7.25 s while preserving the exact routing check.

Those results establish the **dependency and data-preparation structure**. They do not by themselves establish cluster-wide throughput or the final production hardware configuration.

### The resulting boundary

The architecture now has a precise beginning:

```text
CLIENT
  │
        ├── embedding
        │ embedded representation + request metadata
  ▼
SERVER
  │
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
        │ final representation
  ▼
CLIENT
        └── model exit: aggregation, normalization, lm_head, token selection
```

The important rule is:

> **The client owns embedding and model exit and communicates with the server. The server owns Layer 0 and pipeline coordination. The composed equation determines the model's layer and routing dependencies, allowing later-layer data to be prepared when those dependencies are already known. The distributed layer units own Layers 1–92.**

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

The evidence and caveats are recorded in [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md).

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

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the layer-specific attention paths, routing, MoE computation, state and weights.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures the 15.77 GB layer store, 423–635 MB trunk range, runtime-memory behavior, expert-read behavior and inter-layer payload.
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

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the carried residual, snapshots, KDA state, convolution history and attention cache state.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures the layer-boundary payload at multiple depths and sequence sizes.
- The verified Clover-K3 implementation, which shows that the layer input contains the residual and accumulated snapshots rather than simply a token-ID stream.

These sources establish the current runtime-state boundary and its measured size.

They do not yet establish the final inter-node fabric, compression scheme, serialization format or production network topology. Those remain deployment questions.

The next section can therefore address the key concurrency question:

> **If a token must pass through Layers 1–92 in order, how can multiple layers remain active without forcing the entire pipeline to wait on one token at a time?**

## 8. Pipeline concurrency: dependency without global idling

Section 7 established that the boundary between layers carries runtime state rather than model weights. The next question is the natural one:

> **If the mathematical model is sequential, does the infrastructure also have to execute as one globally idle-at-every-step sequence?**

No.

The model imposes a dependency order for each token, but that does not require every layer unit to be idle whenever another layer is working.

### The dependency is per token

For one token position, the mathematical dependency remains strict:

```text
Layer 1
   ↓
Layer 2
   ↓
Layer 3
   ↓
...
   ↓
Layer 92
```

Layer 3 cannot consume the output of Layer 2 before Layer 2 produces it.

That dependency cannot be removed merely by distributing the layers.

But the infrastructure does not have to process only one token at one layer at a time.

Once Token A has moved from Layer 3 to Layer 4, Layer 3 can work on another available token or position.

Conceptually:

```text
             Layer 1   Layer 2   Layer 3   Layer 4   Layer 5

Token A         ●         ●         ●         ●         ●
Token B                   ●         ●         ●         ●
Token C                             ●         ●         ●
Token D                                       ●         ●
```

The diagonal dependency for each token remains intact while the pipeline as a whole becomes concurrently occupied.

This is the basic pipeline principle:

> **Sequential dependency does not imply global sequential utilization.**

### The layers become pipeline stages

With one layer per unit, the infrastructure naturally becomes a pipeline:

```text
server entry
    ↓
Layer 1
    ↓
Layer 2
    ↓
Layer 3
    ↓
...
    ↓
Layer 92
    ↓
server exit
```

The output of one stage becomes the input of the next.

While Layer 20 is processing one piece of work, Layer 19 does not need to remain idle if another piece of work is ready.

The same applies throughout the pipeline.

This is different from attempting to parallelize the mathematical layers themselves.

The layers remain ordered for each dependency chain.

What becomes parallel is the **work occupying different stages of the pipeline**.

### Routing creates a second scheduling opportunity

The routing work adds another important dimension.

The model equation makes the router part of the exact computation. The routing-predictability experiment then established that, for a previously seen prompt, the exact expert IDs required by a later layer can be known before that layer begins its expert computation.

That allows the system to separate two events:

```text
know which data is required
        ↓
start moving that data
        ↓
perform the computation that consumes it
```

The data movement does not have to wait until the last arithmetic operation before it can begin.

In the measured Clover-K3 implementation, the next layer's expert reads are started while the current layer is still performing attention, projection and routing work.

The live router still executes and validates the cached IDs.

So the optimization is not speculative model execution.

It is **earlier preparation of data whose exact identity is already known**.

### The measured lookahead result

This behavior was measured on the Clover-K3 implementation.

The baseline warm run was approximately:

```text
8.75 s
```

With cross-layer expert-read lookahead:

```text
7.25 s
```

The measured reduction was approximately:

```text
1.49 s
17.0%
```

The device utilization moved from approximately 80% to 98–99%, while the measured expert-read rate increased from roughly 11.0 GB/s to 13.5 GB/s.

The profile also showed pipeline stall falling from approximately 2.42 s to 0.027 s.

These are measurements of the tested Clover-K3 serving path on the measured hardware. They are evidence that dependency-aware data preparation can reduce idle time in that implementation; they are not a general claim about all hardware or all workloads.

### Why this matters to the distributed architecture

The result changes how the layer pods should be thought about.

A pod should not be treated as:

```text
receive state
→ wait
→ read experts
→ compute
→ send state
→ wait
```

if some of the required expert data can already be prepared.

Instead, the intended flow is closer to:

```text
receive / identify required work
        ↓
prepare next-layer data
        │
        ├──────────────┐
        ↓              ↓
current-layer       storage reads
computation         for next layer
        │              │
        └──────┬───────┘
               ↓
          next-layer ready
```

The computation dependency remains exact.

The storage dependency is moved earlier where the equation permits it.

This is the important distinction between **parallelizing the model** and **pipelining the infrastructure around the model**.

### Pipeline occupancy

The architecture can therefore be understood as a sequence of independently owned stages with overlapping work:

```text
Time →

Layer 1:  AAAAAA  BBBBBB  CCCCCC
Layer 2:        AAAAAA  BBBBBB  CCCCCC
Layer 3:              AAAAAA  BBBBBB  CCCCCC
Layer 4:                    AAAAAA  BBBBBB
...
```

The exact scheduling policy, batching strategy and number of concurrent requests are deployment questions.

The architectural principle is simpler:

> **A layer waits only for the state it actually depends on, not for unrelated work elsewhere in the model.**

This is what allows the physical system to remain active while preserving the model's mathematical ordering.

### What this does not claim

This section is intentionally limited.

It does not claim:

- that all 92 layers can always be fully utilized;
- that one fixed batch size is optimal;
- that pipeline latency is eliminated;
- that cross-layer communication is free;
- or that the measured 17.0% improvement applies to every deployment.

Those questions require the real distributed implementation and workload measurements.

The result established here is narrower and more useful:

```text
model dependency
      ↓
defines what must wait

equation-derived routing knowledge
      ↓
defines what can begin earlier

pipeline placement
      ↓
allows independent stages to work concurrently
```

### Evidence behind this section

The evidence comes from:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the sequential layer composition and state transitions.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures the routing lookahead, device utilization and pipeline-stall reduction.
- The verified Clover-K3 implementation, which uses cached routing IDs only for earlier data reads and checks them against the live router.

Together these establish the distinction between **mathematical dependency** and **physical scheduling opportunity**.

The next section can therefore address the scaling consequence:

> **If each layer is an independent unit and work can be pipelined, can each layer be scaled independently when one part of the model becomes the resource constraint?**

## 9. Independent layer scaling

Section 8 established that the model can remain mathematically sequential while the infrastructure pipelines different work across layers. The next scaling question is:

> **If one layer becomes the resource constraint, must the entire model be replicated to add capacity?**

With the layer boundary defined by the equation, the answer is structurally no.

A layer is an independently owned computation and data unit, so its storage, compute resources and replicas can be considered separately from the other layers.

### Scaling follows the layer, not the whole model

Consider Layer 37.

Its unit is:

```text
Layer 37
├── trunk
├── router
├── 896 experts
├── local state
└── execution resources
```

If Layer 37 requires additional capacity, the scaling unit can be Layer 37 itself:

```text
Layer 37
   ├── replica A
   ├── replica B
   └── replica C
```

while the other layers retain their existing placement.

This is fundamentally different from adding another complete copy of:

```text
Layer 0 + Layer 1 + ... + Layer 92
```

The architecture therefore creates a finer scaling granularity.

### Why independent scaling is possible

The reason is data ownership.

Each layer owns its own:

- trunk;
- router;
- expert store;
- layer-specific attention parameters;
- layer-local state;
- and execution resources.

The next layer does not need to know where the previous layer's weights are stored.

It only needs the runtime state produced by the previous stage.

That means a replicated Layer 37 unit can serve the same Layer 37 computation for different work while the rest of the model remains unchanged.

Conceptually:

```text
             ┌── Layer 37 replica A ──┐
Layer 36 ────┼── Layer 37 replica B ──┼── Layer 38
             └── Layer 37 replica C ──┘
```

A scheduler can distribute independent work among those replicas.

The exact scheduling policy is a deployment decision. The architectural property is that the replicas contain only the layer that is being scaled.

### Storage scales with layer ownership

Each MoE layer has approximately:

```text
896 experts × 17.55 MB ≈ 15.72 GB
measured layer store       ≈ 15.77 GB
```

Therefore a replica of Layer 37 needs another Layer 37 store, not another 1.45 TB model.

This is the physical meaning of layer-local ownership.

If a layer is replicated (R) times, its persistent expert storage scales approximately as:

```text
R × 15.77 GB
```

for that layer, rather than:

```text
R × 1.45 TB
```

for the entire expert pool.

This does not mean replication is always desirable. It means the cost of replication is localized to the computational unit being replicated.

### The measurements show why placement matters

The node-level contention experiment provides an important constraint.

When multiple layer processes share the same physical node, their storage/read resources are not independent.

The measured experiment showed approximately:

| Layer units sharing one node | Observed behavior |
|---|---|
| 1 | ~37.5 ms per stage |
| 2 | ~74–77 ms per stage |
| 3 | ~91–106 ms per stage |
| 4 | ~134–138 ms per stage |

The aggregate memory bandwidth remained in roughly the same **26–30 GB/s** range as more pods were added.

The important conclusion is not a universal throughput number.

It is that **placing more layer units on the same resource boundary does not create independent memory bandwidth**.

The pods begin competing for the same physical resource.

The experiment identified memory bandwidth as the binding constraint on that test machine rather than simply counting CPU cores or RAM capacity.

Therefore the architecture has two distinct scaling dimensions:

```text
logical scaling:
replicate the layer that needs capacity

physical scaling:
provide enough independent hardware resources
to keep those replicas from contending
```

### Replication does not mean duplicating the model

This distinction is central to the scaling argument.

A monolithic deployment might respond to a capacity requirement by adding another complete model instance.

The layer architecture can instead respond at the layer boundary:

```text
whole-model replication
        ↓
~1.45 TB expert data per copy

versus

layer replication
        ↓
~15.77 GB expert data per replicated MoE layer
```

The second number is not automatically the final infrastructure cost because every layer also has trunk, runtime memory and communication requirements.

But it demonstrates the difference in **scaling granularity**.

The unit being replicated is the unit that is actually constrained.

### Independent scaling must preserve the pipeline

There is one important limitation.

A layer cannot be replicated in isolation from the model dependency graph.

Layer 37 still depends on Layer 36 and must produce the state required by Layer 38.

So independent scaling means:

```text
independent resource ownership
        ≠
independent mathematical execution
```

The dependency chain remains:

```text
Layer 36 → Layer 37 → Layer 38
```

Replication simply provides multiple physical workers capable of executing the Layer 37 function.

The scheduler must therefore preserve the ordering and state affinity required by each request while distributing independent work across replicas.

### What this enables

Once layers are independently scalable, the infrastructure can be sized around actual resource requirements rather than total model size.

For example, different layers can have different resource profiles because K3 itself has different layer types:

```text
KDA layer
   ≠
MLA layer
   ≠
Layer 0 dense stage
```

A deployment can therefore reason separately about:

- storage capacity;
- memory;
- CPU/GPU allocation;
- replicas;
- network placement;
- and cache capacity.

The final allocation should come from measurements of the real distributed implementation, not from assuming every layer has identical requirements.

### Evidence behind this section

The evidence comes from:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which establishes layer-local parameters, operations and state.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures layer-store size and the contention behavior when multiple layer units share hardware.
- The architecture defined in Sections 3–8, which establishes that runtime state crosses the boundary while persistent layer weights remain local.

Together these establish the scaling **granularity**: a layer can be treated as an independently placeable and replicable unit, while the mathematical dependency graph remains intact.

They do not yet establish which layers will actually require replication in production. That is a workload-dependent deployment measurement.

The next section can therefore separate three different scaling dimensions:

> **client scaling, server scaling, and layer scaling are not the same problem and should not be solved by replicating the same resource.**

## 10. Client, server, and layer scaling are separate dimensions

Section 9 established that Layers 1–92 can be scaled independently. This section makes the complete production boundary explicit: the client, the server-side inference pipeline, and the individual layer pods are separate execution responsibilities.

The important point is that the diagram below describes the dependency path for one token. It does not describe the system's global execution schedule.

### The production flow

```
CLIENT
  │
  ├── token → vector
  │   embed_tokens lookup (~2.35 GB table)
  │
  ▼
SERVER-SIDE INFERENCE PIPELINE
  │
  ├── Layer 0
  │
  ├── Layer 1 → Pod 1
  ├── Layer 2 → Pod 2
  ├── Layer 3 → Pod 3
  │       ...
  └── Layer 92 → Pod 92
  │
  ▼
CLIENT
  │
  └── final vector → token
      lm_head (~2.35 GB vocabulary projection)
```

The client therefore performs the two natural token/vector boundary operations and does not need to host Layer 0 or the 1.45 TB expert pool.

```
token ID
   ↓
embedding lookup
   ↓
vector
   ↓
[server-side model transformation]
   ↓
final vector
   ↓
lm_head
   ↓
token ID
```

### Layer 0 is part of the server pipeline

Layer 0 is deliberately kept at the server-side model entry. It is dense and does not require the 896-expert store used by Layers 1–92. More importantly, its position at the beginning of the model flow gives the server the model state from which downstream work can be prepared.

The equation-derived routing behavior allows exact expert identities for a previously seen prompt to be available early enough to begin later-layer reads while earlier computation is still running. The live router remains the correctness authority.

Therefore the server-side entry is not simply an API gateway:

```
Client
   │
   │ embedding vector
   ▼
Server / Layer 0
   │
   │ prepared runtime state
   ▼
Layer 1 pod
```

Layer 0 is the first model transformation in the production pipeline.

### Layers 1–92 are separate work units

Each layer pod owns its layer's computation and data:

```
Pod 1   → Layer 1  → next state
Pod 2   → Layer 2  → next state
Pod 3   → Layer 3  → next state
...
Pod 92  → Layer 92 → final state
```

A pod does not need the complete model. It needs its own trunk, router, 896 experts where applicable, layer-local state and execution memory.

The runtime representation produced by one layer becomes the input to the next layer.

This gives the infrastructure a clean ownership rule:

> **Layer weights stay with their layer pod; runtime state moves along the pipeline.**

### The diagram is sequential for one token — the system is concurrent

For one token, the mathematical dependency is necessarily ordered:

```
L0 → L1 → L2 → L3 → ... → L92
```

Layer 2 cannot transform a token before Layer 1 has produced the state it needs.

But that does not mean Layer 2 must sit idle while the entire request completes.

Once Layer 1 has passed one unit of work downstream, it can work on another unit while Layer 2 works on the first.

For example:

```
Time →

Layer 0:  AAAAAA  BBBBBB  CCCCCC  DDDDDD
Layer 1:        AAAAAA  BBBBBB  CCCCCC  DDDDDD
Layer 2:              AAAAAA  BBBBBB  CCCCCC  DDDDDD
Layer 3:                    AAAAAA  BBBBBB  CCCCCC
...
Layer 92:                         AAAAAA  BBBBBB
```

Here Token A may be at Layer 5 while Token B is at Layer 4, Token C is at Layer 3, and Token D is at Layer 2. Each token still follows L0 → L1 → ... → L92. They are simply occupying different stages at the same time.

This is the central concurrency principle:

> **Sequential dependency does not imply global sequential utilization.**

### Work is separated so idle time is not propagated through the whole model

A monolithic execution model can make it appear that one request owns the entire model until completion.

The layer pipeline changes that scheduling unit.

Each pod is responsible for a bounded piece of work. When it finishes that piece for one token, it can accept other ready work instead of waiting for that token to finish Layers 1–92.

Conceptually:

```
Token A:  L0 → L1 → L2 → L3 → L4 → ...
Token B:       L0 → L1 → L2 → L3 → ...
Token C:            L0 → L1 → L2 → ...
Token D:                 L0 → L1 → ...
```

The dependency chain remains intact for every token, while the infrastructure remains occupied by different tokens and requests.

This is why the architecture should be described as a pipeline, not as 93 machines executing sequentially.

### Concurrency exists at multiple levels

There are several different kinds of work separation:

**Across requests**

```
Request A ─┐
Request B ─┼─→ shared pipeline stages
Request C ─┘
```

**Across tokens / positions**

Different positions from a prefill can occupy different stages or be processed in batches according to the runtime scheduler.

**Across layers**

Different pods can be doing useful work simultaneously because they are processing different ready states.

**Inside a layer**

Expert reads, attention work and other layer-local operations can overlap where the implementation and hardware allow it.

The exact scheduling policy is a deployment concern. The architectural property is that the work does not have to be serialized into one global request-at-a-time execution path.

### Routing creates another opportunity to remove idle time

The equation and routing experiments provide an additional scheduling opportunity.

For a previously seen prompt, the exact expert IDs required by later layers can be known while earlier layers are still running. Clover-K3 uses that information to start later-layer expert reads early.

The sequence becomes:

```
current layer computation
        │
        ├──────────────→ next-layer expert reads
        │
        ▼
current layer result
        │
        ▼
next layer computation
```

The read is prepared before the next layer needs the data.

The live router still executes and validates the cached route, so this is not speculative model execution. It is dependency-aware prefetching.

The measured Clover-K3 implementation reduced the tested warm path from approximately 8.75 s to 7.25 s, with device utilization increasing from roughly 80% to 98–99%. These measurements demonstrate the value of overlapping independent work in that implementation; they are not a universal throughput claim.

### The three scaling dimensions remain separate

The production system therefore has three different scaling boundaries:

| Boundary | Responsibility | Scaling response |
|---|---|---|
| Client | token→vector and final vector→token boundary work, requests | add/scale clients or request-serving capacity |
| Server pipeline | request handling, Layer 0 and pipeline coordination | scale server-side inference capacity |
| Layer pod | one layer's trunk, experts, state and computation | replicate the constrained layer |

A client bottleneck does not require another copy of the expert pool.

A server coordination bottleneck does not automatically require replication of every layer.

A Layer 37 bottleneck does not require another complete K3 model.

The architecture allows the scaling response to match the actual constrained resource.

### The server should not become a central runtime relay

The server coordinates the production pipeline, but it should not unnecessarily become a central relay for every layer-to-layer state transition.

The runtime path is conceptually:

```
Client
  ↓
Server / Layer 0
  ↓
Pod 1
  ↓
Pod 2
  ↓
...
  ↓
Pod 92
  ↓
Server / exit
  ↓
Client
```

The runtime state follows the layer dependency graph.

This preserves the separation between:

```
control / coordination
        and
model runtime data movement
```

For a distributed deployment, the internal layer-to-layer path should therefore be treated as the production inference fabric rather than as public client traffic.

### Why this architecture matters

The goal is not to make the diagram look distributed.

The goal is to ensure that no component is required to perform work that belongs to another component simply because the model was originally packaged as one large model.

The client handles the natural token/vector boundaries.

The server handles the model entry and production pipeline.

Each layer pod handles its own layer.

Different tokens and requests can occupy different stages concurrently.

And data preparation can begin as soon as the equation makes the dependency known.

That is the efficiency property this architecture is designed to provide.

### What is established and what is not

The repository establishes:

- the layer-local model structure;
- the client head/tail footprints;
- Layer 0's distinct dense role;
- layer-local expert stores;
- the runtime state passed between layers;
- routing predictability and exact validation;
- and measured lookahead behavior.

It does not yet establish the final production ratio of:

```
clients : server capacity : layer-pod replicas
```

That depends on workload, sequence length, batching, concurrency, cache reuse, network placement and actual deployment measurements.

The architecture establishes the boundaries and opportunities for concurrency. The deployment experiment must determine the final capacity.

### Evidence behind this section

This section is written to make the production execution model explicit and to distinguish per-token dependency from system-wide scheduling.

The evidence comes from:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the ordered layer transformations and carried state.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures layer footprints, inter-layer payloads, node contention and routing lookahead.
- [k3-analysis/k3-client-server-architecture.md](../../../research/k3/architecture/k3-client-server-architecture.md), which records the approximately 2.35 GB embedding lookup and approximately 2.35 GB final vocabulary projection at the client boundary.
- The verified Clover-K3 implementation, which validates cached expert IDs against the live router.

These sources support the architectural separation and the concurrency model. They do not by themselves establish the final production throughput or optimal deployment size.

The next section can therefore examine the efficiency consequence:

> **Why does organizing infrastructure around the model's layer-local work change the resource-sizing problem compared with one monolithic large-GPU deployment?**
