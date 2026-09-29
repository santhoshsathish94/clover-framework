# Scaling resource sizing and evidence

**Status: proposed design; distributed behavior is not verified.** [Reading path](README.md). Section numbers are retained for existing references.

## 11. Why this changes the resource-sizing problem

The previous sections established the architecture:

```
Client
   ↓
Layer 0
   ↓
Layer 1 → Layer 2 → ... → Layer 92
   ↓
Client
```

with Layer 0 at the server-side entry and Layers 1–92 owned by their respective pods.

This section explains why that decomposition changes the **resource-sizing problem**.

The argument is not that a distributed layer architecture is automatically faster than a large GPU.

The argument is that the model's actual resource requirements are **not uniform across the whole model**, so sizing one machine around the total model can allocate resources at a much coarser granularity than the computation requires.

### A monolithic model starts with the total footprint

K3 has approximately:

```
82,432 experts
≈ 1.45 TB of expert weights
```

The conventional sizing question therefore becomes:

> How much hardware is required to hold and execute the complete model?

But the equation and measurements show that the model does not use that entire expert pool at one point in the computation.

A five-token prefill touched:

```
5,683 distinct (layer, expert) pairs
≈ 99.72 GB
```

against the approximately 1.45 TB total expert pool.

The model therefore has a large **aggregate footprint** but a much smaller **layer-local working requirement** at any one point.

### The layer architecture changes the sizing unit

Instead of sizing around:

```
whole model
≈ 1.45 TB expert pool
```

the layer architecture sizes around:

```
one layer
≈ 15.77 GB expert store
+ 423–635 MB trunk
+ runtime memory
+ layer-to-layer communication
```

The 92 layer stores together still contain approximately the same total expert data.

Nothing has been magically compressed away.

The difference is **where the resource is required**.

The infrastructure can therefore provide storage and compute resources near the layer that uses them instead of requiring one execution boundary to own the entire model.

### Storage capacity and compute capacity become separate decisions

A large model footprint does not imply that all model data must be resident in the same fast memory tier at the same time.

The measured layer experiments showed that a 15.77 GB expert store can remain on local storage while the layer process keeps a much smaller resident working set.

That creates separate resource decisions:

```
persistent storage
        ↓
holds layer experts

resident memory
        ↓
holds active computation and state

compute
        ↓
executes the layer

network fabric
        ↓
moves runtime state to the next layer
```

A monolithic design tends to couple these requirements around one large execution boundary.

The layer design allows them to be considered independently.

This is an efficiency property, not yet a throughput claim.

### The runtime state is much smaller than the model weights

The model does not pass its weights between layers.

It passes runtime state.

For the measured K3 implementation, the hop payload for five positions was approximately:

```
70.53 MB
```

across all 92 layer transitions.

For one position it was approximately:

```
14.11 MB
```

This is fundamentally different from moving the model itself.

The architecture therefore keeps the large, relatively stationary data local to its owner and moves the smaller representation required to continue the computation.

Conceptually:

```
large, stationary:
Layer N weights
       │
       │ local
       ▼
Layer N computation
       │
       │ small runtime state
       ▼
Layer N+1
```

That is the physical basis for separating the model into layer-local resources.

### More GPU is not automatically the same as more useful capacity

The measurements also show why simply adding more compute hardware is not a sufficient architectural argument.

On the test machine, the one-pod experiment reached approximately 24.5 GB/s steady expert-read bandwidth. When multiple layer processes shared the same physical node, aggregate memory bandwidth remained around 26–30 GB/s while the individual pods increasingly competed for that bandwidth.

The measured behavior was approximately:

```
1 pod  → ~37.5 ms
2 pods → ~74–77 ms
3 pods → ~91–106 ms
4 pods → ~134–138 ms
```

The important observation is that adding more logical work units did not create proportional physical memory bandwidth.

The binding resource was the shared memory subsystem.

This is why the architecture must size **the resource that constrains the actual workload**, rather than simply increasing one class of compute hardware.

The same principle applies to a GPU-based deployment: if the dominant cost is moving expert data or feeding the execution units, additional arithmetic capacity alone does not necessarily remove the limiting resource.

### The model's arithmetic intensity does not remove the data problem

The scaling analysis records an overall K3 arithmetic intensity of approximately **10.3 FLOP per byte** for the relevant workload.

That number is useful because it puts the model on a concrete compute-versus-data scale.

But it should not be interpreted as:

> therefore the answer is simply a larger GPU.

The model contains a large collection of independently owned expert weights and a layer-by-layer dependency graph.

The question is not only how many floating-point operations can be executed.

It is also:

```
Where are the required bytes?
When are they needed?
Who owns them?
Can they be prepared before computation reaches them?
What physical resource moves them?
```

The equation makes those dependencies explicit.

The scaling architecture then places the resources around those dependencies.

### The architecture does not eliminate the total model cost

This distinction is important.

The layer architecture does **not** claim:

```
1.45 TB → 15.77 GB
```

as though the remaining 91 layers disappeared.

The actual transformation is:

```
1.45 TB global expert pool
          ↓
92 × approximately 15.77 GB layer stores
```

The aggregate storage remains roughly the same.

What changes is the **ownership and scaling boundary**.

Likewise, the architecture does not claim that every deployment can use inexpensive hardware. Each layer still requires sufficient storage bandwidth, resident memory, compute and network capacity for its workload.

The benefit comes from being able to allocate those resources according to the actual layer requirements.

### Replication becomes localized

This becomes particularly important when capacity must increase.

With whole-model replication:

```
additional capacity
        ↓
another complete model footprint
```

With layer-local replication:

```
Layer 37 constrained
        ↓
replicate Layer 37
        ↓
another Layer 37 resource footprint
```

A MoE layer replica adds approximately one layer expert store, rather than another 1.45 TB expert pool.

The other layers do not need to be duplicated merely because Layer 37 needs additional capacity.

This is the central resource-efficiency difference between the two scaling granularities.

### Concurrency makes the separation useful

The architecture would not provide much value if every request still required one machine to perform the complete model sequentially.

Section 10 established the opposite.

For one token:

```
L0 → L1 → L2 → ... → L92
```

But across multiple tokens or requests:

```
Layer 5  → Token A
Layer 4  → Token B
Layer 3  → Token C
Layer 2  → Token D
```

can all be active simultaneously.

That means the layer resources are not merely storage containers. They become **pipeline stages** that can remain occupied with independent ready work.

The infrastructure therefore separates both:

1. **where the model data lives**, and
2. **when each piece of work executes**.

### Prefetch adds another efficiency mechanism

The routing-cache experiment provides a second example of why following the model's dependency graph matters.

For a previously seen prompt, exact later-layer expert IDs can be known early enough to begin their reads while earlier-layer computation is still running.

The tested Clover-K3 path changed from approximately:

```
8.75 s → 7.25 s
```

with measured device utilization increasing from roughly 80% to 98–99%.

Again, this is not evidence that every distributed deployment will achieve the same improvement.

It demonstrates a more general principle:

> **Once the computation is represented precisely, work that is known to be required can be moved earlier in the execution schedule.**

The scaling architecture can then place that work close to the resource that will consume it.

### What this section establishes

The evidence supports a narrower conclusion than "distributed is faster."

It establishes that K3 has several different resource scales:

```
whole-model storage
        ↓
~1.45 TB expert pool

layer-local storage
        ↓
~15.77 GB per MoE layer

layer trunk
        ↓
~423–635 MB

runtime inter-layer state
        ↓
measured in MB, not TB

client head/tail
        ↓
~2.35 GB each
```

These are different resources serving different parts of the model.

A production architecture can therefore size them separately.

That is the efficiency argument.

### What remains unproven

This section deliberately does not claim that the architecture is already cheaper, faster, or more scalable in every deployment.

Those conclusions require:

- actual distributed deployment;
- real network measurements;
- real layer execution measurements;
- concurrent workload testing;
- failure and recovery behavior;
- scheduling measurements;
- and cost comparison against a monolithic deployment.

Those experiments belong in the later validation sections.

What is established now is the structural reason the comparison should be made:

> **A monolithic deployment sizes around the aggregate model. Clover-K3 sizes around the resources actually required by each stage of the model's execution.**

### Evidence behind this section

This section is supported by:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which establishes the layer-by-layer computation and state dependency.
- [k3-analysis/k3-data-problem.md](../../../research/k3/experiments/k3-data-problem.md), which measures the 1.45 TB expert pool and the much smaller per-prompt distinct expert requirement.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures layer storage, trunk sizes, resident memory behavior, node contention, runtime hop payloads and routing lookahead.
- The verified Clover-K3 implementation, which provides the exact layer execution and routing validation.

These measurements establish the resource boundaries and the reason for separating them. They do not yet establish final production cost or throughput.

The next section can therefore make the distinction explicit:

> **The architecture is being proposed first as an efficiency model. Throughput is a separate experimental question.**

## 12. Efficiency versus throughput

The previous section established why the architecture changes the resource-sizing problem. This section makes an important distinction before any production result is claimed:

> **Clover-K3 is currently an efficiency architecture. Throughput is a separate experimental result.**

These two questions are related, but they are not the same.

### Efficiency asks how resources are used

Efficiency asks questions such as:

```text
How much model data must be stored?
Where is that data stored?
How much data is resident at one time?
How much runtime state must move?
Which layer owns each resource?
Can work be overlapped?
Can a constrained layer be scaled independently?
```

The measurements already provide evidence for these questions.

For example:

- the full expert pool is approximately 1.45 TB;
- one MoE layer's expert store is approximately 15.77 GB;
- the layer trunk is approximately 423–635 MB;
- the five-position inter-layer payload is approximately 70.53 MB across the complete 92-layer path;
- client-side embedding and final vocabulary projection are approximately 2.35 GB each;
- multiple layer processes sharing one node compete for the same physical memory bandwidth.

These measurements describe **resource organization and utilization**.

They do not require a claim about final tokens per second.

### Throughput asks how much work the system completes

Throughput is a workload-level measurement.

Examples include:

```text
tokens / second
requests / second
prompts / second
time per generated token
time per batch
```

Throughput depends on much more than the size of an individual layer.

It depends on:

- request concurrency;
- prompt length;
- generated sequence length;
- batch composition;
- cache reuse;
- routing-cache availability;
- layer scheduling;
- inter-layer network latency;
- storage/cache behavior;
- compute utilization;
- and contention between concurrent workloads.

Therefore a layer footprint alone cannot establish production throughput.

### Why the distinction matters

It would be incorrect to take:

```text
1.45 TB model
        ↓
92 layer stores
```

and conclude directly:

```text
therefore faster
```

The decomposition proves a different thing:

```text
model structure
        ↓
resource ownership
        ↓
independent placement
        ↓
independent scaling opportunities
```

Whether those opportunities produce higher throughput must be measured after the production pipeline exists.

This is especially important because the architecture introduces new resources as well as removing old coupling.

A distributed pipeline introduces:

- layer-to-layer communication;
- scheduling;
- synchronization at dependency boundaries;
- network placement;
- replica coordination;
- and failure handling.

Those costs must be measured rather than assumed away.

### The existing lookahead result is a different kind of evidence

Clover-K3 already has one measured optimization that demonstrates the value of exploiting the equation's dependency structure.

For a previously seen prompt, exact later-layer expert IDs can be available early enough to start expert reads while the current layer is still computing.

The measured path changed approximately:

```text
baseline       8.75 s
lookahead      7.25 s
reduction      1.49 s
```

with device utilization moving from approximately 80% to 98–99%.

That is useful evidence for **overlap efficiency**.

It is not a measurement of the final distributed architecture's throughput.

The distinction is:

```text
lookahead experiment
        ↓
measures one optimization on the current implementation

distributed deployment
        ↓
must measure end-to-end throughput
```

The first informs the second, but they are not interchangeable.

### Efficiency can improve without increasing throughput

This is possible because throughput may be limited by a different resource.

For example:

```text
less resident memory
```

is an efficiency improvement even if tokens/second stays unchanged.

Likewise:

```text
less duplicated storage
```

can reduce infrastructure requirements without changing single-request latency.

And:

```text
better pipeline occupancy
```

may reduce idle resources without immediately increasing end-to-end throughput if another stage remains the bottleneck.

Therefore the architecture should be evaluated using both classes of measurements.

### Throughput can also improve without proving the architecture is efficient

The reverse is also possible.

A larger machine might produce more tokens per second simply because it provides more raw compute or memory bandwidth.

That does not by itself prove that the resources are being used efficiently.

The comparison therefore needs two separate dimensions:

| Dimension | Question |
|---|---|
| Efficiency | How much resource is required and how effectively is it utilized? |
| Throughput | How much inference work is completed per unit time? |

A valid production comparison should report both.

### The eventual experiment

The distributed deployment should therefore measure at least:

```text
single-request latency
prefill latency
decode latency
tokens / second
requests / second
layer utilization
storage bandwidth
memory bandwidth
network bandwidth
resident memory
CPU/GPU utilization
pipeline idle time
```

Those measurements should be collected across increasing concurrency rather than from one isolated request.

The resulting curves will show whether the pipeline:

```text
scales linearly
        or
hits a layer bottleneck
        or
hits a network bottleneck
        or
hits a storage/memory bottleneck
```

That is the point at which a throughput claim becomes evidence-based.

### What this section establishes

At the current stage, the strongest defensible statement is:

> **Clover-K3 defines an efficiency-oriented architecture by aligning resource ownership with the model's computation. Whether that architecture produces a throughput advantage is an empirical question for the distributed deployment.**

This keeps the architectural claim and the performance claim separate.

The next section can therefore quantify what is already known well enough to size today, without pretending that unknown deployment behavior has already been measured.

### Evidence behind this section

This section is supported by:

- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which contains the measured storage, memory, contention, payload and lookahead experiments.
- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which establishes the computation and dependency structure.
- The verified Clover-K3 implementation and its lookahead measurements.

These establish the distinction between measured resource behavior and unmeasured distributed throughput.

## 13. What is known and can be sized now

The previous section separated efficiency from throughput. This section turns the measured architecture into concrete quantities that can already be used for planning.

The important distinction is:

> **We can size the structural resources now. We cannot yet claim the final production hardware configuration.**

The model equation and the scaling measurements already establish the size and ownership of the major data components.

### The complete model footprint is known

The expert pool contains:

```
82,432 experts
≈ 1.45 TB total expert weights
```

Those experts are distributed across:

```
92 MoE layers
× 896 experts per layer
= 82,432 experts
```

So the expert storage requirement is not an unknown quantity.

It is a measured model property.

### Each MoE layer has a known persistent footprint

A single MoE layer contains:

```
896 experts
≈ 17.55 MB per expert
≈ 15.72 GB packed expert data
≈ 15.77 GB measured layer store
```

The layer also has its non-expert trunk.

The measured trunk sizes are:

```
KDA layer: ≈ 635 MB
MLA layer: ≈ 423 MB
```

There are 68 KDA layers and 24 MLA layers among Layers 1–92. Layer 0 is the remaining KDA layer, giving 69 KDA and 24 MLA layers across the full model.

Layer 0 is separate:

```
Layer 0 trunk: ≈ 1.172 GB
Layer 0: dense, no expert store
```

Therefore the persistent storage boundary is already well defined:

```
Client
  ├── embedding lookup      ≈ 2.35 GB
  └── lm_head               ≈ 2.35 GB

Server
  └── Layer 0               ≈ 1.172 GB

Layer pods
  └── Layers 1–92
       ├── trunk             ≈ 423–635 MB
       └── expert store      ≈ 15.77 GB
```

These numbers are sufficient to estimate storage requirements before selecting the final server hardware.

### The aggregate expert storage can be reconstructed

The layer decomposition provides a useful consistency check.

```
92 × 896 = 82,432 experts
```

and:

```
92 × approximately 15.77 GB
≈ 1.45 TB
```

The aggregate is therefore consistent with the original model footprint.

The architecture has not changed the model's total expert data.

It has made the ownership boundary explicit.

### Runtime memory is a separate quantity

The persistent store size does not directly determine the resident memory requirement.

A layer process needs memory for:

- its trunk;
- active expert buffers;
- residual and snapshot state;
- KDA recurrent state where applicable;
- ShortConv history;
- MLA cache state where applicable;
- temporary execution buffers.

The current measurements establish the storage footprint and observed working-set behavior, but the final production resident-memory requirement depends on the actual layer implementation and concurrency.

Therefore:

```
15.77 GB layer store
        ≠
15.77 GB mandatory private RAM
```

This distinction is important when selecting hardware.

### The inter-layer communication budget is also measurable

The runtime state crossing layer boundaries has already been measured.

For the K3 implementation:

| Positions | Total payload across the 92-hop path |
|---:|---:|
| 1 | 14.11 MB |
| 5 | 70.53 MB |
| 16 | 225.71 MB |
| 64 | 902.82 MB |

These are useful planning numbers because they establish the scale of the internal inference fabric.

They do not by themselves determine the required network speed.

The network requirement depends on:

- how many requests are concurrently in flight;
- how many positions are transferred together;
- whether multiple layer hops share the same physical links;
- placement of replicas;
- and the desired latency budget.

But the payload itself is no longer an unknown.

### Client-side model data is also known

The client has two model-side data structures:

```
embed_tokens ≈ 2.35 GB
lm_head      ≈ 2.35 GB
total        ≈ 4.70 GB
```

The two operations are different.

The embedding table is a lookup:

```
token ID
   ↓
one row of embed_tokens
   ↓
7,168-element vector
```

The final vocabulary projection is a full projection:

```
final vector
   ↓
lm_head
   ↓
163,840 logits
   ↓
selected token
```

This means the client is not carrying a second copy of the expert model.

It carries only the two model boundaries that naturally convert:

```
token → vector
vector → token
```

That makes the client-side footprint approximately 4.70 GB of model data, separate from the server-side Layer 0 and the layer-pod expert stores.

### The storage topology can therefore be estimated

Before deployment, the persistent model data can already be represented as:

```
Client
≈ 4.70 GB
  ├── embedding
  └── lm_head

Server-side Layer 0
≈ 1.172 GB

92 layer pods
≈ 92 × 15.77 GB expert stores
+ 92 layer trunks
```

The expert stores dominate the distributed persistent footprint.

The trunks add a smaller amount relative to the expert stores, with the exact total determined by the KDA/MLA composition.

This gives enough information to plan storage capacity and identify which component dominates it.

### What can be sized now

Based on the current evidence, the following can already be sized or bounded:

| Resource | What is known now |
|---|---|
| Expert storage | ~1.45 TB aggregate |
| Per-MoE-layer expert store | ~15.77 GB |
| Per-layer trunk | ~423–635 MB |
| Layer 0 trunk | ~1.172 GB |
| Client head/tail model data | ~4.70 GB |
| Inter-layer payload | ~14.11 MB per position across 92 hops |
| Five-position payload | ~70.53 MB |
| Layer ownership | one layer per pod |
| Expert count | 896 per MoE layer |
| Total MoE layers | 92 |

These are model and architecture quantities rather than assumptions about a particular hardware vendor.

### What cannot be sized with confidence yet

Several production quantities remain workload-dependent:

- resident memory required by a fully implemented layer;
- compute capacity required per layer;
- network bandwidth required under target concurrency;
- number of replicas required for each layer;
- server capacity for Layer 0 and request coordination;
- scheduling overhead;
- failure/recovery capacity;
- and end-to-end throughput.

Those require the actual production pipeline.

This is an important boundary in the document.

We should not turn measured model dimensions into invented hardware specifications.

### Why this matters

The architecture now has a useful property:

> **The unknowns are no longer the model's structure. The remaining unknowns are deployment behavior.**

The model data can be partitioned, measured and assigned before hardware is chosen.

The deployment experiment then answers the next layer of questions:

```
known model structure
        ↓
known resource footprint
        ↓
deploy
        ↓
measure real behavior
        ↓
size hardware and replicas
```

This is the correct order because hardware should be selected from observed resource requirements rather than from the model's total parameter count alone.

### Evidence behind this section

This section is supported by:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the model's tensors, layer structure and state.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which measures the layer stores, trunk sizes, runtime payloads, memory behavior and contention.
- [k3-analysis/k3-client-server-architecture.md](../../../research/k3/architecture/k3-client-server-architecture.md), which records the approximately 4.70 GB client head/tail model footprint.

These sources establish the quantities that can be sized now. They do not establish final production hardware or replica counts.

## 14. What remains unknown until deployment

Section 13 established the quantities that can already be sized from the model and measured experiments. This section defines the boundary between those known quantities and the behavior that can only be established by running the actual distributed system.

The purpose is to prevent a resource estimate from being mistaken for a deployment result.

### The layer footprint does not determine the final hardware

A layer has a known persistent footprint:

- approximately 15.77 GB expert store;
- approximately 423–635 MB trunk;
- runtime state and working memory.

But that does not tell us whether one particular CPU, GPU, or accelerator configuration is sufficient for the complete layer implementation.

The actual layer pod must execute attention, normalization, router computation, expert selection, selected expert computation, recurrent state updates where applicable, snapshot aggregation, and the remaining layer operations.

The current storage experiments isolate important data-movement behavior. They do not constitute a full production layer benchmark.

Therefore the required compute device remains an experimental quantity.

### Resident memory must be measured with the real layer

The measured expert store is persistent data.

A production pod may keep some of that data cached or resident depending on storage backend, cache policy, active expert set, request concurrency, batching, prefetch depth, and execution strategy.

The runtime state also grows with workload shape.

Therefore the correct measurement is not simply:

    RAM = 15.77 GB

but:

    resident memory
    = trunk
    + active expert data
    + runtime state
    + execution buffers
    + cache
    + concurrency overhead

The actual value must be measured with the complete layer implementation under representative workloads.

### Network requirements depend on concurrency and placement

The inter-layer payload is known.

The required network capacity is not.

A five-position workload carries approximately 70.53 MB across the complete 92-hop path. But concurrent requests can place multiple layer transitions on the network simultaneously.

The required capacity therefore depends on payload size, transitions per second, concurrency, and topology.

If adjacent layers share a physical link, their traffic can contend. If replicas are placed on the same host, they can contend for local memory bandwidth instead.

The architecture therefore gives us a measurable communication budget, but deployment determines the required network fabric.

### Replica count is an observed scheduling result

The architecture allows a constrained layer to be replicated independently.

It does not tell us in advance how many replicas are necessary.

The operational sequence is:

    layer demand
        ↓
    measured service capacity
        ↓
    queueing / idle time
        ↓
    required replicas

If Layer 37 becomes a limiting stage under a target workload, deployment can measure its queue depth and service time and determine whether another Layer 37 replica is useful.

The same process applies independently to other layers.

### Pipeline behavior must be measured end to end

The mathematical dependency is already known:

    L0 → L1 → L2 → ... → L92

The deployment question is how efficiently independent work can occupy those stages.

The system must therefore measure stage utilization, queue depth, time waiting for the next layer, network transfer time, expert-read time, compute time, synchronization time, and end-to-end latency.

A pipeline can have individually fast stages and still perform poorly if work frequently waits between stages.

That behavior cannot be inferred from layer sizes alone.

### Prefetch and routing lookahead must remain correctness-gated

The current Clover-K3 implementation provides measured evidence that exact routing information for a previously seen prompt can be used to start later expert reads early.

This is not treated as an approximation. The cached route is checked against the live router, and disagreement is a correctness failure.

Deployment must therefore measure both the benefit and the cost of prefetching.

The cost can include cache memory, unnecessary storage reads, synchronization, route validation, and additional scheduling complexity.

The existing 17.0% lookahead reduction is evidence for the current implementation and workload. It is not a universal deployment coefficient.

### Failure and recovery are also unknown

A 92-stage pipeline introduces more operational boundaries than a monolithic process.

Deployment must therefore test pod failure, replica failover, lost in-flight state, storage failure, network interruption, restart time, state reconstruction, and recovery without corrupting model output.

The mathematical model defines what the computation should produce. It does not define the operational behavior of a failed distributed system.

### Cost cannot be inferred yet

The architecture changes the unit of resource ownership. It does not yet establish the cost of the final system.

A meaningful cost comparison requires measured values for hardware, storage, networking, power, replication, utilization, and operational overhead.

A lower per-pod footprint does not automatically mean a lower total infrastructure bill.

### What deployment must answer

| Unknown | Deployment evidence required |
|---|---|
| Compute capacity | Complete layer benchmark |
| Resident memory | Full layer workload under representative concurrency |
| Network capacity | Measured inter-layer traffic and contention |
| Replica count | Queueing and service-time measurements |
| Pipeline efficiency | Stage utilization and end-to-end latency |
| Prefetch value | Read/compute overlap and correctness validation |
| Failure behavior | Fault-injection and recovery tests |
| Cost | Measured infrastructure and utilization |
| Throughput | End-to-end workload benchmark |

### What this section establishes

> **The architecture is sufficiently defined to build the experiment, but the production performance characteristics remain empirical.**

The current evidence establishes what the model contains, how resources can be partitioned, and what data crosses the boundaries.

Deployment must establish how fast it runs, how much memory and network it requires, how many replicas are useful, where contention appears, what it costs, and how it behaves under failure.

Only after those measurements can the architecture be described with production performance claims.

### Evidence behind this section

This section is supported by:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which defines the computation and state dependencies.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which provides measured layer storage, memory-bandwidth contention, inter-layer payloads and the lookahead experiment.
- The verified Clover-K3 implementation, which provides the correctness-gated routing lookahead behavior.

These sources establish what is already known and where deployment evidence is still required.
