# Scaling deployment validation

**Status: proposed design; distributed behavior is not verified.** [Reading path](README.md). Section numbers are retained for existing references.

## 15. The deployment experiment and validation plan

The previous sections established the architecture, the resource boundaries, what can already be sized, and what remains unknown.

This final section defines how those unknowns should be measured.

The purpose of the experiment is not to prove the architecture by assumption. It is to expose the architecture to real workloads and measure whether the proposed resource boundaries behave as expected.

### Start with a correctness gate

The distributed system must first produce the same model result as the verified Clover-K3 implementation.

The reference path is the correctness oracle.

For each test workload, compare:

- final logits;
- selected token;
- intermediate state where practical;
- routing decisions;
- and relevant cache/state transitions.

The existing verification already establishes bit-exact agreement for the implemented model path:

- 79,742,816 floats identical for embedding-to-token;
- 96,587,584 floats identical for prefill plus decode;
- zero mismatches;
- maximum ULP difference of zero.

The distributed experiment must preserve that property.

Performance measurements are not meaningful if the distributed pipeline changes the computation.

### Deploy the smallest complete pipeline first

The first deployment should contain the complete logical path:

    client
      ↓
    server-side Layer 0
      ↓
    Layer 1
      ↓
    Layer 2
      ↓
      ...
      ↓
    Layer 92
      ↓
    server-side output
      ↓
    client lm_head

This is the smallest deployment that exercises the proposed architecture without replacing real stages with assumptions.

The client remains responsible for token-to-vector and final vector-to-token operations.

The server-side inference fabric owns Layer 0 and the internal Layer 1–92 pipeline.

### Establish a single-layer baseline

Before running the full pipeline, measure each layer independently.

For every layer record:

- trunk load time;
- expert-read time;
- attention time;
- router time;
- expert computation time;
- state-update time;
- total service time;
- resident memory;
- storage bandwidth;
- CPU/GPU utilization.

This establishes the service characteristics of each stage.

It also identifies which layers have materially different resource profiles.

KDA and MLA layers should not be assumed to have identical behavior simply because they occupy the same position in the pipeline.

### Establish the communication baseline

Run the same layer transitions with controlled network conditions.

Measure:

- payload size;
- serialization time;
- transmission time;
- deserialization time;
- end-to-end hop latency;
- link utilization;
- packet loss or retransmission where applicable.

Then repeat at increasing concurrency.

The objective is to determine when communication becomes a meaningful part of stage service time.

The measured payloads from Section 13 provide the starting workload sizes; deployment determines their real cost.

### Establish pipeline behavior

Once individual stages and communication are characterized, run the complete 92-stage pipeline.

Start with one request and then increase concurrency systematically.

For each concurrency level measure:

| Measurement | Purpose |
|---|---|
| End-to-end latency | User-visible cost |
| Prefill latency | Initial prompt processing |
| Decode latency | Incremental generation cost |
| Tokens/sec | Work completion rate |
| Requests/sec | Service capacity |
| Layer utilization | Stage occupancy |
| Queue depth | Scheduling pressure |
| Pipeline idle time | Dependency-induced waiting |
| Network bandwidth | Fabric pressure |
| Storage bandwidth | Expert-read pressure |
| Resident memory | Working-set pressure |
| CPU/GPU utilization | Compute pressure |

The resulting curves show where the pipeline stops scaling.

### Test the independent-scaling hypothesis

Select a layer that becomes a measured bottleneck.

Replicate only that layer.

Then repeat the same workload.

The experiment should answer:

    Does adding one replica of the constrained layer
    reduce its queueing and improve system behavior
    without requiring replication of the entire model?

This is the central scaling hypothesis of the architecture.

The result should be measured rather than assumed.

The experiment should also test whether placing multiple replicas on the same physical node simply moves the bottleneck to shared memory bandwidth, as already observed in the local contention measurements.

### Test routing lookahead separately

Run the pipeline with:

1. lookahead disabled;
2. correctness-gated lookahead enabled.

Compare:

- expert-read overlap;
- device utilization;
- pipeline idle time;
- storage bandwidth;
- total latency;
- route-validation failures;
- and unnecessary reads.

The existing 8.75 s → 7.25 s result establishes that this mechanism can produce measurable overlap in the current implementation.

The distributed experiment determines whether the same principle remains useful when layer boundaries and network transfers are introduced.

### Test scaling dimensions independently

The architecture has three distinct scaling dimensions:

    Client
       ↓
    Server-side inference pipeline
       ↓
    Layer pods

Do not change all three at once.

First establish a stable client configuration.

Then vary pipeline resources.

Then vary individual layer replicas.

This makes it possible to identify which resource caused an observed change rather than attributing every result to the architecture as a whole.

### Test failure and recovery

After correctness and performance behavior are established, introduce controlled failures.

At minimum:

- stop a layer pod;
- remove a layer replica;
- interrupt a network path;
- interrupt expert storage;
- restart a failed process.

Measure:

- detection time;
- recovery time;
- lost work;
- state reconstruction;
- output correctness after recovery;
- and impact on other in-flight requests.

This establishes whether the logical layer boundary is also a workable operational boundary.

### Define success before measuring

The experiment should not be declared successful simply because a distributed version runs.

The architecture should be evaluated against explicit observations:

1. The distributed pipeline produces the same model result as the verified reference.
2. Resource ownership follows the defined layer boundaries.
3. Inter-layer state remains bounded and measurable.
4. Independent layer replication can be performed without replicating the entire model.
5. Pipeline concurrency can be increased without introducing unexplained correctness failures.
6. Bottlenecks can be identified at the layer, storage, memory, compute, or network level.
7. Deployment measurements are sufficient to derive actual hardware and replica requirements.

These are validation criteria, not performance promises.

### The expected outcome

The experiment has two possible useful outcomes.

If the measured system behaves as the architecture predicts, the measurements provide evidence for the scaling model.

If a different bottleneck appears, that is equally valuable: the architecture is then refined from observation rather than protected by assumption.

This follows the Clover principle:

    Context
      ↓
    Direction
      ↓
                Execution
      ↓
                Outcome
                        ↓
                Growth
      ↓
    Context

The deployment is the next context.

The measurements provide the evidence for the next direction.

### Evidence behind this section

This section is grounded in:

- [k3-analysis/k3-model-equation.md](../../../research/k3/model/k3-model-equation.md), which provides the mathematical model and correctness reference.
- [k3-analysis/clover-scaling-architecture.md](../../../research/k3/architecture/clover-scaling-architecture.md), which provides the measured resource, contention, payload and lookahead baselines.
- The verified Clover-K3 implementation, which provides the executable reference behavior and correctness gate.

The purpose of the experiment is therefore not to manufacture a conclusion. It is to let deployment measurements determine what the architecture actually achieves.
