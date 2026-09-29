# Clover-K3 documents

The [program](../README.md) is the single-machine reference. Distributed serving remains
a proposed design. These documents separate what the implementation does, what was measured,
and what still needs to be built and validated.

Folder guides: [reference](reference/README.md), [evidence](evidence/README.md),
[scaling](scaling/README.md).

| Read | Purpose |
|---|---|
| [Model equation](../../research/k3/model/k3-model-equation.md) | Mathematical specification and its verification scope |
| [Implementation equation](reference/implementation-equation.md) | What the compiled program evaluates, including differences from the specification |
| [Evidence methods](evidence/evidence-methods.md) | Distinct campaigns, available harnesses, configurations and reproduction gaps |
| [Results](evidence/results.md) | Recorded first-token comparison and separate generation outputs |
| [Scaling entry point](scaling/README.md) | Current boundary and chapter navigation |
| [Scaling overview](scaling/overview.md) | How model understanding exposed the data-movement problem |
| [Architecture contract](scaling/architecture.md) | Client, server, layers, state, concurrency and replication |
| [Resource sizing and evidence](scaling/resources.md) | Known sizes, measured inputs and unknown deployment requirements |
| [Deployment validation](scaling/validation.md) | Proposed correctness, performance and recovery experiments |
| [Investigation index](../../research/k3/README.md) | Derivations, measurements, corrections and historical context |

Do not merge the two equations or turn a recorded result into a general performance claim.