# Evidence methods and reproduction limits

**Scope reviewed: 2026-09-29.** This is a map of recorded evidence and available tooling,
not a new benchmark run. The single-machine measurements concern a Ryzen 9 7950X3D,
124 GB RAM and NVMe storage under the configurations recorded in the linked sources.

| Observation | Method and source | What it does not establish |
|---|---|---|
| Preserved logits checksum | [gate.sh](../../gate.sh), [build and run instructions](../../README.md#the-gate) | Correctness for all prompts, all configurations or a distributed implementation |
| Model-equation float comparisons | [Model equation verification](../../../research/k3/model/k3-model-equation.md) | General model quality; coverage is limited to the tested prefill and decode paths |
| 34/34 first-token agreement; 394.94 s vs 1344.23 s | [Results](results.md), [optimization journal](../../../research/k3/experiments/k3-equation-solution.md) | Multi-token agreement or a hardware-independent 3.40x improvement |
| Twelve generated tokens per prompt | [Recorded outputs](results.md), [generation driver](../../gen.py) | A matched multi-token timing or quality comparison against the independent engine |
| Cached-route lookahead, 8.75 s to 7.25 s | [Measurement journal](../../../research/k3/architecture/clover-scaling-architecture.md), [campaign script](../../proof-campaign.sh) | Performance on unseen prompts, GPUs or a distributed cluster |

## Running the supported reference

Follow [configuration and build](../../README.md#configuration), supply the external checkpoint
and packed trunk, and run the preserved gate before interpreting timing. Model weights,
machine-specific generated indexes and the original binary logits artifact are not shipped.
The reference's build flags and gate are not changed by this documentation reorganization.

## Historical campaign tooling

`proof-campaign.sh` compares two configurations of Clover: baseline with route capture,
then cached-route lookahead, checking their logits. It does not invoke the independent
engine. It assumes `/opt/clover-k3` and writes `/root/k3proof`, so it is a historical
machine-specific harness, not a portable command reproducing the comparison table.

The independent-engine comparison and the generation run are separate campaigns.
In the first comparison prompt, 5.73 seconds of the observed gap was attributed to the
other configuration rereading the trunk while Clover retained it in memory. Configuration
is part of the result, not something to omit from the claim.

## Artifact and provenance gaps

The journals preserve commands, measurements and original artifact locations. A complete
relocatable bundle tying every headline result to exact source revisions, prompt inputs,
environment settings and raw outputs has not been assembled here. The original machine's
artifact paths are provenance references, not proof that those files are currently reachable.
No remote artifacts were fetched or benchmarks rerun during this organization work.

For the next campaign, retain both engines' commit IDs, build commands, hardware and storage
layout, prompt/token inputs, environment variables, cache state, per-run timings, correctness
outputs and artifact checksums together. Record unfavorable and inconclusive outcomes too.