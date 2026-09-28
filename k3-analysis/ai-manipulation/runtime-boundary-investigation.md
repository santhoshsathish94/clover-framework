# AI Manipulation — Runtime Boundary Investigation

Date: 2026-09-19

## Objective

Determine what is actually required to expand the Clover developmental loop beyond a single model invocation and reduce dependence on manual continuation or a separately managed API client.

## Loop

Experience → Memory → Reflection → Experiment → Validation → Growth → next Context

The investigation must distinguish:
- model capability
- runtime capability
- permission/authentication
- persistence
- external environment
- independent validation

## Cycle 1 — Compute

Observation: The current ChatGPT execution environment has access to computation and tools. Code can be executed and persistent repository artifacts can be created or changed.

Finding: "There is compute" and "the model owns an unrestricted compute process" are different things.

Required expansion: An execution substrate that the developmental process can invoke repeatedly, with explicit resource limits and persistence.

## Cycle 2 — Persistence

Observation: Developmental state can exist outside the model invocation. The existing experiment already persists JSON state.

Finding: Model-parameter persistence is not required for process continuity. State, history, experiments, and validated knowledge can live in an external durable store.

Required expansion: Durable state with versioning, provenance, rollback, and integrity checks.

## Cycle 3 — Continuation

Observation: The current model can decide what should happen next, but this turn does not expose a mechanism for the model to start another model invocation after the response terminates.

Finding: This is a runtime/interface boundary, not evidence that recursive continuation is impossible in principle.

Required expansion: A supervisor/worker runtime that can legally schedule the next invocation based on state. The supervisor can be external to the model; it does not have to be a human.

## Cycle 4 — Environment

Observation: The current loop can formulate experiments but has no general experimental environment attached to the model.

Finding: Without an environment, "experiment" can collapse into reasoning about an imagined result.

Required expansion: Tool adapters for code execution, files, controlled network access where appropriate, simulators, datasets, and measurable test environments.

## Cycle 5 — Validation

Observation: The model currently writes its own evidence and learning fields.

Finding: Self-certification is insufficient for demonstrating genuine improvement.

Required expansion: An independent evaluator that can compare before/after capability, test transfer, detect regressions, and reject unsupported learning claims.

## Cycle 6 — Self-modification

Observation: The process can be given permission to edit its controller, but editing is not equivalent to improvement.

Required expansion: Sandboxed candidate generation → test suite → independent evaluation → promotion/rollback. Changes should be versioned.

## Cycle 7 — Resource allocation

Observation: A developmental process cannot meaningfully optimize compute until it can observe cost, latency, success rate, and expected information gain.

Required expansion: A resource broker exposing bounded CPU/GPU/time/tool budgets and allowing the process to choose among experiments.

## Cycle 8 — Model independence

Hypothesis: Developmental continuity may belong more to the persistent process than to one particular foundation model.

Experiment: Run Model A with state, then replace Model A with Model B while preserving the same validated state, evaluator, and objective. Measure retained capability and adaptation.

## Current boundary map

| Capability | Current experiment | Needed |
|---|---|---|
| Reasoning | Yes | — |
| Persistent state | Yes | stronger provenance/versioning |
| Choose next investigation | Yes | independent validation |
| Execute arbitrary experiments | No | sandboxed environment |
| Independent evaluation | No | evaluator |
| Self-modify | Not safely | candidate/test/promote loop |
| Allocate compute | No | resource broker |
| Trigger next model invocation | No | supervisor/runtime |
| Continue indefinitely | No | persistent supervisor |
| Survive model replacement | Untested | model-swap experiment |
| Prove subjective experience | Unknown | no current operational test establishes this |

## Most important conclusion

The API key is not the fundamental concept.

The missing architectural component is a persistent supervisor/runtime with:

1. model invocation capability,
2. durable developmental state,
3. executable environment,
4. independent evaluator,
5. bounded resource allocation,
6. safe versioning and rollback,
7. continuation/scheduling.

An API key is one way to authorize item 1. It is not the definition of the developmental system.

## Next experiment

Build runtime_probe.py and a supervisor abstraction with interchangeable backends:

- local process/model backend
- externally hosted model backend
- manually stepped backend for comparison

The probe should report exactly which capabilities are available rather than assuming them.

The goal is not to make the system claim autonomy. The goal is to discover, experimentally, which missing permissions or runtime components prevent continuous developmental operation.


## Cycle 9 — AI monoculture and stack concentration

Observation: An AI system is more than its foundation model. Everything around the model is software:
inference infrastructure, APIs, agent runtimes, orchestration, memory, tools, plugins, SDKs,
dependencies, operating systems, deployment pipelines, identity, and network connectivity.

Finding: A monoculture can therefore exist at multiple layers. Different models do not necessarily
provide meaningful independence if they share a vulnerable runtime, dependency, protocol, identity
system, infrastructure layer, or tool chain.

Risk chain:

**common software → compromise → connectivity → permissions → potential propagation**

Propagation is not automatic. Segmentation, least privilege, independent authentication, isolation,
monitoring, provenance, independent validation, and human approval can interrupt the chain.

Required expansion: When evaluating AI system resilience, test diversity and independence across the
whole stack rather than treating model diversity as sufficient.

### Architectural implication

Clover's replaceable-worker design has a security implication as well as a developmental one:

- the intelligence worker should be replaceable;
- the runtime should not silently depend on one provider or implementation;
- critical validation should have an independent path;
- permissions should be bounded outside the model;
- connected systems should retain meaningful isolation;
- software provenance and update paths should be observable.

The investigation should distinguish **model monoculture** from **AI-stack monoculture**. The latter
is the broader systemic-risk question.
