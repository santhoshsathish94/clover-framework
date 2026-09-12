# Clover AI

> Working in progress — direction for the next major Clover release.

Clover AI is an open-source implementation direction for building AI systems that can produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

This file is not intended to explain the Clover Framework.

For the framework itself, start here:

- https://cloverframework.com/
- `README.md`
- `docs/04-framework.md`
- `docs/03-principles.md`

---

## Clover AI — Core Direction

> **The model can propose.  
> The system decides.  
> The tools enforce.  
> Evidence verifies.  
> Humans remain accountable.**

Clover AI should demonstrate that increasing model capability does not require giving the model unrestricted authority.

The implementation should keep the Clover cycle operational:

```text
Context → Direction → Execution → Outcome → Growth → Context
```

The model is one component of the system, not the system itself.

---

## 1. Context

Context must be engineered from the real system.

Clover AI should prioritize:

- relevant system context
- provenance and freshness
- permission-aware retrieval
- exact source material where required
- context sufficiency checks
- deterministic context extraction where possible

For software engineering, investigate:

```text
Source
  ↓
AST / syntax
  + lexical information
  + semantic information
  + symbol relationships
  ↓
Ranking / graph expansion / reranking
  ↓
Context assembly
```

> **Insufficient information is a valid state.**

Retrieval improves available information; it does not make retrieved information true.

Relevant reference: `docs/05-context-engineering.md`

---

## 2. Direction

Direction must remain owned by an accountable human or accountable system.

Clover AI may interpret, plan, decompose, and execute a direction, but it must not silently redefine the objective or acquire authority simply because the model is capable of doing something.

> **AI capability may scale, but accountability cannot be delegated to the model.**

Relevant references:

- `docs/02-philosophy.md`
- `docs/03-principles.md`
- `docs/04-framework.md`

---

## 3. Execution

AI execution should happen through explicit tools and system-enforced permissions.

The model should request an action; the surrounding system should decide whether the action is allowed.

Initial tool capability classes:

```text
READ
WRITE
EXECUTE
EXTERNAL
```

Permissions should be constrained by the task, resource, environment, network, data sensitivity, reversibility, and blast radius.

The objective is not maximum tool access.

> **Give the minimum authority required to accomplish the intended outcome.**

The model should not be the final enforcement layer for its own authority.

---

## 4. System Boundaries

Boundaries must be enforceable outside the model.

A prompt saying `do not access production` is an instruction, not an enforcement mechanism.

Clover AI should investigate architectures where:

- development and production are separated
- credentials remain outside the model
- tools own authorization and secrets
- destructive operations can require approval
- network access is restricted
- workspace and repository access are scoped
- the agent cannot change its own permissions

Preferred pattern:

```text
AI
 ↓
Tool request
 ↓
Policy / authorization
 ↓
Tool execution
 ↓
Evidence
```

---

## 5. Outcome and Verification

Outcome means what actually happened in the real system — not what the model says happened.

Clover AI should prefer deterministic verification wherever possible:

- compiler
- tests
- AST / LSP
- git
- schema inspection
- database state
- API responses
- logs
- metrics
- browser/system checks

> **If the answer is computable, compute it.**

Where deterministic verification is insufficient, independent model verification may be used, followed by human verification where required.

A verification failure is a useful outcome and should be retained for Growth.

Relevant references:

- `docs/07-outcome.md`
- `docs/08-governance.md`

---

## 6. Growth

Growth should come from measured outcomes and failures.

The system should learn:

- what worked
- what failed
- why it failed
- whether context was sufficient
- whether the model was capable of the task
- whether tools or permissions were insufficient
- what verification caught or missed
- where human intervention was required

Growth should improve the next Context, not simply increase autonomy.

---

## 7. Agentic Engineering

Clover AI may use agentic workflows for real engineering tasks.

Agents should have:

- clear objectives
- bounded context
- bounded tools
- bounded permissions
- stop conditions
- observable actions
- verification
- escalation paths

The goal is not maximum autonomy.

> **The useful question is what level of autonomy the system can safely and reliably support for a particular task.**

Agent and model execution should remain separable so different models can be evaluated against the same environment.

---

## 8. Deterministic Tools Over Token Consumption

AI reasoning should be used where reasoning provides value.

Use deterministic systems for facts they can establish more reliably.

Examples:

```text
Symbol relationships → AST / LSP
Test status           → test runner
Git state             → git
Log patterns          → log extraction
Schema state          → schema inspection
```

This is a core Clover AI engineering principle.

---

## 9. Model Strategy

Clover AI should not assume that the largest model is the best model.

Evaluate models on complete task outcomes, not model reputation alone.

The model ladder should be experimentally evaluated across:

- small coding SLMs
- 7–8B class models
- North Mini / similar agentic models
- larger models where required

Selection criteria should include:

- verified task success
- tool-use reliability
- context handling
- reasoning capability
- latency
- memory requirements
- infrastructure cost
- licensing
- deployment constraints

> **Use the smallest amount of model capability and infrastructure that can reliably produce the required outcome.**

Model routing should eventually allow different capability levels for different tasks.

---

## 10. Infrastructure Direction

Infrastructure should be discovered through measurement rather than assumed in advance.

Initial progression:

```text
Hosted validation
    ↓
Existing Linux Kubernetes pods
    ↓
Larger CPU / GPU deployment where evidence requires it
    ↓
Owned stable deployment
    ↓
Horizontal scaling
```

Clover AI should investigate:

- CPU inference
- GPU inference
- quantized inference
- Kubernetes scheduling
- model routing
- RAM-resident inference
- storage-streamed inference

### Storage-streamed inference

The Kimi K3 systems work demonstrates an important hypothesis worth testing: parameter count does not by itself determine the required resident memory if the architecture and serving system allow a working set to be streamed from storage.

This is an experiment, not an assumption.

Compare:

```text
RAM-resident
RAM + local NVMe/cache
Storage-streamed
```

Measure:

- peak RSS
- storage throughput / IOPS
- cache hit rate
- load time
- first-token latency
- tokens/sec
- task completion time
- concurrency
- successful verified tasks
- cost per verified task

The question is not merely whether a model can run.

> **Can it complete useful work at acceptable latency, reliability, and cost?**

---

## 11. Evaluation

Clover AI should be evaluated on real engineering outcomes.

Build a reproducible evaluation set with machine-checkable pass conditions wherever possible, including tasks where the correct result is to identify insufficient information.

Track at minimum:

- verified task success rate
- task solve rate
- malformed tool calls
- unnecessary tool calls
- retries
- human interventions
- verification failures
- wall-clock time
- tokens per successful task
- trajectory efficiency
- cost per verified successful task
- human minutes per successful task

Do not claim capability before the system is built and measured.

> **Build it. Measure it. Verify it. Understand the failure. Improve it. Repeat.**

---

## 12. Observability

Every meaningful agent run should be reconstructable where practical:

```text
Context received
    ↓
Direction given
    ↓
Model decisions / requests
    ↓
Tools executed
    ↓
Permissions applied
    ↓
Actual system changes
    ↓
Evidence
    ↓
Verification
    ↓
Outcome
```

Production AI needs traces, not merely screenshots.

Observability is part of accountability, not only debugging.

---

## 13. Open Source and Attribution

Clover AI will be open source and should respect the work that makes it possible.

For models, datasets, research, inference engines, libraries, infrastructure, and other upstream work:

- respect the applicable license
- provide required attribution
- preserve required notices
- distinguish upstream work from Clover work
- document important dependencies
- contribute improvements back where appropriate

> **Build on the work that made this possible without pretending the foundation came from us.**

---

## 14. What This Work Should Prove

Clover AI should eventually demonstrate, through reproducible engineering evidence, that an AI system can:

1. receive sufficient context
2. operate under accountable direction
3. use bounded tools
4. operate within system-enforced permissions
5. produce meaningful engineering work
6. recognize insufficient information
7. recover from failures where possible
8. expose observable execution
9. verify outcomes with evidence
10. improve from measured outcomes

Successful and unsuccessful experiments should both be documented.

---

## Relevant Clover Documentation

The Clover AI implementation should be developed alongside the existing framework rather than duplicating it.

Start with:

- **Website:** https://cloverframework.com/
- **Framework:** `docs/04-framework.md`
- **Principles:** `docs/03-principles.md`
- **Philosophy:** `docs/02-philosophy.md`
- **Context Engineering:** `docs/05-context-engineering.md`
- **Outcome:** `docs/07-outcome.md`
- **Governance:** `docs/08-governance.md`
- **AI Responsibility:** `docs/ai-responsibility/README.md`
- **AI Future Hypothesis:** `hypothesis/ai-future.md`

These documents contain the framework and supporting reasoning. This file should remain focused on the implementation direction for Clover AI.

---

## Status

**Work in progress.**

Model choice, serving architecture, infrastructure, tool interfaces, evaluation methodology, security boundaries, and deployment strategy remain experimental until validated by implementation and measurement.
