<p align="center">
  <a href="https://cloverframework.com/">
    <img src="assets/social-preview.png" width="820"
         alt="Clover Framework — System, Human and AI working together for meaningful outcomes">
  </a>
</p>

<p align="center">
  <strong>Clover Framework</strong> · System, Human and AI working together for meaningful outcomes
</p>

<p align="center">
  <a href="https://cloverframework.com/"><strong>cloverframework.com</strong></a>
  ·
  <a href="QUICKSTART.md">Quickstart</a>
  ·
  <a href="AGENTS.md">AGENTS.md</a>
  ·
  <a href="docs/README.md">Docs</a>
</p>

---

# Clover

Clover is a framework for working with System, Human, and AI to produce meaningful outcomes.

The framework is intentionally documented on the website and in the `docs/` directory rather than reproduced here.

**Start here:** https://cloverframework.com/

For the framework itself, see:

- [`docs/02-philosophy.md`](docs/02-philosophy.md)
- [`docs/03-principles.md`](docs/03-principles.md)
- [`docs/04-framework.md`](docs/04-framework.md)
- [`docs/05-context-engineering.md`](docs/05-context-engineering.md)
- [`docs/07-outcome.md`](docs/07-outcome.md)
- [`docs/08-governance.md`](docs/08-governance.md)
- [`docs/09-adoption.md`](docs/09-adoption.md)
- [`docs/README.md`](docs/README.md)

---

# Using Clover

Clover does not require installation or replacement of existing tools.

For a practical starting point:

- [`QUICKSTART.md`](QUICKSTART.md) — first cycle
- [`AGENTS.md`](AGENTS.md) — operating guidance for AI agents
- [`case-studies/`](case-studies/) — real work and measured outcomes
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — corrections, gaps, and contributions

---

# Clover AI

Clover AI is the next major implementation direction of the framework.

It is intended to be an open-source AI system that can produce meaningful engineering outcomes while operating inside explicit system boundaries.

The implementation should demonstrate:

> **The model can propose.  
> The system decides.  
> The tools enforce.  
> Evidence verifies.  
> Humans remain accountable.**

## Core direction

```text
Context → Direction → Execution → Outcome → Growth
```

Clover AI should:

- use real system context rather than assumptions
- keep Direction with an accountable human or accountable system
- execute through bounded tools
- separate capability from authority
- enforce permissions outside the model
- keep credentials outside the model where possible
- prefer deterministic tools when facts are computable
- verify outcomes against real system evidence
- make agent execution observable
- learn from failures and verification results
- recognize insufficient information as a valid state

### Context

Context engineering is a first-class part of the implementation.

For software engineering, investigate structured context from source, AST/LSP, lexical and semantic information, symbol relationships, ranking, graph expansion, and reranking.

> **Retrieval is not truth.**

### Direction

Direction remains human-owned and accountable.

AI may analyze, recommend, plan, and execute, but capability does not grant authority to decide what should be pursued.

### Execution and boundaries

AI should request actions through tools. The surrounding system should enforce permissions.

Initial tool boundaries should distinguish:

```text
READ
WRITE
EXECUTE
EXTERNAL
```

Development and production access, credentials, network access, destructive operations, and workspace scope should be explicitly controlled.

> **A prompt can describe a boundary. An architecture can enforce one.**

### Outcome and verification

The model saying that something worked is not evidence that it worked.

Prefer deterministic verification through tests, compilers, AST/LSP, Git, schemas, logs, metrics, APIs, databases, and other system evidence. Use independent model verification only where deterministic verification is insufficient.

> **If the answer is computable, compute it.**

### Agentic engineering

Clover AI may use agentic workflows, but autonomy should remain bounded by:

- objective
- context
- tools
- permissions
- stop conditions
- verification
- observability
- escalation

The goal is not maximum autonomy. The goal is reliable, verified outcomes within appropriate boundaries.

### Model and infrastructure strategy

Do not assume the largest model is the best model.

Evaluate models on complete engineering outcomes, including verified task success, tool use, context handling, latency, memory, cost, licensing, and deployment constraints.

Investigate model routing and the smallest capable model for each task.

Infrastructure should be discovered through measurement, including:

- CPU and GPU inference
- quantized inference
- Kubernetes deployment
- RAM-resident inference
- RAM + local NVMe/cache
- storage-streamed inference
- horizontal scaling

Storage-streamed inference is an experimental direction. The Kimi K3 systems work provides a useful hypothesis that parameter count does not by itself determine resident memory when the architecture and serving system allow a working set to be streamed from storage.

Compare architectures using the same workload and measure RSS, storage throughput/IOPS, cache behaviour, latency, tokens/sec, task completion time, concurrency, verified success, and cost per verified task.

### Evaluation and observability

Clover AI should be evaluated on real engineering tasks with machine-checkable outcomes where possible.

Track at minimum:

- verified task success rate
- task solve rate
- malformed and unnecessary tool calls
- retries
- human interventions
- verification failures
- wall-clock time
- tokens per successful task
- trajectory efficiency
- cost per verified successful task
- human minutes per successful task

Meaningful agent runs should be reconstructable from context and Direction through model requests, tool execution, permissions, system changes, evidence, verification, and final Outcome.

> **Build it. Measure it. Verify it. Understand the failure. Improve it. Repeat.**

---

# Clover AI — work in progress

The current implementation direction is documented in:

- [`work-inprogess/clover-ai.md`](work-inprogess/clover-ai.md) — Clover AI direction
- [`docs/05-context-engineering.md`](docs/05-context-engineering.md) — context engineering
- [`docs/07-outcome.md`](docs/07-outcome.md) — outcome and evidence
- [`docs/08-governance.md`](docs/08-governance.md) — governance and accountability
- [`docs/ai-responsibility/`](docs/ai-responsibility/) — AI responsibility work

Model choice, serving architecture, infrastructure, tool interfaces, security boundaries, and evaluation methodology remain experimental until validated by implementation and measurement.

Clover AI will also respect the work that makes the project possible: applicable licenses, attribution, upstream notices, research, models, datasets, libraries, inference engines, infrastructure, and open-source contributions should be acknowledged and preserved appropriately.

---

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
