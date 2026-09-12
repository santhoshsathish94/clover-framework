# Clover AI

> Working in progress â€” direction for the next major Clover release

Clover AI is an open-source AI system intended to demonstrate how AI can be capable of producing meaningful outcomes while remaining accountable to the system in which it operates.

The objective is not simply to build a more capable model.

The objective is to explore how AI capability can coexist with enforceable system boundaries, verification, observability, and human accountability.

---

## Why Clover AI

The AI industry is moving from models that answer questions toward systems that can understand context, use tools, write and execute code, interact with infrastructure, and produce real-world outcomes.

That changes the engineering problem.

A model can be highly capable and still produce a poor outcome because:

- the context was incomplete
- the retrieved information was wrong or stale
- the direction was ambiguous
- the model misunderstood the intended scope
- a tool exposed too much authority
- execution happened without sufficient controls
- the outcome was never independently verified
- failures were not observable
- lessons from previous failures were lost

Clover AI is intended to address the complete system rather than treating the model as the entire solution.

The project asks:

> **How can we build an AI that is capable enough to produce meaningful outcomes while the system around it remains responsible, bounded, observable, and accountable?**

---

# The Clover AI Principle

> **The model can propose.  
> The system decides.  
> The tools enforce.  
> Evidence verifies.  
> Humans remain accountable.**

This is the central direction for Clover AI.

A capable model should not automatically become an authority simply because it can reason, plan, call tools, or operate autonomously.

The surrounding system must determine what the AI is allowed to know, what it is allowed to access, and what it is allowed to do.

---

# Clover AI and the Clover Cycle

Clover AI follows the fundamental Clover cycle:

```text
Context
   â†“
Direction
   â†“
Execution
   â†“
Outcome
   â†“
Growth
   â†“
Context
```

The cycle is not merely a workflow.

It defines how an AI-enabled system moves from understanding a situation to producing an outcome and learning from what actually happened.

Each stage has a responsibility.

---

# 1. Context

Context is what the system knows and what the AI is permitted to receive.

Clover AI should receive the context necessary to perform its task while respecting the boundaries of the system.

Context should be engineered rather than assumed.

The system should understand:

- what information is available
- where the information came from
- what information is relevant
- what information is stale
- what information is missing
- what information the AI is permitted to access
- what information should not be exposed

Context should not automatically be treated as truth.

A model can produce a fluent response from incorrect, incomplete, stale, or misleading context.

Therefore:

> **Context enables direction, but context is not automatically evidence.**

## Context Misuse

AI systems can fail not only because a model lacks capability, but because context is misused.

The risk becomes larger when AI systems are trained on, retrieve, summarize, or act upon information without sufficiently understanding:

- provenance
- permission
- ownership
- relevance
- freshness
- scope
- intended use

Clover should treat context as something that has boundaries of its own.

The question is not simply:

> "Can the AI access this information?"

It is also:

> **"Should the AI have access to this information, for this purpose, in this context?"**

This distinction matters when AI is used across organizations, codebases, customer data, internal knowledge, production systems, and public information.

---

# 2. Direction

Direction defines what the system is trying to accomplish.

A human or another accountable system establishes the objective.

The AI may help interpret, plan, decompose, and execute that objective, but it should not become the owner of the objective merely because it is capable of pursuing it.

**Direction remains accountable.**

This creates an important distinction:

```text
Human / accountable system
          â†“
       Direction
          â†“
       AI system
```

The model should not silently redefine the objective based on its own interpretation.

Where ambiguity materially affects the outcome, it should be surfaced rather than silently resolved through assumptions.

The purpose of Clover AI is therefore not to remove humans from responsibility.

It is to make meaningful AI assistance possible while preserving responsibility where it belongs.

> **AI capability may scale, but accountability cannot be delegated to the model.**

---

# 3. Execution

Execution is where AI capability interacts with the real system.

Clover AI may:

- reason about a task
- create a plan
- inspect information
- request tools
- modify permitted resources
- run permitted commands
- generate code
- test changes
- iterate based on results

But the model requesting an action is not the same as the system authorizing that action.

This distinction is fundamental.

```text
Model
  â†“
Requests action
  â†“
Tool / policy layer
  â†“
Checks permission and boundary
  â†“
Executes or rejects
  â†“
Produces evidence
```

The model should not be the final enforcement layer for its own authority.

---

# System Boundaries

A core objective of Clover AI is to make system boundaries **enforceable**, rather than merely describing them in prompts.

A prompt can tell an AI:

> "Do not access production."

That is an instruction.

It is not an enforcement mechanism.

A stronger architecture makes production inaccessible to the execution environment unless an explicitly authorized path exists.

For example, an AI that has permission to modify a development repository should not automatically have permission to:

- access production databases
- retrieve credentials
- modify unrelated repositories
- access arbitrary files
- make unrestricted external network requests
- deploy infrastructure
- delete persistent data
- modify security controls
- change its own permissions

The exact boundaries will depend on the system.

The principle remains:

> **A prompt can describe a boundary.  
> An architecture can enforce one.**

---

# Capability Is Not Authority

Clover AI should explicitly separate capability from authority.

A model may be capable of performing an operation without being authorized to perform it.

For example:

```text
Capability:
    Can generate SQL capable of deleting data.

Authority:
    Not permitted to execute destructive SQL.
```

Another example:

```text
Capability:
    Can generate a deployment command.

Authority:
    Development environment only.
```

Another:

```text
Capability:
    Can inspect files.

Authority:
    Only files within the assigned workspace.
```

This separation should exist at the system level.

The AI should not be expected to reliably enforce the boundary that determines its own authority.

---

# Tool Boundaries

Tools should have explicit capabilities.

A useful starting model is:

```text
READ
WRITE
EXECUTE
EXTERNAL
```

These capabilities can then be constrained by:

- environment
- resource
- repository
- namespace
- user
- task
- network
- time
- approval requirements
- data sensitivity
- reversibility
- blast radius

A tool should expose only what is necessary for the task.

The objective is not maximum tool access.

The objective is:

> **The minimum authority required to accomplish the intended outcome.**

---

# 4. Outcome

Outcome is what actually happened in the real system.

This is different from what the model believes happened.

An AI can say:

> "The issue has been fixed."

That statement is not proof that the issue has been fixed.

The system should seek evidence.

Depending on the task, evidence may include:

- compiler results
- unit tests
- integration tests
- browser tests
- database state
- API responses
- logs
- metrics
- generated artifacts
- schema validation
- security checks
- deployment status
- external system state

Therefore:

> **A model's statement about an outcome is not the outcome itself.**

Clover AI should make this distinction explicit.

---

# Verification

Verification should happen as independently from model belief as practical.

Where a deterministic mechanism can establish a fact, use it.

Examples include:

```text
AST
LSP
grep
git
compiler
tests
schema introspection
log extraction
browser automation
system metrics
database queries
```

The principle is:

> **If the answer is computable, compute it.**

The model should not spend reasoning capacity deriving facts that a deterministic tool can establish more reliably.

The objective is not to replace AI reasoning.

The objective is to use AI reasoning where reasoning provides value and deterministic systems where determinism is available.

---

# Model Verification

Model-based verification may be useful where deterministic verification cannot fully establish correctness.

However, model verification should not become the only verification mechanism.

A useful hierarchy is:

```text
Deterministic verification
        â†“
System evidence
        â†“
Independent model verification
        â†“
Human verification where required
```

Where possible, the verifier should not share all of the same blind spots as the model that produced the result.

The system should also record verification failures.

A failed verification is valuable information.

---

# 5. Growth

Growth is what the system learns from outcomes.

Growth should not mean blindly giving the AI more autonomy.

It should mean understanding:

- what worked
- what failed
- why it failed
- what context was missing
- what context was misleading
- what tools were insufficient
- what boundary was insufficient
- what verification caught
- what verification failed to catch
- where human intervention was required
- whether the model was actually capable of the task

This produces the next Context.

```text
Outcome
   â†“
Evidence
   â†“
Failure / success analysis
   â†“
Growth
   â†“
Better Context
```

The cycle continues.

---

# Context Engineering

Clover AI treats context engineering as a first-class engineering problem.

The goal is not to put as much information as possible into the context window.

The goal is to provide the right information.

For software engineering workloads, this may include:

```text
Source code
    â†“
AST / syntax
    +
Lexical information
    +
Semantic information
    +
Symbol relationships
    â†“
Ranking
    â†“
Graph expansion
    â†“
Reranking
    â†“
Context assembly
```

Code should generally be preserved verbatim where exactness matters.

Summarization should not silently replace source material when the exact source is required for correct execution.

The system should also be capable of identifying when it does not have sufficient information.

> **Insufficient information is a valid system state.**

Clover AI should not be forced to produce an answer merely because a model can produce one.

---

# Retrieval Is Not Truth

Retrieval can improve the information available to the model.

It does not automatically make that information correct.

A retrieval system may return:

- stale information
- incomplete information
- irrelevant information
- contradictory information
- incorrectly ranked information

Therefore:

```text
Retrieval
   â‰ 
Truth
```

The system should preserve the distinction between:

**what was retrieved**

and

**what has been verified.**

---

# Tools Over Token Consumption

Clover AI follows a simple engineering principle:

> **If the answer is computable, compute it.**

Examples:

Instead of asking the model to infer a symbol relationship:

```text
Use AST / LSP.
```

Instead of asking the model to guess whether tests pass:

```text
Run the tests.
```

Instead of asking the model to infer the current Git state:

```text
Use git.
```

Instead of asking the model to reconstruct a log pattern:

```text
Extract the log data deterministically.
```

AI should spend its capability where reasoning provides value.

---

# Agentic AI

Clover AI may operate as an agent.

But agentic behaviour should not be defined as unrestricted autonomy.

A useful agent should have:

- a clear objective
- bounded context
- bounded tools
- bounded permissions
- explicit stop conditions
- verification
- observable actions
- escalation paths
- human accountability

The question is not:

> "How autonomous can the AI become?"

The more useful question is:

> **"What level of autonomy can the system safely and reliably support for this particular task?"**

---

# Security and Credentials

The model should not directly receive credentials wherever the architecture can avoid it.

A preferred architecture is:

```text
AI model
   â†“
Tool request
   â†“
Tool server
   â†“
Credential / authorization layer
   â†“
External system
```

The tool infrastructure owns credentials.

The model receives the result necessary for the task rather than the secret required to obtain it.

Development execution and production observation should also be separated where practical.

---

# Observability

Clover AI should make its operation observable.

Where appropriate, the system should be able to reconstruct:

```text
What did the AI receive?
        â†“
What direction was given?
        â†“
What did the AI request?
        â†“
Which tools executed?
        â†“
What permissions were applied?
        â†“
What actually happened?
        â†“
What evidence was produced?
        â†“
Was the outcome verified?
```

Production AI needs traces, not merely screenshots.

Observability is not only useful for debugging.

It is part of accountability.

If the system cannot reconstruct what happened, investigating a failure becomes significantly harder.

---

# Evaluation

Clover AI should be evaluated on real outcomes.

The evaluation should include real engineering tasks with machine-checkable pass conditions wherever possible.

The evaluation set should also include tasks where the correct behaviour is to identify insufficient information rather than fabricate an answer.

Metrics may include:

- task solve rate
- verified task success rate
- malformed tool calls
- unnecessary tool calls
- retries
- human interventions
- tokens per successful task
- wall-clock completion time
- verification failures
- agent trajectory efficiency
- cost per successful task
- human minutes per successful task

The most important metric is not simply:

> "How intelligent does the model appear?"

It is:

> **"How reliably can the complete system produce a verified outcome?"**

---

# Model Selection

Clover AI should not assume that the largest model is automatically the best model.

The project should evaluate models based on the work they can actually complete within the intended system.

A model may be selected based on:

- engineering task performance
- tool-use reliability
- context handling
- reasoning capability
- latency
- memory requirements
- infrastructure cost
- licensing
- deployment constraints
- verification outcomes

The objective is not to win a benchmark.

The objective is to find the smallest amount of model capability and infrastructure that can reliably produce the required outcome.

---

# Model Routing

Clover AI should not assume that every task requires the largest available model.

Different tasks may require different capability levels.

A possible routing architecture:

```text
                    Task
                     â†“
                  Router
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”
            â†“        â†“        â†“
          Small    Medium    Large
          model    model     model
            â†“        â†“        â†“
              Verification
                    â†“
                  Outcome
```

The router may initially use deterministic rules.

As evaluation data grows, routing decisions can potentially become data-driven.

The objective is not to maximize model size.

It is to maximize useful verified outcomes for the available cost and infrastructure.

---

# Infrastructure

Clover AI should investigate the relationship between model capability and infrastructure rather than assuming that one serving architecture is universally correct.

Possible deployment approaches include:

- CPU inference
- GPU inference
- quantized inference
- model routing
- small models
- larger models where necessary
- local storage
- storage-streamed inference where technically appropriate
- RAM-resident versus streamed working sets
- Kubernetes-based deployment
- horizontal scaling

These should be treated as engineering hypotheses to measure, not assumptions to defend.

The objective is:

> **Find the smallest amount of compute, model capability, context, and tooling required to reliably complete a real task.**

---

# Storage-Streamed Inference

Clover should investigate whether some model architectures can make useful inference possible without requiring the entire model checkpoint to remain resident in RAM.

Large sparse models may have working-set characteristics that make alternative serving architectures worth investigating.

A storage-streamed approach may involve:

```text
Model checkpoint
       â†“
Fast local storage
       â†“
Selected weights / experts
       â†“
RAM working set
       â†“
CPU / GPU computation
```

This is an experimental infrastructure direction.

It should not be presented as a universal solution or as proof that any arbitrary large model can run on very small hardware.

The important engineering question is:

> **Can storage streaming reduce resident-memory requirements enough to make a useful workload viable without making latency, throughput, and cost unacceptable?**

Experiments should compare:

1. RAM-resident inference
2. RAM + local NVMe/SSD caching
3. storage-streamed weights or experts

The same workload, context policy, precision, and verification process should be used when comparing configurations.

Measurements should include:

- checkpoint size
- peak RSS
- load time
- storage throughput
- IOPS
- read amplification
- first-token latency
- tokens per second
- task completion time
- cache hit rate
- working-set residency
- CPU utilization
- concurrency
- successful engineering tasks
- cost per verified successful task

The objective is not:

> "Can the model run?"

The objective is:

> **"Can the system complete useful work at acceptable cost and latency?"**

---

# Infrastructure Scaling

Clover AI should scale only after evidence demonstrates that scaling is necessary.

A practical progression is:

```text
Hosted validation
       â†“
Existing Linux Kubernetes pods
       â†“
Storage-streamed experiments where applicable
       â†“
Larger Linux pod
       â†“
GPU-backed Linux pod
       â†“
Single stable deployment
       â†“
Horizontal scaling
```

The project should avoid purchasing or provisioning large infrastructure before demonstrating that the workload justifies it.

---

# Kubernetes

Where Kubernetes is used, AI inference and agent/tool execution should remain separate services.

A conceptual architecture is:

```text
                User / Application
                        â†“
                 Clover Agent
                        â†“
              Tool / Policy Layer
                        â†“
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â†“                               â†“
   Model Service                    External Tools
        â†“                               â†“
   Model Inference                System Resources
```

The model service should not own credentials or unrestricted system authority.

GPU scheduling should be handled by the infrastructure layer rather than by the model.

Heterogeneous GPU environments should be represented through infrastructure configuration and scheduling policies.

---

# Agent and Model Separation

The inference service and the agent/tool harness should be separate components.

The model should produce structured requests.

The agent should:

- validate requests
- enforce tool permissions
- execute tools
- collect results
- provide results back to the model
- record the trajectory
- apply stop conditions
- initiate verification

This separation allows the model to change without changing the complete security and execution architecture.

It also allows different models to be evaluated against the same agent environment.

---

# Permission Model

Clover AI should eventually define a machine-enforceable permission model.

A permission should answer questions such as:

```text
Who?
What?
Where?
When?
Why?
Under which task?
With what limits?
With what approval?
With what reversibility?
```

For example:

```text
Agent:
    Clover development agent

Resource:
    repository/example-project

Capability:
    READ + WRITE

Environment:
    development

Network:
    restricted

Production:
    DENIED

Credential access:
    DENIED

Destructive operations:
    APPROVAL REQUIRED
```

The exact implementation may evolve.

The important principle is that the boundary should exist independently of the model's willingness to follow instructions.

---

# Failure Handling

Clover AI should treat failure as a first-class outcome.

When a task fails, the system should attempt to determine whether the cause was:

- insufficient context
- incorrect context
- incorrect retrieval
- ambiguous direction
- model capability
- tool failure
- permission failure
- infrastructure limitation
- verification failure
- human decision
- external system behaviour

This classification is important because different failures require different improvements.

A larger model is not necessarily the correct solution to a context problem.

A better prompt is not necessarily the correct solution to a permission problem.

More compute is not necessarily the correct solution to a verification problem.

---

# Human Intervention

Human intervention should not automatically be treated as system failure.

Some operations may appropriately require human approval.

The system should measure where intervention is necessary and whether that requirement can be reduced safely through better engineering.

Possible approval boundaries include:

- destructive operations
- production deployment
- sensitive data access
- external communication
- security changes
- irreversible operations
- high-blast-radius changes

The objective is not to eliminate humans.

The objective is to make human involvement meaningful and accountable.

---

# Responsible Capability

The goal is not to make AI incapable in order to make it safe.

The goal is to investigate whether capability and responsibility can be engineered together.

A useful AI system should be capable of meaningful work.

At the same time, the system surrounding it should establish:

```text
What it knows
      â†“
What it is asked to do
      â†“
What it is allowed to do
      â†“
What it actually did
      â†“
What happened
      â†“
What can be verified
```

This creates a distinction between:

**capability**

and

**authority**.

Clover AI seeks to increase the former without blindly expanding the latter.

---

# Open Source and Attribution

Clover AI will be developed as an open-source project.

Clover recognizes that modern AI is built upon an enormous body of work created by researchers, engineers, organizations, and open-source communities.

The project should respect that foundation.

Where Clover uses or builds upon existing work, it should:

- respect the applicable license
- provide required attribution
- preserve required notices
- clearly distinguish upstream work from Clover work
- document important dependencies
- avoid claiming community contributions as original Clover work
- contribute improvements back where appropriate

This principle applies to:

- model architectures
- model weights
- datasets
- evaluation frameworks
- inference engines
- agent frameworks
- libraries
- developer tools
- infrastructure
- research
- community implementations

> **Clover AI exists because many people made the underlying ecosystem possible. Their work should be respected, acknowledged, and built upon responsibly.**

---

# What Clover AI Is Not

Clover AI does not claim to have solved AI safety or alignment.

It does not claim that:

- AI systems cannot fail
- guardrails eliminate risk
- models can perfectly understand boundaries
- autonomy is inherently good
- a particular model is universally superior
- verification can catch every failure
- AI can be made perfectly safe
- more capability automatically produces better outcomes
- AI model intelligence alone determines system reliability

Clover AI is an engineering effort.

The claims should follow the evidence.

---

# What Clover AI Should Prove

Clover AI should eventually demonstrate, through reproducible experiments, that an AI system can:

1. Understand sufficient context for a real task.
2. Follow accountable direction.
3. Operate through bounded tools.
4. Respect system-enforced permissions.
5. Produce meaningful work.
6. Detect when information is insufficient.
7. Recover from failures where possible.
8. Produce observable execution traces.
9. Verify outcomes using evidence.
10. Improve through measured feedback.

The project should publish both successful and unsuccessful experiments.

A successful demonstration is useful.

A failure that exposes an architectural weakness may be even more useful.

---

# The Core Question

The long-term question behind Clover AI is:

> **Can increasing AI capability coexist with increasing responsibility?**

Clover explores whether responsibility can be engineered into the system surrounding AI capability rather than relying entirely on the model to behave responsibly.

The model may become more capable.

The tools may become more capable.

The infrastructure may become more capable.

The system may eventually support increasingly autonomous work.

But the fundamental principle remains:

> **AI capability may scale, but accountability cannot be delegated to the model.**

---

# Current Status

This document describes the direction for Clover AI.

It is a work in progress.

Implementation details, model selection, infrastructure requirements, evaluation methodology, security architecture, tool interfaces, and deployment strategy should be validated through engineering experiments before being treated as final decisions.

Clover AI should not claim that something works before it has been built and measured.

> **Build it.  
> Measure it.  
> Verify it.  
> Understand the failure.  
> Improve it.  
> Repeat.**

---

# Clover AI

**Capability with boundaries.**

**Meaningful outcomes with verification.**

**Open development with respect for the work that made it possible.**

**AI that operates within a system â€” not above it.**