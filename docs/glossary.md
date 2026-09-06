# Glossary

Plain-language definitions for the terms used across this framework. If a term here is not clear,
that is a bug in the writing — [tell us](../CONTRIBUTING.md).

## Clover and the five stages

| Term | What it means here |
|---|---|
| **Clover** | This framework. Clover is a way of working with System, Human, and AI to produce meaningful outcomes through real Context, human Direction, delegated Execution, validated Outcome, and the Growth each cycle carries into the next. See [the framework](04-framework.md). |
| **System** | The reality the work happens in. Outcomes are the changes that get applied to it. |
| **Actors** | The two participants in any cycle — the **human**, who holds Direction and accountability, and **AI**, which supplies capability and execution. They work inside the **system**, which is the reality the outcome has to exist in. |
| **System cycle** | The five stages the actors run — **Context → Direction → Execution → Outcome → Growth**. |
| **Stage** | One part of the system cycle. Each stage has one job. There are five. |
| **Accountability** | Being answerable for the outcome afterward. It cannot sit with AI, which can perform work and report on it but cannot carry the consequence. When execution moved to AI, accountability tended to move out of scope with it. Clover establishes it back in the system, on the human actor who can truly take up the role. |
| **Leaf** | A leaf of the clover mark. The picture has five. In the documents the parts of the cycle are called stages; the fifth leaf represents Growth. |
| **The common clover** | Three leaves: Direction, Execution, Outcome. A common way of working where context is mostly what a human provides. |
| **The lucky clover** | Four leaves. Context arrives from System reality and comes first, changing what the other three are worth. |
| **The growth clover** | Five leaves. Growth is the fifth stage: preserve what the cycle taught and promote what has held. |
| **Context** | What the work reasons from, and where the cycle starts. It may be an existing system or the reality already established while a system is being built: data, behavior, history, constraints, evidence, and previous cycles. |
| **Direction** | Where human purpose and accountability enter the system. Humans choose what matters, the desired outcome, priorities, constraints, boundaries, and what must not happen. AI can be capable enough to suggest directions, but humans should always have the authority to decide what to pursue. |
| **Execution** | Where the outcome is pursued by working with the system, using the Direction decided and the Context available. AI determines and carries out how the work should happen, and the system's boundaries bind both actors — neither the human nor AI may violate them. |
| **Outcome** | What actually happened, as the real system shows it. It is both the failure and the success; an unfavorable Outcome is still an Outcome. A closed task, a passing build, or a confident report sits outside this on its own. |
| **Growth** | The fifth stage of the system cycle. Whatever the Outcome taught, at any size, carried back into Context. One wrong answer, understood and written down, counts; no repetition and no scale are required. A system that does not retrospect its growth will not produce better outcomes. What accumulates over time can sit with humans, AI systems, the systems being worked on, teams and organizations. What frontier AI providers do with volumes of interaction data is a larger and separate question — see [the hypothesis layer](../hypothesis/ai-future.md). |
| **The fifth leaf** | Growth, drawn as the fifth leaf of the mark. The leaf also carries the unknown boundary of how far capability and learning may develop. |

## Everything else

| Term | What it means here |
|---|---|
| **Orchestration** | Coordinating people, AI, tools, and context so that work produces an outcome the environment confirms, rather than each part doing its own thing. |
| **Orchestration environment** | The access layer between AI and the current systems an organization uses. Read-only connections to whatever those systems keep — their records, their history, their measurements, the working environments. It is what feeds the Context stage. See [the orchestration environment](orchestration-environment.md). |
| **MCP server** | The software route into a system: a small service that gives an agent a scoped way to read one of them — a repository, a datasource, a log store, an environment. Read-only, and scoped to what the human driving the work already has access to. Systems that are not software need the same thing by some other means. |
| **Capability** | Anything that can do work: a human, a team, an AI model, an agent, a tool, a system. Capability does not by itself grant authority or accountability. |
| **Intent** | What a human actually wants to achieve. The outcome, and not the task. |
| **Output** | What got produced — a file, a patch, a report. Distinct from the **Outcome**, which is the change in the real world that was wanted. Work can produce output and reach no outcome. |
| **Evidence** | What was actually done to check a claim — an assertion, one manual look, a repeatable test, a before-and-after measurement, or the original signal gone from the real environment. Say which. See [Outcome](07-outcome.md#how-strong-is-your-evidence). |
| **Experience** | What was learned from one cycle — what was tried, what happened, what the evidence showed. It can become part of the context files beside the work. |
| **Expertise** | A reusable pattern that emerges from several *validated* experiences. One cycle is not expertise. |
| **Ownership** | Within a task or piece of work, the named human accountable for the outcome they direct. Work can be delegated; ownership cannot. This does not transfer accountability for a model, product or deployment away from the organisation that builds, releases or operates it. |
| **Agent** | An AI system that can take actions and use tools, rather than only produce text. |
| **Agentic workflow** | A designed loop of agent steps that repeats a known process. Orchestration differs in that it keeps what the outcome taught it. |
| **Autonomy** | In Clover, this refers only to how much of the *path* AI is allowed to determine inside human Direction. It never means ownership of purpose, acceptable risk, priorities, boundaries, or accountability. A more capable model does not create authority over the destination. |
| **Delegated execution** | The amount of operational work a human or organization chooses to have AI perform inside human Direction. It can expand or contract by context and evidence. It does not transfer purpose, acceptable risk, priorities, boundaries, the destination, or accountability to AI. |
| **Telemetry** | The signals a running system emits about itself. In software: logs, metrics, traces, error rates. In any other system: whatever it records about its own operation while it runs. |
| **Blast radius** | How much damage a change could do if it is wrong. A bigger blast radius means more human approval. |
| **Non-production** | Any setting that is not the live system — a copy, a test bench, a rehearsal, and in software local, test and staging. It can still contain sensitive data and is not automatically safe to expose. |
| **Guardrail** | A constraint that keeps work inside safe boundaries: a required approval, a scope limit, a check that must pass. |
| **Boundary** | A limit on the work set by human Direction — scope, approval points, what must not happen. A boundary can be widened or narrowed by the accountable human. |
| **Obligation** | A duty that comes from law, regulation, or an organization's own rules. It sits outside Clover's boundaries, applies to the system and to both actors, and no one inside a cycle can trade it away. Adopting Clover neither satisfies it nor removes it. See [governance](08-governance.md#the-rules-each-actor-works-inside). |
| **Root cause** | The underlying reason a problem occurs. Distinct from the symptom, and from the workaround that hides it. |
| **Workaround** | Something that stops the pain without fixing the cause. Legitimate for stabilizing an incident, and not a destination. |
| **Thrashing** | Repeated confident attempts at a fix, none of which work. The signal to stop fixing and go back to Context. |
