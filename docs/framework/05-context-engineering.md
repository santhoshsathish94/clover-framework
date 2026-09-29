# Context

Context is one of the related activities in [the Clover framework](04-framework.md), listed first for explanation rather than as a mandatory entry point. It covers what the system needs to know about reality before acting. What Growth keeps from an Outcome becomes Context for subsequent work.

The system already exists before anyone asks for anything, or there is an established reality from which a new system is being built. Context is therefore more than what the human remembers to hand over. It grounds Direction and can be revisited when Execution or Outcome exposes something previously unknown.

In the common clover, context is only what one human can hand over. That is more than what they type — they can attach files, or point at the repository they happen to be working in — and it stays bounded by what that one human can reach and remember. In the lucky clover it becomes the current systems the organization uses: every repository with its many projects and documentation, the datasources the applications connect to, the logs and telemetry, the deployment environments, and the running applications. Tests, history, whatever was already tried, and what earlier passes wrote down sit here too. For a system being built, requirements, designs, dependencies, decisions, experiments, current behavior, constraints, and evidence are part of the same reality.

Capability is only as good as the material it reasons from. A human cannot solve a problem they have never looked at, and an agent cannot either, however good its tools are.

## System first: reality before assumptions

Clover's priority is the reality and the **actors** who work in it. The system is the reality; the actors in it are the human and AI. The system is the grounding reality. Humans then set Direction and remain accountable. AI uses that system Context and human Direction to determine how work should happen and to execute within those boundaries.

Context is therefore not merely context about an AI task. It is the context of the system itself — including what it is becoming when the system is under construction.

## Reason from the real environment

Where the environment can answer a question, read the environment. That is the working rule for this stage, and the rest of this page is how it plays out.

An AI model reasoning from assumptions returns something that reads correctly and describes a system nobody has. The failure is quiet, because a confident description of the wrong architecture looks the same as a confident description of the right one until somebody checks. Checking is usually cheap. The code, the logs, the tests, and the running system are all reachable, and reaching them costs far less than a change built on a guess.

Prompting is a small part of this stage. A carefully worded request against thin context still produces a guess.

## What context is made of

Different problems need different material. These are the sources worth checking before anyone concludes there is nothing to read:

- **The system as written.** Source, architecture, configuration, dependencies, and the tests that describe how it is meant to behave.
- **The system as it runs.** Logs, telemetry, traces, runtime state, error signals. What the system does, rather than what the documentation says it does.
- **The rules around it.** Business rules, constraints, regulatory requirements, and the decisions that produced the current design.
- **The history.** Tickets, incidents, commits, reviews, and earlier attempts at the same problem, including the ones that went nowhere.
- **What earlier passes kept.** Recorded experience, evidence that held up, dead ends worth not repeating, and patterns that have already proved out across several cycles. Every pass writes into this group, and the next pass reads from it.

Organizations usually hold most of this already. The work is retrieval and assembly more often than it is authorship.

## How the material gets reached

This part is a setup, not a principle.

1. **Stand up read-only MCP servers** in front of the repositories, the datasources, the logs and telemetry, and the environments, so an agent can read them directly rather than being told about them.
2. **Scope every connection to what the human driving the work already has access to**, at the privileges they already hold. Nothing new is being granted.
3. **Start with one environment. Development is enough.** Widen to other non-production environments as it proves out.

That access already exists and is already used, often with nobody tracking it. Setting it up this way makes it deliberate, scoped and visible, and it surfaces stale credentials, unreviewed access paths and data nobody has looked at, before any of those become an incident.

[The orchestration environment](../guides/orchestration-environment.md#building-one) covers what to connect and in what order. [Governance](../governance/08-governance.md#questions-your-security-team-will-ask) answers the questions a security team will ask.

## Enough to reason correctly

Two failures sit on either side of this stage.

Thin context is the familiar one. The problem gets solved in the abstract, the answer is coherent, and it lands somewhere next to the actual system. Hoarding is the other. A context packed with everything reachable buries the handful of facts the problem turns on, and a long context is not a substitute for a relevant one.

The bar is enough context to reason correctly about this problem. Reaching it is iterative: gather, try to explain the problem end to end, notice which step cannot be explained, then go and get that.

Gaps are worth naming out loud. "The deployment history for this service was not reachable" changes what the next step should be, and a gathering pass that reports only what it found hides the part that matters most. Assumptions nobody has tested belong in the same list.

## Who holds what

Access and judgment stay with a human: which systems may be read, which data may be used, which sources can be trusted, and when the context in hand is good enough to act on. [Governance](../governance/08-governance.md) covers how that access is held, and [the orchestration environment](../guides/orchestration-environment.md) covers what it takes to make the material reachable at all.

AI does most of the gathering. Reading code, tracing a call path, pulling telemetry, reconstructing an incident from logs, finding the attempt from a year ago that nobody remembered. This is the stage where a capable agent saves the most time, and it saves it on work people were skipping anyway.

AI can also identify missing context and ask for it. That does not make AI the owner of what matters; it makes AI useful at finding the information needed to act within human Direction.

## Where context lives

Context that survives the cycle needs somewhere to sit, or it stays a good intention. The mechanism that works is unglamorous: **plain markdown files committed alongside the code they describe.**

One per effort, or per repository, updated as the work proceeds rather than written up at the end. A useful file answers four questions:

| | |
|---|---|
| **The goal** | What outcome is being pursued, and what "done" means |
| **What has been established** | Findings that are settled, and the evidence behind them |
| **What remains** | Open questions, unverified assumptions, and next steps |
| **What was learned** | Things that turned out to be wrong, dead ends worth not repeating |

This matters more with AI in the loop than without it. A session ends and its working memory goes with it. A file in the repository stays. Anyone picking the work up — a colleague, a different agent, the same agent tomorrow — starts from what has already been established instead of re-deriving it, and the re-derivation is where both the cost and the drift come from.

Two properties make it work:

- **It sits next to the code**, so it travels with the change, gets reviewed with the change, and cannot quietly disagree with a wiki nobody opens.
- **It is written continuously.** Context recorded at the end is a report. Context recorded as the work proceeds is a working memory, and that is the difference between resuming and restarting.

Every pass adds context, and a failed pass adds as much as a successful one. The rule is plain:

> After each success and each failure, the context files are written before the next attempt.

What one cycle established is what the next cycle starts from. Judgment about what is worth keeping and where a recorded finding has gone stale stays with a human, because a wrong finding written down spreads and somebody has to notice.

Writing that file is Growth, the fifth stage of the system cycle, and what it keeps becomes Context. That is how the loop closes, and it happens without changing the underlying AI model: repeated observations can expose system patterns, missing assumptions, and useful practices that improve future cycles. What accumulates may sit with humans, teams, organizations, systems, and AI systems in different ways, and none of it transfers Direction.

## What happens there

This stage catches most of the AI-specific failures. Fabricated references, a root cause that is plausible and wrong, and agreement offered where judgment was needed all come from acting on material that was never checked. [How AI fails](../guides/how-ai-fails.md) covers each pattern and why it happens.

It is also where a failed check lands. When [Outcome](07-outcome.md) shows that the intended outcome did not occur, what that taught is kept and the cycle returns here rather than to Execution, because the usual reason a change failed is that something about reality was missing. The next attempt needs new information: what the environment did instead, which assumption broke, which signal nobody had looked at yet.
