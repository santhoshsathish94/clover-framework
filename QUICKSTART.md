# Quickstart

Run a first Clover cycle on a real task in about 15 minutes.

Clover is a way of working with the reality and the actors in it — System, Human and AI — through the system cycle
they run. Every system that worked has worked this way. What AI did was move execution somewhere
accountability could not follow, and Clover puts it back with the human actor who can take up the
role.

There is nothing to install. The cycle runs on real work, with whatever tools and AI assistant are
already in use.

> **Context → Direction → Execution → Outcome → Growth**

## What this cycle adds

Most AI work today runs on three stages. A human gives the Direction, AI performs the Execution from
whatever that human typed in, and after several passes the result becomes Outcome. It works, and it
is ordinary.

This cycle starts with Context instead. The assistant reads the real systems before anyone describes
them, and the Direction is given against what is there. That pairing is what the rest of the
framework is about.

## 1. Pick a real task (2 min)

Something small and real works best — a fault, a recurring complaint, a small change, a question
nobody can answer quickly. It does not have to be software. Any system that keeps a record of what
it did will do. The cycle needs concrete work to run on.

## 2. Give the assistant something real to read (3 min)

Hand over a source instead of describing the problem from memory: whatever the system itself keeps.
Its records, its history, its measurements, the thing as it is running. For software that means the
repository, the failing test, the log lines, the ticket, the query results, the application in a
development environment, and a read-only MCP server in front of that one system is enough. Whatever
the system, the access is scoped to what the human driving the work already has, at the privileges
they already hold. One source is enough to start.
[What to connect, and in what order →](docs/guides/orchestration-environment.md#building-one)

## 3. Say what you are trying to get done (2 min)

Now that the assistant can see the systems, tell it the outcome, not only the task. What would count
as done, what must not happen, and what it may read. A good assistant asks for all three before it
changes anything, and it asks for a specific missing piece rather than guessing when a system cannot
be connected.
[What we ask of agents →](AGENTS.md)

Then point. Say which part changed last week, which step has always been fragile, where to look
first. The systems are a haystack, and a pointer from someone who works in them every day is worth
more than a longer description of the task.

## 4. Copy the orchestration brief (1 min)

Take [`templates/orchestration-brief.md`](templates/orchestration-brief.md), copy it, and rename it
for the task. That one page is the working artifact for the whole cycle, and it lives beside the
work rather than in a chat window.

## 5. Walk the five stages (6 min)

For this optional guided exercise, fill each section in order. Clover does not require this template or a fixed stage order in ordinary work. Every stage in the brief carries a **copy-and-paste prompt** for an AI
assistant and a note on **who owns what**.

| Stage | Leave the stage with… |
|---|---|
| **Context** | What the real systems showed, and the gaps you could not reach named |
| **Direction** | A one-sentence outcome, what must not happen, where the answer probably is, and what would demonstrate it |
| **Execution** | The smallest focused path, an owner per step, and the change itself |
| **Outcome** | Evidence that maps back to the outcome, and what it does not cover |
| **Growth** | What the pass taught, written back into Context, and where the next cycle will read it |

> **At Outcome, say what you checked, what you observed, and where you stopped.** Reaching the
> strongest available evidence every time is not the goal. Being accurate about what you have is.
> See [Outcome](docs/framework/07-outcome.md#how-strong-is-your-evidence).
>
> If the evidence does not hold, go back to **Context**, rather than to Execution.

## 6. Growth — write back what the pass taught (1 min)

Growth is the fifth stage, and a human performs it like the other four. Whatever the Outcome taught,
at any size, goes back into Context. One wrong answer, understood and written down, counts. No
repetition and no scale are needed for it to be real.

Every pass adds context, and that includes the passes that failed. Before the next attempt, write
what the attempt showed and what it ruled out back into the Context section of the brief.

Save the filled-in brief with the work, wherever the work is kept. Because it is written down, the
next agent or the next human can pick the work up without redoing the investigation.

## See a full example

[`examples/production-exception-remediation/`](examples/production-exception-remediation/) runs a
recurring production `500` through the five stages, with the prompts used at each one. It is a
software scenario because that is the one most readers will recognize. Swap in another scenario,
from any kind of system, and follow the same path.

## If you remember one thing

Get the context from the real systems before acting, and be accurate about what the evidence showed.
The environment decides whether the outcome happened, and the file left behind decides how much the
next cycle costs.

New here? Start with [the problem this solves](docs/framework/01-problem.md) and
[the Clover framework](docs/framework/04-framework.md). Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md).
