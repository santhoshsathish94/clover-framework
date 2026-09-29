# Orchestration Brief — &lt;Task or Outcome&gt;

> A one-page brief for running a single task through Clover:
> **Context → Direction → Execution → Outcome → Growth**
>
> Copy this file, rename it for the task, and fill each section as the work proceeds. The prompts are
> starting points for an AI assistant — adapt them to the tools at hand. Keep the brief short. The
> value is in real context, clear direction, and honest evidence, not in length.
>
> Nothing here assumes software. The system can be anything that keeps a record of what it did.

| Field | Value |
|---|---|
| **Task / outcome** | &lt;what are we trying to achieve?&gt; |
| **Owner (human)** | &lt;who is accountable for the outcome?&gt; |
| **AI capability** | &lt;which assistant, agents, tools?&gt; |
| **Date** | &lt;yyyy-mm-dd&gt; |

---

## 1. Context — *what the real systems show*

Start here. Gather what the problem actually turns on, from the systems the organization already
runs rather than from assumption. Keep this section current — every pass adds to it.

- **Known context:** …
- **Sources read (records, history, measurements, the system as it runs; for software: code, tickets, logs, telemetry, data, environments):** …
- **Open questions and untested assumptions:** …
- **Could not reach:** …
- **What earlier attempts showed, and what they ruled out:** …

> **Prompt:** "Before we agree on what to do, read &lt;sources&gt;. Summarize what you now know about
> how the system actually behaves, list what is still unknown, and say what you could not reach. For
> anything you could not reach, name the specific piece of information you need and ask me for it
> rather than assuming it."

**Ownership:** The human decides which systems may be read, which sources can be trusted, and when
there is enough to set the direction. AI does most of the gathering and reports the gaps.

---

## 2. Direction — *what needs to be done*

With the systems read, state the outcome worth reaching, the edges of the work, and what would show
it happened.

- **Problem / trigger:** …
- **Desired outcome (one sentence):** …
- **What would demonstrate it:** …
- **Out of scope / must not change:** …
- **Needs human approval:** …

> **Prompt:** "Here is the problem: &lt;describe&gt;. Given what you have just read, restate the
> outcome we actually want in one sentence, propose what would demonstrate that outcome in the real
> environment, and ask me about anything I have left out — especially what this work must not touch."

**Ownership:** The human controls what matters, the desired outcome, constraints, boundaries, and
what must not happen. Approval stays with the human. AI sharpens the wording, names conflicts
between stated goals, and asks about the parts nobody mentioned.

---

## 3. Execution — *how the work happens, and doing it*

Take the smallest coherent path to the outcome, with an owner on every step.

- **Approach chosen:** …
- **Steps, with owner (human / AI) per step:** …
- **Risks and dependencies:** …
- **Changes made:** …
- **Human decisions and approvals:** …

> **Prompt:** "Given that context, propose the smallest focused path to the outcome. For each step,
> say who should own it and why. Call out the risks. Then implement step &lt;n&gt; only, and stop
> before anything outside the agreed scope."

**Ownership:** AI determines how the work should happen and executes within those boundaries, and
comes back when the evidence contradicts the plan. The human sets those boundaries and approves what
needs approving.

---

## 4. Outcome — *what the environment showed*

What did the environment show actually happened? Record it whether it was favorable, unfavorable,
partial, inconclusive, or simply not what was expected.

- **Evidence (a measurement, a before-and-after, a trial outside the live system, a person confirming; for software: tests, telemetry, a run outside production):** …
- **What was checked, what was observed, and where this stopped:** … *(see [Outcome](../docs/framework/07-outcome.md#how-strong-is-your-evidence))*
- **Does the outcome from Direction now hold?** …
- **Human verification:** …
- **Still unverified:** …

> **Prompt:** "Evaluate whether the outcome we stated in Direction was achieved, using concrete evidence. Map each
> piece of evidence back to it. Say what you checked, what you observed, and where you stopped. List
> anything you could not verify."

**Ownership:** The human sets the standard for sufficient evidence and accepts or rejects it. AI runs
the checks and reports them accurately, including what did not work.

*If the evidence does not hold, go back to **Context**, rather than to Execution. A failed check usually
means something about reality was missing, and repeating the change only reaches the same place
faster.*

---

## 5. Growth — *what the cycle taught, carried back into Context*

Growth is the fifth stage, and a team performs it like the other four. Whatever the Outcome taught,
at any size, goes back into Context. One wrong answer, understood and written down, counts. Every
attempt produces context, whether it succeeded or failed, and that context is what the next attempt
reasons from.

> After each success and each failure, write the context files before the next attempt.

- **What this attempt showed, and what it ruled out:** …
- **Worth keeping (what was written down, the prompts, the checks and boundaries that held, the context files):** …
- **Where it is written, so the next cycle actually reads it:** …
- **Pattern or one-off?** … *(a single good outcome is an anecdote; a pattern has held across several cycles)*
- **Next direction this revealed:** …

> **Prompt:** "Summarize what this attempt showed and what it ruled out. Write it back into the
> Context section of this brief, and say where else it should live so the next cycle reads it. Say
> whether this is a pattern that has held before or a single result."

**Ownership:** The human decides what becomes durable knowledge and what has gone stale. AI writes it
while the details are still accurate.

---

*Part of [Clover](../README.md). Every pass adds context, and sometimes it changes the Direction
itself, because the work showed that a different outcome was the one worth having.*
