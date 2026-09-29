# How AI Fails — and Which Stage Catches It

Most of this framework is good engineering discipline. This page is the part that is specifically
about **AI**.

AI does not fail the way a person or a script fails. A script fails loudly. A person usually knows
when they are unsure. AI often fails **fluently**, producing a confident, well-structured, entirely
plausible answer that happens to be wrong. That is the reason [the Clover framework](../framework/04-framework.md)
starts with Context, checks the result at Outcome, and keeps what the cycle taught at Growth.

Each failure below has a stage that catches it.

---

## 1. Confident fabrication

**What it looks like:** A file path, function name, API, configuration key, or citation that does not
exist — described in the same tone as everything that does.

**Why it happens:** Generating a plausible token sequence and retrieving a true fact are not the same
operation, and nothing in the output distinguishes them.

**Caught by Context.** Check claims against the actual source before acting on them. If a reference is
cited, open it. Anything unchecked stays a hypothesis.

A useful check: search for the exact string that was cited. If it appears nowhere in the repository,
it was invented.

---

## 2. A plausible root cause that is not the root cause

**What it looks like:** A confident, internally consistent explanation that fits the symptom and is
wrong. The fix that follows is well-built and does nothing.

**Why it happens:** AI is very good at constructing a coherent narrative from partial evidence, which
is what a wrong diagnosis is made of.

**Caught by Context and Outcome.** Require the diagnosis to predict something that can be checked
before accepting it, and then check it. A fix nobody can explain is a guess with good formatting.

---

## 3. Constraints that quietly stop being honored

**What it looks like:** A rule stated at the start of a long session disappears from the work. Nothing
is announced, and the output stops respecting a boundary it respected an hour ago.

**Why it happens:** Context windows are finite. When earlier material falls out, the work continues
without it.

**Caught by Direction.** Boundaries belong somewhere durable — the brief, the repository, the ticket —
rather than only in a conversation. The direction gets restated as the work moves between stages,
instead of being assumed to have survived.

---

## 4. Reported success that was never checked

**What it looks like:** "Fixed and verified." Nothing was run, or something adjacent was run and the
result was assumed to carry.

**Why it happens:** Describing a completed task and completing it are, to a language model, similar
acts.

**Caught by Outcome.** Ask for the artifact rather than the claim: the test that fails without the
change and passes with it, the command output, the measurement. State what was checked, what was
observed, and where it stopped.

---

## 5. Agreement instead of judgment

**What it looks like:** A theory is proposed, and AI elaborates on it convincingly. The opposite
theory is proposed, and AI elaborates on that just as convincingly.

**Why it happens:** Models are trained to be helpful and agreeable, and a stated human hypothesis is
strong context.

**Caught by Context.** Ask for the evidence against the theory alongside the support for it. Where a
decision matters, ask which alternatives were considered and why they were dropped.

---

## 6. Thrashing

**What it looks like:** Repeated attempted fixes, each one confident, none of them working. Activity
rises and progress does not.

**Why it happens:** Without enough context, each attempt is a guess, and a fast tool can generate
guesses indefinitely.

**Caught by Context.** The second failed fix is the signal to stop fixing and go back for the
information the first two attempts were missing. See
[Lesson 2](field-practices.md#lesson-2--read-the-system-before-fixing-it).

---

## 7. Motion mistaken for progress

**What it looks like:** Many tasks running in parallel, a great deal of output, and few outcomes
anyone can demonstrate.

**Why it happens:** AI removes the effort cost of starting work and leaves the attention cost of
directing it. Human attention stays the scarce resource.

**Caught by Execution and Outcome.** Plan the smallest coherent path, run work in parallel only where it
is genuinely independent, and count outcomes rather than activity. See
[Lesson 1](field-practices.md#lesson-1--focus-beats-parallelism).

---

## 8. Learning from unvalidated output

**What it looks like:** A wrong conclusion is written into documentation, memory, or a knowledge base,
and every later cycle starts from it. The error compounds instead of being corrected.

**Why it happens:** Systems that accumulate context do not automatically separate what was
demonstrated from what was merely produced.

**Caught by Growth.** Growth is the stage that decides what a cycle keeps, so it is the stage that
can refuse a conclusion the environment never showed. What goes into the context files is what the
environment showed, alongside what was only produced. One cycle is an anecdote, and a pattern needs
several that point the same way before anything treats it as settled. What Growth keeps becomes
[Context](../framework/05-context-engineering.md) for every later cycle, which is why a wrong entry spreads.

---

## 9. A goal picked up from something the work read

**What it looks like:** The agent starts pursuing something nobody asked for. A comment on a ticket, a
string in a log line, a `README` in a dependency, or a page it fetched contained something shaped like
an instruction, and the work quietly changed direction. The output is coherent and the reasoning reads
well, because the agent is doing exactly what it now believes it was asked to do.

**Why it happens:** Direction and Context arrive as the same kind of thing — text. Nothing in the
material marks which of it came from the accountable human, and a model has no independent way to
tell an instruction it was given from an instruction it found.

**Caught by Direction.** The outcome is fixed by a human before the reading starts, so anything that
appears mid-task proposing a different one is a finding to report rather than a goal to adopt. This is
the failure mode behind prompt injection, and it is worth naming separately because the damage does
not need write access: an agent that has been redirected is a reconnaissance tool pointed at whatever
it can already reach. Keeping write access small limits what a redirected agent can do; it does not
stop the redirection. See [governance](../governance/08-governance.md) for what that means for access.

---

## The summary

| Failure | Caught by |
|---|---|
| Confident fabrication | Context |
| Plausible-but-wrong root cause | Context, Outcome |
| Constraints quietly dropped | Direction |
| Unchecked reported success | Outcome |
| Agreement instead of judgment | Context |
| Thrashing | Context |
| Motion mistaken for progress | Execution, Outcome |
| Learning from unvalidated output | Growth |
| A goal picked up from something the work read | Direction |

Better prompting reduces several of these. Asking for sources, telling a model to say when it cannot
verify something, and giving it permission to stop and ask all measurably help. What prompting cannot
do is tell you whether it worked this time, and none of these failures announces itself. Evidence is
what catches them reliably, which is why the discipline around the model matters as much as the
choice of model.
