# Practices & Field Lessons

Lessons from running [the Clover framework](../framework/04-framework.md) on real, high-stakes work. The
[philosophy](../framework/02-philosophy.md) covers what we believe. The [principles](../framework/03-principles.md) cover what
makes each stage hold up. This page records what happened when they met actual deadlines.

Each lesson records the situation, what became clear, and the practice it produced.

---

## Lesson 1 — Focus beats parallelism

AI can run many tasks at once. Human attention still cannot.

### From the field

AI agents and parallel sessions made it easy to run three initiatives at the same time — two
release-blocking production issues and a new feature for an upcoming launch. It felt highly
productive. Many things were moving, and there was visible activity everywhere.

The outcomes told a different story. One blocker was ultimately resolved better by someone else. The
other accumulated band-aid fixes instead of a real resolution. The new feature came out weak.

The problem was not AI capacity. **Context and human direction were spread too thin.** With attention
divided three ways, the Context behind each task stayed shallow and the Direction given to it stayed
vague, so the output was shallow too.

Dropping the parallelism and working one stream at a time finally resolved the hardest issue at its
root (see [the React memory leak case study](../../case-studies/02-react-rsc-memory-leak.md)).

### The practice

- Treat **human attention as the scarce resource**, rather than AI throughput. The ability to run many
  things in parallel is a temptation and not a mandate.
- Give one workstream real Context and focused Direction before starting the next.
- Watch for the **illusion of progress**. Motion across many tasks and outcomes are different things,
  and only one of them can be demonstrated.
- Beware **false economy**. Under-resourcing a task while also spreading attention thin costs more
  than it saves.

**Reinforces:** [Principle 3 — Execution applies AI capability to Context and
Direction](../framework/03-principles.md#3-execution-applies-ai-capability-to-context-and-direction). Run work in parallel only
where it is genuinely independent.

---

## Lesson 2 — Read the system before fixing it

A fix attempted before the problem is understood is a guess, however confident it sounds.

### From the field

On a hard bug, AI repeatedly jumped straight to attempted fixes, and repeatedly failed. Each attempt
was a guess at a solution before anyone knew what was actually happening.

Changing the approach changed everything: step back, gather what the system was really doing end to
end, add logging where the picture had holes, plan the change against that, implement it, and check
the outcome. With that sequence, the root cause was found and fixed quickly — far faster than the
fix-first thrashing that preceded it.

### The practice

- Work the problem in order, for AI and humans alike:
  1. Gather the context end to end.
  2. Instrument wherever the context is thin.
  3. Decide the change based on what the environment showed.
  4. Make the change.
  5. Check that the intended outcome actually happened.
  6. Write down what the attempt taught before starting the next one.
- When an agent starts **thrashing** on repeated failed fixes, that is the signal to stop fixing and
  go back for the information the attempts were missing.
- **A fix nobody can explain is not a diagnosis.** Ask for the reasoning before the change.

**Reinforces:** [Principle 1 — Context comes from the real
system](../framework/03-principles.md#1-context-comes-from-the-real-system) and [Principle 4 — Outcome is what
reality shows](../framework/03-principles.md#4-outcome-is-what-reality-shows).

---

## Lesson 3 — A workaround is not the destination

Stabilize the incident first, then keep going until the underlying problem is understood.

### From the field

In the React and Next.js memory-leak investigation, disabling stack-trace capture with
`--stack-trace-limit=0` stopped the production memory growth and stabilized the application. That was
the right immediate mitigation, and it was not the end of the investigation.

The workaround removed the symptom while also reducing useful error-stack observability. Continuing
the investigation revealed the deeper retention mechanism in React Server Components. That made it
possible to pursue an upstream fix instead of permanently carrying a local workaround.

The resulting contribution was submitted to React so the solution could potentially help other
applications facing the same class of problem.

### The practice

- **Mitigate first when production is at risk**, and label the mitigation as temporary while doing it.
- Ask what the workaround is **hiding or disabling**, and not only whether the incident stopped.
- Once the system is stable, return to the root-cause work with measured evidence.
- Prefer a **generalized fix** when the underlying defect belongs to a shared framework or dependency,
  and contribute it upstream where that is appropriate.

**Reinforces:** [Principle 4 — Outcome is what reality
shows](../framework/03-principles.md#4-outcome-is-what-reality-shows) and [Principle 1 —
Context comes from the real
system](../framework/03-principles.md#1-context-comes-from-the-real-system). A
workaround resolves an incident. Understanding the root cause can resolve a class of problems.

---

## Lesson 4 — You do not have to hold the context to be accountable for it

The knowledge is usually already in the system. Being out of date is an access problem.

### From the field

A defect arrived in a flow that the human responsible for it had never worked on. They had been
leading a team rather than writing code for some time, and features and flows had shipped in the
interim that they had simply never seen. On the old terms, the only options were to go and learn the
area first, or hand it to whoever last touched it.

Instead the orchestration layer was pointed at it. It read across the repositories involved, navigated
the running application to reproduce the reported behavior, and showed exactly where it occurred. The
fix followed, went through normal review, and was checked against the running system.

What is worth noticing is where the knowledge came from. The AI did not know the domain. **The domain
was in the repository and in the running application the whole time** — encoded in the code, the
configuration, and the behavior of the system itself. What changed was that it became reachable
without first being memorized by a person.

### The practice

- Treat "I am not current on this area" as a question about access rather than a reason to hand the
  work away. Ownership of the outcome stays where it was.
- Include the **running application** in what the orchestration layer can reach. A great deal of what
  a system does is only observable by using it, and reproducing a report is often faster than
  reasoning about whether it could happen.
- Ask for **the location and the evidence before the fix** — where the behavior occurs and how that
  was established. A patch offered without either is a guess with good formatting.
- Do not mistake this for AI understanding the domain. It read the domain from the system, which
  matters when the system is the thing that is wrong.
- Keep review proportionate. Being out of context is a reason to look harder at the evidence, and not
  a reason to approve faster because the explanation sounded confident.

**Reinforces:** [Principle 1 — Context comes from the real
system](../framework/03-principles.md#1-context-comes-from-the-real-system). Knowledge
held by a system stays available when the people who held it are not.

---

## Lesson 5 — Write the context down, or pay for it again

A session's memory dies with the session. A file in the repository does not.

### From the field

Two efforts ran in the same period, and the difference between them was not difficulty.

In the first, a memory-exhaustion investigation, findings were written into markdown files committed
alongside the code and updated as the work went: what was being pursued, what had been established,
what was still open, what had turned out to be wrong. Each pass began from the accumulated picture.
The investigation was long, and it converged.

In the second, work on a fix in an open-source library, that discipline was not applied. Nothing
persisted between passes. Without the accumulated context, the full flow was never understood, and the
work degenerated into repeated attempts at a fix — the same thrashing described in
[Lesson 2](#lesson-2--read-the-system-before-fixing-it). It recovered when the cycle was applied deliberately:
gather the context, state the direction, act, check the outcome, write down what was learned, and go
again.

The same person, the same tools, the same period. The variable was whether understanding was being
written down.

### The practice

- Keep a **working context file next to the code**, and update it while the work is happening. A
  write-up produced afterwards is a report. A file maintained during the work is memory.
- Record **what was ruled out and why**, not only what was found. Dead ends are the most expensive
  thing to rediscover.
- State **the goal, what is settled, and what remains**. Anyone resuming should be able to act without
  reconstructing the reasoning first.
- Treat **repeated failed attempts as a symptom of lost context**. An agent that cannot see what has
  already been tried will try it again, confidently.
- **Write the file back into Context before the next pass.** After each success and each failure, the
  context files are written before the next attempt, which is the only thing that makes the next pass
  cheaper than the last. That write is Growth, the fifth stage of the system cycle, and one corrected
  wrong answer is enough for it to count.

**Reinforces:** [Principle 1 — Context comes from
the real environment](../framework/03-principles.md#1-context-comes-from-the-real-system) and
[Context engineering](../framework/05-context-engineering.md#where-context-lives).

---

## Closing

These lessons share a root. AI multiplies whatever Context and Direction it is given. Teams that read
the system first and then say what they want get outcomes that hold up. Teams that skip the reading
get a lot of output and few outcomes.

Every lesson here came out of the fifth stage. Somebody ran a cycle, watched what it produced, and
wrote down what it taught. None of them needed repetition or scale before they were worth keeping.
