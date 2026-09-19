# The Problem

Clover is a way of working with the reality and the actors in it. The system is the reality; the
actors in it are the human and AI. What they run is the system cycle:
**Context → Direction → Execution → Outcome → Growth**.

None of that is a new way of working. Every system that worked has worked this way: somebody
understood the situation, somebody decided what mattered and answered for it, the work got done,
reality showed what happened, and what it taught carried into the next attempt.

AI does not change that cycle. It makes every stage easier, better and faster, and all the actors
grow with it. What AI did change is accountability. Execution moved to something that cannot be
accountable, and accountability went out of scope with it. Clover establishes accountability back in
the system, through the human actor who can truly take up the role.

## AI already does expert work

AI reads a codebase it has never seen and explains how the system behaves. It reviews a change the
way an experienced engineer reviews one. It designs test cases from how the system actually works
rather than from the wording of a ticket. It reads a diff for the class of mistake that causes
security incidents. During an incident it reconstructs what happened from logs and telemetry, fast
enough to be useful while the incident is still open.

Those answers come out of analysis. AI traces a call path through unfamiliar code, reads a stack
trace against the real source to work out which branch ran, and follows a symptom through the logs
into the data underneath. When somebody goes and checks the answer, it usually holds.

So capability is not what most teams are short of. Skilled people are available, and capable AI is
available too. Something else decides whether the work turns into an outcome anyone can rely on.

## The context is still one human's handover

Almost everywhere AI is used today, everything it knows about the system it is changing arrives from
one human. That is more than typing. They can attach files, or point at the repository they happen
to be working in. It is still bounded by what that one human can reach and remember, put together in
a hurry, about a codebase nobody has held in their head for years.

The ceiling on the work is the ceiling on what one human can hand over.

We kept seeing the same pattern at every level:

- Somebody starts fixing before knowing what is broken, and the fix lands near the problem rather
  than on it.
- A team adds people to move faster. What each of them knows travels badly, so the work turns into
  handoff, lost context, and rework.
- An organization gives everyone AI. A few people get good at it, the rest carry on as before, and
  the capability never becomes shared.
- An agent makes a change, the change fails, and it tries a variation of the same change. Nothing
  was missing except information.

Each one is the same failure. Capability without direction, real information, evidence, and anything kept from last time rarely
turns into a reliable outcome. AI does not create that gap. It makes the gap obvious, because AI
will act on missing information faster and more confidently than a human will.

That is the problem Clover addresses.

## Accountability went out of scope

Underneath every symptom above is one loss. In a system that worked, the human who did the work
could be asked about it afterward. They could explain the reasoning, say what they had assumed, and
answer for the result. Execution and accountability traveled together, and mostly nobody had to
think about it.

AI broke that pairing. It can do the work, describe the work convincingly, report that the work
succeeded, and answer for none of it. When execution moved, accountability tended to move with it —
straight out of scope. What is left is output nobody owns.

Our ways of working were built for humans working with humans, and they assume the pairing. A handoff
assumes a human on both ends. A review assumes an author who can explain their reasoning. "Done"
assumes somebody looked. An AI participant breaks all three at once, and nothing in the process
notices.

Clover's answer is to establish accountability back in the system, on the human actor who can truly
take up the role. AI takes its place as an actor inside the system cycle rather than as a replacement
for it. None of that requires slowing AI down.

The symptoms look the same wherever it happens:

* The same request produces a different answer each time.
* The work runs on stale or missing information.
* Confident answers turn out to be invented.
* Nobody trusts the output enough to ship it without redoing it.
* Nothing shows that the intended outcome actually happened.
* Ownership dissolves the moment the work is delegated.
* AI stays a private assistant instead of becoming part of how the organization operates.

## Better tools do not close it

Organizations usually respond by writing better prompts, adopting a newer AI model, or switching
tools. Individual productivity often improves. The larger problem stays open, because none of those
choices decide where the work should go, what the system needs to know about the real environment,
whether the result held up, or what the organization keeps afterward.

Picking an AI model is a small decision next to designing how the work happens. A capable
participant, human or AI, needs the information the problem actually requires, boundaries it
respects, ownership that survives delegation, and something at the end that shows the outcome
occurred.

## The organization already holds the context

The material that would tell AI what it needs to know exists already. Every repository, with its
many projects and the documentation kept for each application. The datasources the applications
connect to. The logs and telemetry. The deployment environments. The running applications
themselves.

Nobody has to write any of that out first. It is there, and it is current, which is more than can be
said for most descriptions of it. Reaching it is what the [Context stage](05-context-engineering.md)
covers, and it is the stage that changed what the other four are worth.

Clover organizes that into five questions a team asks every time:

- What do we need to know about reality before acting?
- What needs to be done, what must not happen, and who answers for the result?
- What should we do, and how should the work happen?
- Did reality validate the intended outcome?
- What did that teach, and where is it written down for the next pass?

The [five stages](04-framework.md) are those questions, in that order. Reality comes first, and the
direction is given against what is actually there. What each pass establishes is written down and
becomes the context the next one starts from.

## Where the evidence comes from

Software engineering is where Clover was built and where all of its evidence comes from, and the
[case studies](../case-studies/README.md) are engineering ones. The problem underneath is not a
software problem. Any work where human intent has to reach a capable system, and where somebody has
to know afterward whether it worked, runs into the same gap.



# Systemic Risk: AI Monoculture Is a Stack Problem

AI monoculture is not only about many organizations using the same model. The more important
question is how much of the surrounding software stack becomes common.

An AI deployment is surrounded by software: inference services, APIs, agent runtimes, orchestration
layers, memory stores, tool adapters, plugins, SDKs, dependencies, operating systems, deployment
pipelines, identity systems, networks, and update mechanisms.

A weakness in one shared component can therefore become a correlated weakness across otherwise
independent AI deployments.

The risk chain is:

**common software → compromise → connectivity → permissions → potential propagation**

Propagation is not automatic. Network segmentation, least privilege, independent authentication,
sandboxing, monitoring, model and stack diversity, independent validation, and human approval can
break the chain. The point is that common dependencies create correlated failure modes that
independent systems would not necessarily share.

### Why model diversity alone is insufficient

Two organizations can use different foundation models while still depending on substantially the same
agent framework, protocol, cloud infrastructure, SDK, identity layer, deployment system, or tool
connector.

That means resilience requires looking at the **whole AI stack**, not only the model.

Clover therefore treats diversity as an architectural property:

- different model implementations where practical;
- different runtime and orchestration components where practical;
- independent validation paths;
- constrained permissions and isolated environments;
- explicit provenance for models, software, tools, and updates;
- boundaries between connected systems;
- the ability to replace a component without rebuilding the entire system around it.

The objective is not to eliminate shared infrastructure. It is to avoid making a shared component an
unnoticed single point of systemic failure.

> **AI monoculture is a systemic dependency problem, not merely a model problem.**

This is one reason Clover separates the intelligence worker from the surrounding developmental runtime.
A replaceable worker allows the surrounding system to retain continuity while changing the model or
other components. The same principle can be applied in the opposite direction: the runtime itself
should not become dependent on one common implementation when that dependency creates correlated
risk.
