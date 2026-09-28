# Self-Hosting Small Language Models

**A plan for running open-weight models on infrastructure an organization controls, wired into VS Code, inside the Clover cycle.**

**Status: designed, not built. Nothing here has been observed working.**

**Written 11 September 2026.** Every model fact below has a shelf life measured
in weeks. Check versions, sizes, and serving flags against current vendor
documentation before acting on any of it.

---

## Why this document exists

Renting a model and renting a machine are not the same dependency, and
collapsing them makes this decision harder than it is.

Calling someone else's model means the weights can change under you, the terms
can change under you, and the context you send is context you no longer hold
alone. Running weights you chose, on capacity you rent inside your own
tenancy, gives up none of that. The model is fixed until you move it, the data
path is yours, and the only thing rented is the metal.

So the first move is not to buy hardware. It is to run open weights on capacity
the organization already pays for, and find out whether the capability is good
enough to be worth owning at all.

This document sets out what that takes, for one narrow domain, without
pretending the result is equivalent to a frontier model.

It is a plan. It is written down so it can be argued with before money is spent.

---

## Scope

Deliberately narrow. Widening this later is a decision, not a default.

**In scope:** web development in TypeScript, SQL and database work, deployment
manifests, written documents, and application logs.

**Out of scope:** every other language, mobile, data science, and anything
that is not one of the five above.

The scope matters because it removes work. A narrow scope means one engineering
model rather than several, no vision pipeline, and an evaluation set that can
actually be written by one human in a week.

---

## How this maps onto the cycle

The architecture is not arbitrary. Each part of it belongs to a stage.

| Stage | What holds it here |
|---|---|
| **Context** | Retrieval over the repository, schema, work items, and logs — served as explicit tools |
| **Direction** | The human, plus instruction files that carry standing constraints |
| **Execution** | One engineering model, running inside the VS Code agent loop |
| **Outcome** | The type checker, the test suite, the browser, and a second model from a different family |
| **Growth** | Measured, incremental improvement — the evaluation set first, then each addition kept only if it moved the numbers |

The most common way to get this wrong is to build Execution first, because it
is the part that feels like the product. Context and Outcome are what decide
whether Execution is worth anything.

**Growth here is deliberately small.** A hypothesis about where AI is going is
not a reason to switch everything at once. Growth moves one step at a time, and
a step counts whether it advances the capability far or barely at all — what
makes it Growth is that it was measured and then kept. Clover holds from the
simplest case to the most complex, and the steps in this plan sit near the
simple end. They are Growth even so.

---

## What runs

### Models

Every model below is open weight. The engineering and verifier models are
mixture of experts with roughly three billion active parameters. Their vendors
describe this as the quality of a thirty-billion-parameter model at the
generation speed of a three-billion-parameter one. That claim is the reason
this is affordable at all, and it is a vendor claim rather than a measured one.

| Role | Model | Quantized size | Context | License |
|---|---|---|---|---|
| Engineering | `qwen3.6:35b` | 23 GB | 256K | Apache 2.0 |
| Verifier | `north-mini-code-1.0` | 19 GB | 256K or 488K | Apache 2.0 plus vendor acceptable-use policy |
| Brief assembly | `qwen3.5:4b` | 3.4 GB | 256K | Apache 2.0 |
| Embedding | `qwen3-embedding:4b` | 2.5 GB | 40K | Apache 2.0 |
| Reranker | `Qwen3-Reranker-4B` | approx. 3 GB | — | Apache 2.0 |

The verifier's context window is reported inconsistently by its own vendor —
the model card says 256K, the distribution registry says 488K. Measure it
before designing around either figure.

**One engineering model covers every language in scope.** Splitting by
language is the most common design error here. Modern coding models are
trained multilingually and evaluated on multilingual benchmarks precisely
because single-language specialization stopped being useful. Three language
models would be the same weights three times with different prompts.

**The verifier is deliberately from a different family.** If the model that
writes the code and the model that reviews it share weights, they share blind
spots, and the review approving is not confirmation — it is the same model
agreeing with itself. That reads as verification and is not.

### Hardware

Two GPUs, each with at least 48 GB of memory.

| Card | Contents | Used |
|---|---|---|
| One | Engineering model | 23 GB |
| Two | Verifier, brief assembly, embedding, reranker | approx. 28 GB |

Editor utility work — titles, commit messages, intent detection, and inline
completions — stays on the existing GitHub Copilot subscription rather than
occupying a card.

**This runs on rented capacity first.** The organization's code already sits in
a rented cloud, so a GPU virtual machine in the same tenancy crosses no
boundary that has not already been crossed. Owning physical hardware is a later
decision, and one that should be made on evidence produced by the rented setup
rather than in advance of it.

### Serving

Serve with vLLM, not a desktop runner. The reasons are continuous batching, an
OpenAI-compatible API, Prometheus metrics, and above all **automatic prefix
caching**: in an agent loop every step resends the same system prompt, the same
instruction files, and the same repository context, and prefix caching means
that is processed once rather than every turn.

Scale on `vllm:num_requests_waiting`, never on processor utilization. A
saturated GPU pod shows low processor use, so a processor-based autoscaler will
never scale and will never explain why.

Keep weights on a shared volume. A twenty-three gigabyte container layer makes
pulls unbearable and pins the workload to one node.

---

## The Context service

This is the part that decides whether the rest is worth running, and most of
it is not a model at all.

**Stage one, parse.** Chunk code at function, method, and class boundaries
using a syntax parser — never at fixed character counts. A chunk that splits a
method in half retrieves confidently and returns something incoherent. Attach
file path, symbol name, enclosing class, imports, and last-modified commit to
every chunk.

**Stage two, index three ways.** Dense vectors for meaning. Lexical search for
exact identifiers. A symbol graph for definitions and references. The lexical
index is not optional: semantic search cannot reliably find
`getUserByTenantId`, and skipping this is the usual cause of code retrieval
returning plausible rubbish.

**Stage three, retrieve and expand.** Query all three, fuse the rankings, then
pull each hit's definition and direct callers from the symbol graph. That
expansion is the difference between returning a snippet and returning a
context.

**Stage four, rerank.** A cross-encoder reads the query and each candidate
together and cuts one hundred candidates to eight. After chunking, this is the
largest quality lever in the pipeline.

**Stage five, assemble.** The small model writes the brief. Code passes through
verbatim — it is never paraphrased, because a paraphrased signature is a
fabrication the engineering model will then build on. The brief also states
what was found and excluded, which is what turns a silent omission into a
named gap.

### Logs

Do not embed raw log lines. Extract templates first, collapsing millions of
lines into a few hundred patterns with counts:

```
ERROR Connection to <*> timed out after <*>ms    — 48,213 occurrences
```

This is roughly ten-thousand-to-one compression with almost no loss for triage
purposes, and it makes anomaly detection trivial: a template that never
appeared before, or a count that jumped an order of magnitude. It also removes
identifiers by construction, so template extraction doubles as a redaction
step.

---

## What is never a model

These stay deterministic permanently. Not until models improve — permanently.

| Job | Tool |
|---|---|
| Chunk code at symbol boundaries | Syntax parser |
| Find all references and callers | Language server |
| Exact identifier search | Lexical index |
| Collapse logs to templates | Template extraction |
| What changed | Version control |

These are not capability gaps. They are category errors. "Find all references
to this method" has one correct answer that the language server *knows*. A
model can only approximate it, and it will approximate it convincingly. Asking
replaces a fact with an opinion.

**The rule that survives every model generation: if the answer is computable,
compute it.** Reserve the model for judgment — what matters here, what should
change, what the risk is. Every token spent making it re-derive something a
tool already knows is a token not spent on the actual problem.

---

## Boundaries

### Environments

The setup lives in development and nowhere else.

| Environment | Reach |
|---|---|
| Development | Full — repository, build, tests, database, logs |
| Every other environment | The deployed site over HTTP, as any visitor would see it |
| Anything else | A human hands over a context file |

Two rules make this structural rather than stated.

**The inference pods hold no credentials at all.** The model emits tool calls;
the tool servers hold the secrets and make the connections. A prompt injection
landing in the engineering model therefore has nothing to steal and nowhere to
reach.

**The environment is never a parameter the model controls.** A tool shaped
`run_query(environment, sql)` puts one argument between the organization and
production, and anything influencing the model can flip it. Separate servers
with separate credentials leave nothing to flip.

Where a deployed site needs a login, use a dedicated minimally privileged
account, never an administrator and never a human's own. Expose it as
`login_as(role)` so the secret stays in the tool server and never enters the
context window. Name the accounts so they are distinguishable from human
traffic in access logs.

On production, observe but do not interact. A site reachable by anyone is also
a site with buttons that do things, and an agent working through a checkout
flow to reproduce a fault creates a real order.

### Approval

The agent's deliverable is a branch and a pull request, never a deployment.
Merge is the approval point, and everything downstream is the existing
promotion pipeline. This means no new approval mechanism has to be invented,
and "execute access in development" means *it can run builds and tests*, not
*it can change things*.

---

## Wiring into VS Code

Three constraints shape this, and all three were confirmed from Microsoft's own
documentation.

**Tool calling is mandatory.** A model without working tool calls does not
appear in the model picker at all, with no error explaining why. This makes the
serving configuration a hard requirement rather than a tuning detail.

**Self-hosted models reach VS Code through the Custom Endpoint provider**,
which speaks the standard chat completions shape. Local models work without a
subscription; on a Copilot Business or Enterprise plan an administrator must
first enable the bring-your-own-key policy, and if it is disabled nothing else
matters.

**Retrieval arrives as explicit tools, not as built-in semantic search.**
Semantic search, embeddings, and inline completions are served by GitHub's own
endpoints and cannot be redirected to a self-hosted model. Exposing retrieval
as tools turns out to be an advantage: tool calls appear in the transcript
where they can be audited, and built-in search does not.

Two settings deserve attention. The declared input and output token limits must
sum to the real context window, and the edit tool should be pinned rather than
auto-selected — local models often handle a straightforward find-and-replace
cleanly while failing at patch formats.

### Work tracking

Work items and repositories reach the agent through the Azure DevOps MCP
server, which Microsoft publishes under a permissive license. It ships in two
forms, and the choice is a data decision rather than a convenience one: the
remote server routes context through a Microsoft-hosted endpoint, while the
local server runs beside the developer and talks to Azure DevOps directly.
Microsoft recommends the remote one and is concentrating new development there,
so choosing the local server means accepting a widening feature gap.

Load only the tool domains actually needed, and pin the package version. The
project has already renamed its tools once in a breaking change.

### Configuration traps

Each of these was found in documentation and each fails quietly.

| Trap | What happens |
|---|---|
| Tool-call parser does not match the model template | Model absent from the picker, no error |
| Coding variants use a different parser from base models | Same silent failure, easy to miss |
| Context length left at its default | Serves approximately 41,000 tokens while VS Code believes it has 256,000 |
| No reasoning parser | Raw thinking tags rendered in the transcript |
| Sampling values copied from general advice | Vendor recommendations differ per mode; wrong values cause repetition |
| Tool output returned uncapped | One failing test run evicts everything useful from the window |
| Too many tools loaded at once | Tool-selection accuracy falls sharply on a small model |

Two of those deserve stating separately.

Every tool definition occupies context on every request, and a small model
degrades faster than a large one as the list grows. Load the minimum and add
deliberately.

Tool results must be capped — roughly two thousand tokens, truncated from the
middle, with the omission stated in the output. Silent truncation is worse than
none, because the model then reasons confidently over a fragment it believes is
whole.

---

## Setup plan

Seven phases, numbered from zero. Each ends in something observable. Do not
start a phase before the previous one has produced its evidence.

The order is deliberate: **everything is proved on rented capacity before
anyone is asked to buy hardware.** The organization's code already sits in a
rented cloud, so nothing in phases zero to five crosses a boundary that has not
already been crossed, and each phase produces evidence the ownership decision
will need.

### Phase 0 — Before any capacity is requested

**Nothing here costs money, and it is the highest-value work available.**

1. **Confirm the bring-your-own-key policy is enabled.** Blocking. A
   two-minute answer that gates everything downstream.
2. **State the actual driver.** Cost per resolved task, a data class the
   current subscription does not cover, or ownership of the capability. These
   are measured differently, and the evaluation set has to measure the real one.
3. **Confirm what data the development database holds.** If it is refreshed
   from a production backup, then "the agent only works in development" quietly
   means the agent has production data with execute access, and the boundary is
   decorative.
4. **Write thirty evaluation tasks** from closed work items. Each needs the
   repository state before, the prompt as it was originally asked, and a
   machine-checkable pass condition. Include several where the correct answer
   is "the information needed is not here".
   **Split them twenty and ten.** The twenty are for development. The ten are
   held back and run only at phase boundaries. Tune against thirty tasks for
   long enough and they stop measuring anything.
5. **Write the instruction files** for TypeScript and SQL. These carry standing
   constraints — verification commands, dependency rules, what never to do —
   and they are per-language specialization at zero cost.

**Exit condition:** the policy is confirmed, the driver is written down, the
database question is answered, and thirty tasks exist with runnable pass
conditions.

### Phase 1 — One model on a rented GPU

A single GPU virtual machine in the existing cloud tenancy, for a day. Serve
the engineering model. Confirm tool calls parse, the context window is what was
configured, and reasoning is separated from content — the three failures that
are silent and that nothing later recovers from.

Then wire it into VS Code and run three real tasks against a real repository.

**Exit condition:** an agent session completes a real change end to end.
If it does not, stop. Nothing later fixes this.

### Phase 2 — Choose the engineering model

Run the twenty development tasks against each candidate on identical machines:
Qwen3.6-35B-A3B, North Mini Code, and Laguna XS 2.1. Record solve rate,
malformed tool calls per hundred, tokens per solved task, and wall clock.

Expect the malformed-call rate to separate the candidates more than the solve
rate does, and note that no public benchmark reports it. Confirm the winner
against the ten held-back tasks before committing.

**Exit condition:** a model chosen on measurements from the organization's own
repositories, not from a leaderboard.

### Phase 3 — The full stack, still rented

Two GPUs. Engineering on one, verifier and retrieval models on the other. Add a
gateway for a single authenticated endpoint with fallback between models. Add
tracing at the level of the agent loop — tokens, tool calls, retries, time to
completion — because hardware metrics will report a healthy cluster while the
agent quietly fails.

From here on, **record the running cost per resolved task**. That number is the
only input the ownership decision genuinely needs, and it cannot be
reconstructed later.

**Exit condition:** the evaluation set reproduces its phase 2 results on the
full stack, and cost per resolved task is being recorded.

### Phase 4 — Context and feedback

Build in this order, measuring after each addition:

1. Parsing and lexical search with no model at all — this alone answers a
   surprising share of real queries and gives a baseline
2. The feedback tools: type check, lint, test, schema introspection, and page
   rendering
3. Log template extraction
4. Dense retrieval, kept only if it measurably improves recall
5. The reranker

For web work the page-rendering tool is the one most often skipped and should
not be. Types being correct does not mean the page is not blank. Letting the
agent load its own work and look at it closes the loop on the front end the way
the type checker closes it on types.

**Exit condition:** each addition has a measured before and after. Anything
that did not improve the numbers is removed.

### Phase 5 — Widen

Only now: the verifier wired in as a reviewing agent, work-item context, and —
if the evidence supports it — adapters trained on the organization's own code.
Adapters are last because instruction files reach much of the same result for
nothing, and because training needs trajectories that do not exist yet.

**Exit condition:** the evaluation set is still improving. When it stops,
stop.

### Phase 6 — Decide on ownership

Only at this point is there enough evidence to ask for physical hardware or
dedicated servers, because only now do the following exist:

- a measured solve rate on the organization's own work
- a running cost per resolved task, over months rather than days
- a known utilization pattern, which is what decides whether owned hardware
  sits idle or saturated
- a working stack that can be moved rather than a design that must be built

**The honest outcome of this phase may be to stay on rented capacity.** Owned
hardware wins on sustained high utilization and loses on everything else. If
the machines would idle most of the day, ownership costs more and delivers the
same thing.

If the driver recorded in phase 0 was ownership rather than cost, that is a
legitimate reason to buy anyway — but it should be argued as that reason, with
the cost difference stated rather than hidden.

**Exit condition:** a decision supported by the numbers from phases 3 to 5,
whichever way it goes.

---

## What could make this the wrong plan

Stated plainly, because a plan that names no failure conditions is not a plan.

**The models may not be good enough for the work.** Open models at this size
score well on published benchmarks. Published benchmarks are not the
organization's codebase. Phase 2 exists to find this out before anything is
bought, and the honest outcome of phase 2 may be to stop.

**The VS Code integration may prove brittle.** It depends on a tool-calling
contract between three independently versioned pieces. Any of them can change.

**The cost case may not hold.** Two GPUs, the capacity to run them, and the
engineering time to maintain the pipeline is not obviously cheaper than a
subscription for a team of ordinary size. If the driver is cost, that has to be
computed rather than assumed — which is why phase 3 starts recording it.

**Ownership may not be the cheaper form of control.** Running on rented
capacity in the organization's own tenancy already gives control of the
weights, the data path, and the model choice. Physical ownership adds control
of the metal, which matters for some reasons and not others. Phase 6 exists so
that difference is argued rather than assumed.

**The capability may already be adequate.** If the existing subscription does
the work well, "we own it" is a real reason but it should be stated as the
reason rather than dressed as a technical necessity.

---

## What this document does not establish

| Claim | Status | Not established |
|---|---|---|
| The architecture works | **Designed only** | No part of it has been built or observed running |
| Model sizes and context windows | Read from vendor documentation | Not measured on real hardware; memory use under load is unknown |
| Thirty-billion quality at three-billion speed | Vendor claim for this architecture | Not measured; no independent confirmation cited here |
| Benchmark comparisons | Published by vendors and one third party | Not reproduced independently; none used this organization's code |
| VS Code supports self-hosted models in an agent loop | Stated in Microsoft's documentation | Not confirmed working with these specific models |
| The tool-call parser choice | Documented for the previous model generation | The current generation is undocumented and may differ |
| The verifier model's serving configuration | — | No parser is documented for it at all |
| The verifier's context window | Two figures published by the same vendor | Which one is correct |
| One model suffices for every language in scope | Argued from how the models are trained and evaluated | Not tested on this codebase |
| Two GPUs are enough | Arithmetic on published quantized sizes | Key-value cache use at long context is unmeasured |
| Costs less than a subscription | — | Not computed |
| Owned hardware beats rented capacity | — | Not computed; depends on a utilization figure that does not exist yet |

The most useful rows are the last two. Everything above them is answerable with
a day of work on a rented machine. Those two require months of running cost
from phases 3 to 5, and deciding what the organization is actually buying.
