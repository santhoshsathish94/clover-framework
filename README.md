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

The framework itself is documented at **[cloverframework.com](https://cloverframework.com/)** and in `docs/`.

This repository contains the documentation, practical guidance, case studies, and the implementation work around Clover.

## Start here

- **[Clover Framework](https://cloverframework.com/)** — understand the framework
- [`QUICKSTART.md`](QUICKSTART.md) — use Clover on real work
- [`AGENTS.md`](AGENTS.md) — AI agent operating guidance
- [`docs/README.md`](docs/README.md) — documentation index
- [`case-studies/`](case-studies/) — real work and outcomes
- [`why-clover-is-important.md`](why-clover-is-important.md) — the research behind why this matters

## Clover AI

**Clover AI** is the implementation arm of the project: an open-source AI system that can produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

The first thing built under it is [`clover-k3/`](clover-k3/) — the Kimi K3
forward pass reduced to its mathematics and executed directly. It emits the
same token as an independent implementation of the same model on all 34 prompts
tested, in a third of the time. What it cost and what it does not show are in
[What has been built](#what-has-been-built-and-what-has-not) below.

Its central principle is:

> **The model can propose.  
> The system decides.  
> The tools enforce.  
> Evidence verifies.  
> Humans remain accountable.**

The implementation follows the Clover cycle:

```text
Context → Direction → Execution → Outcome → Growth
```

### Where this direction came from

We started with our own question: **how much of what we assume about running AI is really about the model, and how much is just about how we build everything around it?**

Rather than argue it, we ran something. Using Fareed Khan's [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), which streams a model off disk instead of holding it in memory, we ran a 2.78-trillion-parameter model on a single rented CPU machine. Then we shrank it twice to see what would happen. The first shrink sped it up exactly as we had predicted. The second barely helped — because by then the processor, not the memory, had become the limit. The engine is his work under Apache-2.0 and stays his; Clover does not claim it.

That second result changed the question again. It showed that the wall was not simply the size of the model; it was where the work was happening. On this CPU, moving the always-used trunk from 8-bit to 4-bit saved almost no time because the processor spent the saving unpacking it. That does not tell us that 4-bit is a dead end. It tells us that the answer may depend on the machine.

The new GPU server changes the experiment. The GEX131-1 has enough total local storage to hold the model across its two NVMe drives, so we no longer need to treat the checkpoint as one stream coming from one mirrored storage layout. The first storage experiment used mirrored disks. The next experiment can **shard the model across two disks** and measure whether independent storage paths let us overlap reads, prefetching, and computation.

More importantly, the GEX131-1 GPU has **96 GB of VRAM**. That is enough for the measured INT8 trunk (54.47 GB) and MXFP4 trunk (28.94 GB), but not the BF16 trunk (108.81 GB). That gives us a useful range of placement experiments rather than a simple fit/no-fit question:

> **What actually needs to be on the GPU for each token?**

The next experiment should therefore not assume that the trunk and experts must be processed the way they were on the CPU. We can investigate a working-set architecture:

```text
                 1.56 TB model
                       ↓
              sharded across 2 disks
                       ↓
             CPU RAM / NVMe working set
                  ↙          ↘
          dense trunk       experts
               ↓               ↓
          96 GB GPU working set
                       ↓
                 token generation
```

The GPU may hold the most frequently used trunk components and/or frequently selected experts, while RAM and NVMe hold the larger backing store. The system can prefetch what the next token is likely to need while the GPU is computing the current token.

This is not yet an architecture claim. It is the next experiment.

**Clover is not against any system.** Every approach we have encountered is a sensible answer to the situation its builders were in. Games solved a similar problem by loading only the part of a much larger world that was needed at the moment. `llama.cpp` already demonstrates mixed-device model placement. We are now asking whether the same principle, applied at the level of this model's trunk, experts, storage, and token processing, produces a measurable advantage on hardware we can actually rent.

The question is no longer simply whether a 2.78-trillion-parameter model can run on a small machine.

It is:

> **Can we discover a useful division of the model across GPU, CPU memory, and sharded storage that makes each token faster without requiring the whole model to fit on one device?**

The GEX131-1 is there to answer that question.

The experiments, in the order we ran them, and the predictions we got wrong: [`k3-analysis/heterogeneous-inference.md`](k3-analysis/heterogeneous-inference.md).

**Upstream work:** [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) · [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

The detailed implementation direction is intentionally kept separate from this README:

- [`k3-analysis/clover-ai.md`](k3-analysis/clover-ai.md) — Clover AI implementation direction
- [`k3-analysis/README.md`](k3-analysis/README.md) — everything designed or trialed but not yet established

### What has been built, and what has not

**The model written as mathematics, and then run.**
[`clover-k3/`](clover-k3/) is the Kimi K3 forward pass as a single C file of
2,517 lines, with no dependency beyond libc, libm and OpenMP. It was written
from [the equation](k3-analysis/k3-model-equation.md) rather than from any
existing engine — ten primitive operators, two attention blocks, two MLP blocks,
one composed expression for the whole model.

Stating it that way is what had to come first. The equation document was
transcribed into a program and run against the engine's own tap points:
**79,742,816 float32 values identical, zero mismatches**, and the token the
engine emits. Writing the mathematics down did not merely describe the model;
it is what made the rest checkable, because from then on any change had to
reproduce one md5.

Being the equation rather than an engine is also why it can go faster. Every
operator is stated, so each can be timed and attributed separately, and the
reads each one implies can be scheduled ahead of it. Over the same 34 prompts,
the same machine and the same 1.5 TB checkpoint, against an independent C
implementation of the same model:

| | answers | total wall |
|---|---|---|
| clover-k3 | **34 / 34 identical** | **455.94 s** |
| the other implementation | | 1344.23 s |

The same question in English, French and German emits the same token id.
The full comparison, decoded so the answers can be checked by eye, is in
[`clover-k3/clover-k3-proof.md`](clover-k3/clover-k3-proof.md).

One finding did most of the work. **Routing at position *t* is a pure function
of tokens 0..*t*** — measured 644/644 with a control that fails as it should —
so for a prompt already seen, the experts layer L+1 will need are known while
layer L is still being computed. Reading them a layer early took a 5-token
prompt from 8.75 s to **7.25 s** and the NVMe array from 80% to 98-99% busy.
The device ceiling was then measured directly, with the program's own access
pattern: **14.6 GB/s**, and the run sits at 94% of it. What remains is bytes,
not scheduling.

**What this does not show.** Both programs were asked for one token, not a
sentence. It is not a comparison against the official Kimi runtime, or a GPU,
or vLLM — it is one CPU C implementation against another. Of the 14.79 s gap on
the first prompt, 5.73 s is simply the other implementation re-reading the
54.47 GB trunk from disk each run while clover-k3 holds it in page cache, which
is a configuration choice and not an algorithmic win. Thirty-four prompts,
English-dominant, the longest 24 tokens. Single runs, not repeated trials. The
limits are listed in full in the proof document rather than left to be
discovered.

Things that were built, measured, and **did not** pay are recorded beside the
ones that did, in
[`k3-analysis/clover-scaling-architecture.md`](k3-analysis/clover-scaling-architecture.md):
io_uring is 7-11% slower than the plain reader threads at every queue depth
tried, and preloading the next expert into CPU cache is a wash at best and
costs 0.65 s at worst, because 97.6% of loads already hit L2.

---

A separate question — whether a very large model can run from storage rather
than memory — has now been measured. A 2.78-trillion-parameter model ran on one
rented CPU machine at about 5.3 seconds per token. Shrinking it to 8-bit brought
that to 4.1 and nearly halved the memory it needed; shrinking again to 4-bit
gave almost nothing more, because by then the processor rather than the memory
was the limit. All of it on one machine, one prompt at a time, and none of it on
a GPU. See
[`k3-analysis/kimi-k3-local-evidence.json`](k3-analysis/kimi-k3-local-evidence.json)
and [`k3-analysis/heterogeneous-inference.md`](k3-analysis/heterogeneous-inference.md).

Running it that closely surfaced changes worth offering back, so they were
contributed rather than kept here:
[FareedKhan-dev/kimi-k3-in-c#67](https://github.com/FareedKhan-dev/kimi-k3-in-c/pull/67)
issues expert reads concurrently in 1 MiB pieces, reads a weight matrix once per
prefill pass instead of once per token, and drops repeated reads in the KDA
recurrence. Measured against that project's current `main`, three runs per arm
with every run reported: decode 5.7% faster, prefill 9.9%, a whole run 7.0%.
Output is unchanged and the batched kernel is bit-identical to the one it
replaces.

**What validates this is the measurement, not its reception.** The runs
happened, on the released checkpoint, and the numbers repeat within a fraction
of a percent. The pull request is open and may never be merged; a maintainer
weighs their own roadmap, their own hardware and the cost of maintaining
someone else's code, and can decline for reasons that are entirely sound and
that leave the measurement exactly where it is. Acceptance is a separate
question from whether reality validated the change, and only the second one is
evidence.

What would genuinely weaken it is stated in the pull request itself. The three
changes were measured together, so one of them may contribute nothing. One of
the test machine's two NVMe drives is negotiating a degraded PCIe link, so the
storage change may help less on healthy hardware. And an earlier version of this
work claimed a larger prefill gain against an older base; re-measuring against
current `main` showed roughly a fifth of it had already been earned upstream.
Those are limits in the evidence. Merge status is not.

Supporting areas include:

- [`docs/05-context-engineering.md`](docs/05-context-engineering.md) — context
- [`docs/07-outcome.md`](docs/07-outcome.md) — outcomes and evidence
- [`docs/08-governance.md`](docs/08-governance.md) — accountability and governance
- [`docs/ai-responsibility/`](docs/ai-responsibility/) — AI responsibility

Clover AI is work in progress. The inference layer now has one implementation that is built, gated and measured; model choice, serving, tools, security boundaries, and evaluation remain experimental until the same is true of them.

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
