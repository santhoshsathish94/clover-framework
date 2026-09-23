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

**Clover AI** is the next major implementation direction of the project.

The goal is to build an open-source AI system that can produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

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

That second result is what moves us forward instead of keeping us where we are. It showed the wall was the machine rather than the idea, and that a model this size is really two problems: a small part used for every single word, which wants the fastest device available, and a very large expert part that will never fit anywhere and has to stream from disk.

**Clover is not against any system.** Each one is a sensible answer to the situation its builders are in. What they all have in common is people improving their own system.

So we learn from others who solved this shape of problem in their own way. Games have been fitting worlds bigger than their consoles for decades — loading scenery as you approach it, budgeting time per frame — rather than waiting for bigger hardware. And [`llama.cpp`](https://github.com/ggml-org/llama.cpp) already does the mixed-device version: what is used every time goes on the fast device, what is used rarely goes on the slow one.

Next is a **GPU-Server GEX45-1**. It is too small to host this model and is not meant to — it is there to answer the one question the CPU could not.

The experiments, in the order we ran them, and the predictions we got wrong: [`work-in-progress/heterogeneous-inference.md`](work-in-progress/heterogeneous-inference.md).

**Upstream work:** [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) · [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

The detailed implementation direction is intentionally kept separate from this README:

- [`work-in-progress/clover-ai.md`](work-in-progress/clover-ai.md) — Clover AI implementation direction
- [`work-in-progress/README.md`](work-in-progress/README.md) — everything designed or trialed but not yet established

### What has been built, and what has not

The first experiments live in [`work-in-progress/ai-manipulation/`](work-in-progress/ai-manipulation/).
They are deliberately small, and every result below comes from a bounded toy
domain — a search for coefficients of known functions over a five-task ladder.

Demonstrated there, and nowhere else:

- developmental state survives process termination and is reloaded
- a capability is promoted only after passing anchor tests the engine cannot write
- memory lowered the cost of later tasks: 2,397 evaluations without it against 1,869 with it
- a promotion that breaks an earlier capability reverts the whole state
- a worker cannot certify its own learning; only externally evidenced claims are kept
- developmental continuity survived replacing the worker, tested with two real
  local models from different families rather than stand-ins

Not demonstrated, and not claimed:

- that any of it holds outside the designed task space
- that the process can invent a representation rather than search within one
- that a language model continues the trajectory *usefully* — the handover
  carried between two real local models, but neither produced anything that
  passed the evaluator, and the hosted worker has never completed a cycle
- open-ended development, general capability, or subjective experience

A separate question — whether a very large model can run from storage rather
than memory — has now been measured. A 2.78-trillion-parameter model ran on one
rented CPU machine at about 5.3 seconds per word. Shrinking it to 8-bit brought
that to 4.1 and nearly halved the memory it needed; shrinking again to 4-bit
gave almost nothing more, because by then the processor rather than the memory
was the limit. All of it on one machine, one prompt at a time, and none of it on
a GPU. See
[`work-in-progress/kimi-k3-local-evidence.json`](work-in-progress/kimi-k3-local-evidence.json)
and [`work-in-progress/heterogeneous-inference.md`](work-in-progress/heterogeneous-inference.md).

### What v4.0.0 is waiting for

This release, `v3.1.0`, records the machinery and the corrections to it. It does
not claim the result. Three things stand between here and `v4.0.0`, and none of
them is a matter of writing more code:

1. **A task space the experimenter did not design.** Everything above is a
   search for coefficients of known functions. Until the process meets a problem
   nobody shaped for it, the domain is doing the work rather than the method.
2. **A model that completes a cycle and produces something the evaluator
   accepts.** The handover between workers holds; the content does not yet clear
   the bar. The hosted worker has never run a cycle at all — it reached the API
   and stopped at `429 insufficient_quota`.
3. **The storage-streamed measurement on real work.** The model has since been
   downloaded and run — about 5.3 seconds per word on one rented CPU machine,
   across four prompts — and two smaller versions of it were built and measured.
   That is a step, not the result. It showed that the approach runs and where it
   stops paying on a processor, which is a minor change rather than a finished
   direction. Where it points next is untested.

**`v4.0.0` will be cut when one of those produces evidence, and not before.**

Supporting areas include:

- [`docs/05-context-engineering.md`](docs/05-context-engineering.md) — context
- [`docs/07-outcome.md`](docs/07-outcome.md) — outcomes and evidence
- [`docs/08-governance.md`](docs/08-governance.md) — accountability and governance
- [`docs/ai-responsibility/`](docs/ai-responsibility/) — AI responsibility

Clover AI is work in progress. Model choice, infrastructure, serving, tools, security boundaries, and evaluation are experimental until implemented and measured.

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
