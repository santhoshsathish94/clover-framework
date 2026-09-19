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

A significant part of the new Clover AI direction was inspired by the extraordinary engineering work of **Fareed Khan** in [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c).

His work demonstrated a striking practical possibility: a 2.78-trillion-parameter Kimi K3 system could be run through a portable C99 implementation with a very small resident working set by streaming model data from storage.

More importantly, it challenged an assumption about how large AI systems must be deployed. That led us to ask a broader Clover question: **how much of what we assume about AI infrastructure is actually a constraint of the model, and how much is a constraint of the way we build the surrounding system?**

That question became one of the reasons for the new Clover AI implementation direction and our investigation into small models, model routing, CPU/GPU inference, memory residency, local NVMe/cache, and storage-streamed inference.

We want to explicitly acknowledge Fareed Khan's work here. **Extraordinary engineering can change the questions we think are worth asking.**

Clover does not claim his implementation as its own. The upstream project, its ideas, measurements, code, and applicable licensing and attribution remain his work. Our responsibility is to distinguish that foundation clearly from the experiments and implementation developed within Clover.

**Upstream work:** [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)

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
than memory — was examined against Fareed Khan's work. The engine was built and
its weightless gates passed locally, and the published cache measurements
reproduced exactly from a recorded trace. No token was generated: that needs
roughly 1.7 TB of local storage. See
[`work-in-progress/kimi-k3-local-evidence.json`](work-in-progress/kimi-k3-local-evidence.json).

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
3. **The storage-streamed measurement on real work**, which needs roughly 1.7 TB
   of local disk for the checkpoint and packed trunk.

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
