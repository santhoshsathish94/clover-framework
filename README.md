<p align="center">
  <a href="https://cloverframework.com/">
    <img src="assets/branding/social-preview.png" width="820"
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
- [Research](research/README.md) — investigations, proposals and hypotheses, with their limits
- [Run Clover-K3](clover-k3/README.md) — executable single-machine reference
- [Repository map](docs/repository-organization.md) — where each kind of material belongs

## Clover AI

**Clover AI** is the implementation arm of the project: an open-source AI system that can produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

The first thing built under it is [`clover-k3/`](clover-k3/) — the Kimi K3
forward pass reduced to its mathematics and executed directly. It emits the
same token as an independent implementation of the same model on all 34 prompts
tested, in under a third of the time. What it cost and what it does not show are
in [results and scope](clover-k3/docs/evidence/results.md).

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

**Upstream work:** [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) · [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The engine that opened this line of work is Fareed Khan's, Apache-2.0, and stays his; Clover does not claim it.

---

Supporting areas include:

**What was built**

- [`clover-k3/`](clover-k3/) — the program, how to build it, and the correctness gate it has to reproduce
- [Results](clover-k3/docs/evidence/results.md) — first-token comparison and separate generation outputs
- [Implementation equation](clover-k3/docs/reference/implementation-equation.md) — the equation as the program actually evaluates it, and where that differs from the written one
- [Evidence methods](clover-k3/docs/evidence/evidence-methods.md) — campaigns, configurations, harnesses and reproduction limits

**How it was worked out**

- [K3 investigation](research/k3/README.md) — grouped derivations, measurements, experiments and handoffs
- [Model equation](research/k3/model/k3-model-equation.md) — the model as one composed expression, executed and checked float for float

**The direction and the boundaries**

- [Clover AI direction](research/clover-ai/README.md) — broader proposed implementation
- [AI-Fire research](research/ai-fire/README.md) — dependency and concentration argument; source-ledger work remains open
- [`docs/07-outcome.md`](docs/framework/07-outcome.md) — what counts as evidence, and what does not
- [`docs/08-governance.md`](docs/governance/08-governance.md) — accountability and governance
- [`docs/ai-responsibility/`](docs/governance/evidence/) — AI responsibility

Clover-K3 has a measured single-machine result; the broader Clover AI direction remains a proposal. A 2.78-trillion-parameter model runs from its equation on a Ryzen 9 7950X3D, bit-exact against a preserved gate, and emits the same first token as an independent implementation on every one of the 34 prompts tested — in **394.94 seconds against 1344.23**, a factor of 3.40 under the recorded configurations. A separate generation run averaged about six seconds per token. These are distinct measurements, not a distributed-system or general quality result.

It will keep improving, and the parts of the principle above that it does not yet reach — the tools that enforce, the serving layer, the evaluation that closes the loop — are the next things to build. They will be held to the same standard: built, gated and measured before they are claimed.

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
