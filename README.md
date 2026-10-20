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
- [Run clover-kimi-k3](clover-kimi-k3/README.md) — the Kimi K3 engine: set it up, tune it to your machine, run it
- [Repository map](docs/repository-organization.md) — where each kind of material belongs

## Clover AI

**Clover AI** is the implementation arm of the project: an open-source AI system that can produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

The first thing built under it is [`clover-kimi-k3/`](clover-kimi-k3/) — a C engine that runs all 93 layers of Kimi K3 on one CPU host. It began from the Clover equation of the model and from Fareed Khan's `kimi-k3-in-c`. Its first version, `clover-k3`, is in git history (snapshot linked below). That first version was the Kimi K3
forward pass reduced to its mathematics and executed directly. It emits the
same token as an independent implementation of the same model on all 34 prompts
tested, in under a third of the time. What it cost and what it does not show are
in [results and scope](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/evidence/results.md). The current engine, its set-up and its measurements are in [`clover-kimi-k3/README.md`](clover-kimi-k3/README.md).

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

- [`clover-kimi-k3/`](clover-kimi-k3/) — the engine, the set-up script, the dataset file lists and the dated log of everything measured and every negative result ([status log](clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md))
- [`clover-k3/`, first version](https://github.com/santhoshsathish94/clover-framework/tree/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/) — historical snapshot at commit 013cbc8: the program, its build and the correctness gate
- [Results](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/evidence/results.md) — first-token comparison and separate generation outputs
- [Implementation equation](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/reference/implementation-equation.md) — the equation as the program actually evaluates it, and where that differs from the written one
- [Evidence methods](https://github.com/santhoshsathish94/clover-framework/blob/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/docs/evidence/evidence-methods.md) — campaigns, configurations, harnesses and reproduction limits

**How it was worked out**

- [K3 investigation](research/k3/README.md) — grouped derivations, measurements, experiments and handoffs
- [Model equation](research/k3/model/k3-model-equation.md) — the model as one composed expression, executed and checked float for float

**The direction and the boundaries**

- [Clover AI direction](research/clover-ai/README.md) — broader proposed implementation
- [AI-Fire research](research/ai-fire/README.md) — dependency and concentration argument; source-ledger work remains open
- [`docs/07-outcome.md`](docs/framework/07-outcome.md) — what counts as evidence, and what does not
- [`docs/08-governance.md`](docs/governance/08-governance.md) — accountability and governance
- [`docs/ai-responsibility/`](docs/governance/evidence/) — AI responsibility

clover-kimi-k3 has a measured single-machine result; the broader Clover AI direction remains a proposal. A 2.78-trillion-parameter model runs on a Ryzen 9 7950X3D with 124 GB of RAM. The recorded 1,000-token prompt takes about 300 s (about 0.3 s per input token) and its output is bit-identical to the reference chain; decoding costs 3.1 to 3.5 s per output token, and 2.7 s per token with a draft model on one real 1,078-token reply. The first version, clover-k3, was recorded at about 5.9 s per generated token on the same machine (a historical figure, not re-run). These are one-host measurements, not a distributed-system or general quality result.

It will keep improving, and the parts of the principle above that it does not yet reach — the tools that enforce, the serving layer, the evaluation that closes the loop — are the next things to build. They will be held to the same standard: built, gated and measured before they are claimed.

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
