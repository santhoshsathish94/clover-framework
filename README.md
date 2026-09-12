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

## Clover AI

**Clover AI** is the next major implementation direction of the project.

### Acknowledgement

A significant part of the thinking that led to the Clover AI direction was sparked by the extraordinary engineering work of **[Fareed Khan](https://github.com/FareedKhan-dev)** and his [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c) project.

That work challenged a fundamental assumption about how large AI systems must be deployed: whether the entire model must always be resident in memory. By demonstrating Kimi K3 inference through a portable C99 implementation with a working set streamed from storage, it opened a practical engineering question that we believe deserves much deeper investigation.

This directly influenced the direction of our Clover AI infrastructure experiments and our interest in small, efficient, self-hosted, and storage-aware AI systems.

We want to acknowledge this clearly because the direction did not emerge in isolation. It was influenced by someone else's remarkable work, and that work deserves to be recognized.

The Clover AI work in this repository is our own implementation direction and experimentation. It should not be confused with or presented as an extension of Fareed Khan's project. His work is the inspiration for this particular line of investigation, and applicable licensing, attribution, and upstream notices should always be respected.

> **Extraordinary engineering can change the questions we think are worth asking.**

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

The detailed implementation direction is intentionally kept separate from this README:

- [`work-inprogess/clover-ai.md`](work-inprogess/clover-ai.md) — Clover AI implementation direction

Supporting areas include:

- [`docs/05-context-engineering.md`](docs/05-context-engineering.md) — context
- [`docs/07-outcome.md`](docs/07-outcome.md) — outcomes and evidence
- [`docs/08-governance.md`](docs/08-governance.md) — accountability and governance
- [`docs/ai-responsibility/`](docs/ai-responsibility/) — AI responsibility

Clover AI is work in progress. Model choice, infrastructure, serving, tools, security boundaries, and evaluation are experimental until implemented and measured.

## Contributing

Clover is a work in progress. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
