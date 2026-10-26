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
- [Research](research/README.md) — investigations, proposals and hypotheses, with their limits
- [Clover AI v4 release notes](releases/v4.md) — the current seven-layer system
- [Observe experiment](research/observe/README.md) — the next experiment: reciprocal observation, mutual change, and distributed learning
- [Run Clover](clover-kimi-k3/README.md) — the 7-layer Clover system (CPU workstation engine &amp; 8 &times; RTX 3090 GPU neuro-symbolic, compiler-verified, memory-accumulating stack): set it up, tune it to your machine, run it
- [Repository map](docs/repository-organization.md) — where each kind of material belongs

## Clover System &amp; Architecture

**Clover** is both the operating methodology (`Context → Direction → Execution → Outcome → Growth` in [`AGENTS.md`](AGENTS.md)) and the unified open-source AI system built under it ([`clover-kimi-k3/`](clover-kimi-k3/)) — designed to produce meaningful engineering outcomes while remaining bounded, observable, verifiable, and accountable.

It began on a single CPU host from the Clover equation of Kimi K3 and from Fareed Khan's `kimi-k3-in-c` (with its first snapshot [`clover-k3`](https://github.com/santhoshsathish94/clover-framework/tree/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/) in git history), and has evolved into a **7-layer neuro-symbolic, compiler-verified, memory-accumulating system** running on both single-workstation CPU and an `8 × NVIDIA RTX 3090` GPU server (`192 GiB` VRAM):

1. **Governance &amp; Discernment Kernel ([`AGENTS.md`](AGENTS.md)):** Operates under `Context ⇄ Direction ⇄ Execution ⇄ Outcome ⇄ Growth`, keeping epistemic states (`observed`, `externally_verified`, `hypothesis`, `user_stated_belief`, `unresolved`) distinct.
2. **File-Based Memory Tree &amp; Relational Knowledge Graph ([`k3_memory_graph.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_memory_graph.py), [context](clover-kimi-k3/docs/clover-memory-graph-context.md)):** Human-readable Markdown memory files with YAML front matter (`32` canonical seed entries, `65` typed cross-branch relationships), structural concept-bridge cues, and a closed-loop `staging/` promotion gate (`12/12 = 100%` on held-out cross-domain transfer and closed-loop staging tests).
3. **Ontology-Guided Neuro-Symbolic GraphRAG &amp; Symbolic Verifier ([`k3_ontology_graph.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_ontology_graph.py), [context](clover-kimi-k3/docs/clover-ontology-graphrag-research-context.md)):** Deterministic property graph + 4 Horn inference rules + stratified priority-first path traversal (`[E#]` / `[P#]`) + `CloverSymbolicVerifier`.
4. **HybridAgent Continuous `<latent>` Reasoning &amp; Live Tools ([`k3_agent_engine.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_agent_engine.py), [context](clover-kimi-k3/docs/clover-hybridagent-research-context.md)):** 14-step continuous `<latent>` reasoning (`K3NativeHybridLink` `7,168 → 7,168`), live `<search>` (arXiv, Wikipedia, DuckDuckGo, URL fetch), sandboxed `<python>` (math, unit tests, charts), and `MoonViT3d` vision.
5. **On-the-Fly Generative Compilation for Python &amp; Rust ([`k3_generative_compiler.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_generative_compiler.py), [context](clover-kimi-k3/docs/clover-generative-compilation-research-context.md)):** Statement-boundary partial-program sealors ($\mathcal{S}_{\text{PY}}, \mathcal{S}_{\text{RS}}$ via `rustc 1.95.0`) that catch syntax, arity, prefix-shadowing, stagnant loops, doctests, and Rust borrow/type errors mid-generation and roll back the KV cache (`40/40 = 100%` Extended HumanEval, `5/5 = 100%` Live Rust).
6. **93-Layer 2.78T MoE GPU/CPU Engine + Validated Rust Host Path ([`k3_gpu_experts_lib.cu`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu), [`clover-kimi-k3/src/rust/`](clover-kimi-k3/src/rust/)):** Full 93-layer Kimi K3 backbone + 8 validated Rust host-path crates (`slotcache`, `poolinit`, `supply`, `copyq`, `routetable`, `planner`, `loader`, `engcfg` — `212/212` unit tests, `max_abs_diff = 0.0` C++ vs. Rust parity).
7. **11-Pass Parallel Next-Concept Modeling (NCM) &amp; Native Web / VS Code Agent Surfaces ([`k3_chat_server.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_chat_server.py), [`extension.js`](clover-kimi-k3/vscode-extension/extension.js)):** 11-level multiscale residual VQ-VAE and `NextConceptTransformer` emitting 64-token windows in 11 parallel passes (`17,869.8 tok/s` 8-GPU DDP ceiling), serving both the Clover Web UI (`256K` session compaction + Memory Graph drawer) and the native VS Code Agent extension.

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
