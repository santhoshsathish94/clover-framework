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
- [Clover AI v4 release notes](releases/v4.md) — the current seven-layer system
- [Observe experiment](research/observe/README.md) — the next experiment: reciprocal observation, mutual change, and distributed learning
- [Run Clover](clover-kimi-k3/README.md) — the 7-layer Clover system (CPU workstation engine &amp; 8 &times; RTX 3090 GPU neuro-symbolic, compiler-verified, memory-accumulating stack): set it up, tune it to your machine, run it
- [Repository map](docs/repository-organization.md) — where each kind of material belongs

## Clover System &amp; Architecture (`Clover AI v4`)

**Clover** is both the operating methodology (`Context → Direction → Execution → Outcome → Growth` in [`AGENTS.md`](AGENTS.md)) and the unified open-source AI system built under it ([`clover-kimi-k3/`](clover-kimi-k3/)) — designed to produce meaningful outcomes while remaining bounded, observable, verifiable, and accountable.

It began on a single CPU host from the Clover equation of Kimi K3 and from Fareed Khan's `kimi-k3-in-c` (with its first snapshot [`clover-k3`](https://github.com/santhoshsathish94/clover-framework/tree/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/) in git history), and currently stands as a **7-layer neuro-symbolic, compiler-verified, memory-accumulating system** running on both a single CPU workstation (`Ryzen 9 7950X3D`, `124 GB` RAM) and an `8 × NVIDIA RTX 3090` GPU server (`192 GiB` VRAM), reproducibly bootstrapped from GitHub and Hugging Face ([`Santhoshsathish94/clover-kimi-k3`](https://huggingface.co/datasets/Santhoshsathish94/clover-kimi-k3)):

1. **Governance &amp; Discernment Kernel ([`AGENTS.md`](AGENTS.md)):** Operates under `Context ⇄ Direction ⇄ Execution ⇄ Outcome ⇄ Growth`, keeping epistemic states (`observed`, `externally_verified`, `hypothesis`, `user_stated_belief`, `unresolved`) distinct.
2. **File-Based Memory Tree &amp; Relational Knowledge Graph ([`k3_memory_graph.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_memory_graph.py), [context](clover-kimi-k3/docs/clover-memory-graph-context.md)):** Human-readable Markdown memory files with YAML front matter (`32` canonical seed entries, `65` typed cross-branch relationships), a closed-loop `staging/` promotion gate, and six mechanisms from the 11 foundational AI papers ([`foundations-11-papers-human-guide.md`](clover-kimi-k3/docs/foundations-11-papers-human-guide.md) &amp; [`clover-foundations-to-clover-gpu-context.md`](clover-kimi-k3/docs/clover-foundations-to-clover-gpu-context.md)).
3. **Ontology-Guided Neuro-Symbolic GraphRAG &amp; Symbolic Verifier ([`k3_ontology_graph.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_ontology_graph.py), [context](clover-kimi-k3/docs/clover-ontology-graphrag-research-context.md)):** Deterministic property graph + 4 Horn inference rules + Newell–Simon `CSm`/`VCt` signature pruning + `[E#]` / `[P#]` provenance + `CloverSymbolicVerifier`.
4. **HybridAgent Continuous `<latent>` Reasoning &amp; Live Tools ([`k3_agent_engine.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_agent_engine.py), [context](clover-kimi-k3/docs/clover-hybridagent-research-context.md)):** 14-step continuous `<latent>` reasoning (`K3NativeHybridLink` `7,168 → 7,168`), live `<search>` (arXiv, Wikipedia, DuckDuckGo, URL fetch), sandboxed `<python>` (math, unit tests, charts), and `MoonViT3d` vision.
5. **On-the-Fly Generative Compilation for Python &amp; Rust ([`k3_generative_compiler.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_generative_compiler.py), [context](clover-kimi-k3/docs/clover-generative-compilation-research-context.md)):** Statement-boundary partial-program sealors ($\mathcal{S}_{\text{PY}}, \mathcal{S}_{\text{RS}}$ via `rustc`) that catch syntax, arity, prefix-shadowing, stagnant loops, doctests, and Rust borrow/type errors mid-generation and roll back the KV cache.
6. **93-Layer 2.78T MoE GPU/CPU Engine, Validated Rust Host Path &amp; Direction B Data Movement ([`k3_gpu_experts_lib.cu`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu), [`clover-kimi-k3/src/rust/`](clover-kimi-k3/src/rust/), [context](clover-kimi-k3/docs/clover-data-movement-direction-b-context.md)):** Full 93-layer Kimi K3 backbone + 8 validated Rust host-path crates (`slotcache`, `poolinit`, `supply`, `copyq`, `routetable`, `planner`, `loader`, `engcfg` — `213/213` unit tests, `max_abs_diff = 0.0` C++ vs. Rust parity) + five Direction B data-movement upgrades (`95.22%–95.29%` overall VRAM hit rate, `0` PCIe D2H writebacks on cold/warm reads, `148.5 K3 tok/s` across 93 GPU layers).
7. **11-Pass Parallel Next-Concept Modeling (NCM) &amp; Native Web / VS Code Agent Surfaces ([`k3_chat_server.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/k3_chat_server.py), [`extension.js`](clover-kimi-k3/vscode-extension/extension.js)):** 11-level multiscale residual VQ-VAE and `NextConceptTransformer` emitting 64-token windows in 11 parallel passes (`17,869.8 tok/s` 8-GPU DDP ceiling), serving both the Clover Web UI (`256K` session context) and the native VS Code Agent extension.

### What the 70-Task Evaluation &amp; Head-to-Head Comparison Show

To evaluate Clover without conflating a 70-task system suite with full-benchmark leaderboards, we ran the **exact same 70 prompts** ([`run_real_ai_benchmark_suite.py`](clover-kimi-k3/src/tools/hybrid-ncm-k3/run_real_ai_benchmark_suite.py)) through three standalone frontier reference models (**GPT-6 Astra**, **Claude Opus 5.5**, and **Gemini 4 Argon**, with zero tools and zero external memory/ontology) and through **three controlled arms of Clover's own underlying model** on our cleanly bootstrapped `8 × RTX 3090` server ([full report](clover-kimi-k3/docs/clover-foundations-to-clover-gpu-context.md)):

| Model / Controlled Arm | Public Tasks (`N=58`: MMLU, GSM8K, Code, Structured, Multi-Hop) | Ontology + Memory Tasks (`N=12`) | All 70 Tasks (`Strict / Normalized`) | Compute / Latency |
| :--- | :---: | :---: | :---: | :--- |
| **Arm 0: Clover Underlying Model Alone** *(No Mem / Onto / Tools / GC)* | `30 / 58` (`51.72%`) | `3 / 12` (`25.00%`) | **`33 / 70` (`47.14%`)** | `965.4 s` (`234.0 GPU-s / solved`) |
| **Arm 1: Clover Underlying + Tools &amp; Compiler** *(No Mem / Onto)* | `51 / 58` (`87.93%`) | `6 / 12` (`50.00%`) | **`57 / 70` (`81.43%`)** | `1,015.4 s` (`142.5 GPU-s / solved`) |
| **Arm 2: Full 7-Layer Clover (`v4`)** *(All Mechanisms Active)* | **`58 / 58` (`100.00%`)** | **`12 / 12` (`100.00%`)** | **`70 / 70` (`100.00%`)** | `1,002.3 s` (**`114.5 GPU-s / solved`**) |
| **GPT-6 Astra** *(Standalone, No Tools/Mem/Onto)* | `54 / 58` (`56 / 58 = 96.55%`*) | `7 / 12` (`58.33%`)‡ | `61 / 70` (**`63 / 70 = 90.00%`***) | `333.6 s` (`4.77 s/task`, Cloud API) |
| **Claude Opus 5.5** *(Standalone, No Tools/Mem/Onto)* | `51 / 58` (`57 / 58 = 98.28%`*) | `8 / 12` (`66.67%`)‡ | `59 / 70` (**`65 / 70 = 92.86%`***) | `263.2 s` (`3.76 s/task`, Cloud API) |
| **Gemini 4 Argon** *(Standalone, No Tools/Mem/Onto)* | `50 / 58` (`86.21%`)† | `7 / 12` (`58.33%`)‡ | **`57 / 70` (`81.43%`)** | `206.4 s` (`2.95 s/task`, Cloud API) |

*\*All three frontier models solved `10/10` GSM8K math problems correctly; unmodified regex matched intermediate numbers when `#### <num>` was omitted. †Gemini 4 Argon scored `0/5` on 9×9 Sudoku without a solver (`55..78/81` cells). ‡Standalone models rightly abstained on private organizational ontology and memory records not present in the prompt.*

**Three takeaways for readers and AI agents:**
1. **On standard public tasks, standalone frontier models are already near-perfect (`96.6%–98.3%`), whereas Clover's raw underlying model is not (`51.7%`).**
2. **Where standalone frontier models miss on this 70-task set (`7/12` to `8/12` on Ontology + Memory) is missing Context, not missing intelligence:** no model can know organization-specific records or prior session measurements that are not in its prompt.
3. **Clover's measured contribution is the `+52.86 percentage point` system lift (`33/70 → 70/70`, `2.04×` lower GPU-seconds per solved task) it gives a local open-weights model** by combining verified tool execution, compiler-in-the-loop code generation, symbolic ontology, and persistent memory.

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

**Upstream work:** [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) · [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) · [togethercomputer/HybridAgent](https://github.com/togethercomputer/HybridAgent) · [HazyResearch/ncm](https://github.com/HazyResearch/ncm) · [CosmoIknosLab/ontology-guided-graphrag](https://github.com/CosmoIknosLab/ontology-guided-graphrag).

---

### Supporting areas &amp; boundaries

- **Single-workstation CPU engine (`clover-kimi-k3`):** A 2.78-trillion-parameter model runs on a single Ryzen 9 7950X3D with 124 GB of RAM ([status log](clover-kimi-k3/docs/clover-kimi-k3-cpu-context.md)). The recorded 1,000-token prompt takes `297.5–301.0 s` (`~0.3 s` per input token) and is bit-identical to the reference chain (`1,000/1,000` embeddings, `1,000/1,000` layer-92 outputs, `8,000/8,000` snapshots); decoding costs `3.1–3.5 s` per output token plain, and `2.7 s` per token with DSpark speculative decoding.
- **8 × RTX 3090 GPU 7-layer stack (`clover-kimi-k3/src/tools/hybrid-ncm-k3/`):** Bootstrapped from GitHub and Hugging Face ([`set-up-new-machine.sh`](clover-kimi-k3/src/scripts/set-up-new-machine.sh), [`download-hybrid-ncm-datasets.sh`](clover-kimi-k3/src/scripts/download-hybrid-ncm-datasets.sh)), verified across the 70-task suite ([`benchmarks/results/1936d4eb0d53/gpu/`](benchmarks/results/1936d4eb0d53/gpu/)).
- **Historical &amp; research material:**
  - [`clover-k3/`, first version](https://github.com/santhoshsathish94/clover-framework/tree/013cbc89a99ba506c567b081613049ac76d405a6/clover-k3/) — historical snapshot at commit `013cbc8` (`~5.9 s` per generated token on CPU)
  - [K3 investigation](research/k3/README.md) &amp; [Model equation](research/k3/model/k3-model-equation.md) — grouped derivations, measurements, and float-for-float verification
  - [Observe experiment](research/observe/README.md) — the next experiment (reciprocal observation, mutual change, and distributed learning; tracked separately from `Clover AI v4`)
  - [`docs/framework/07-outcome.md`](docs/framework/07-outcome.md) &amp; [`docs/governance/08-governance.md`](docs/governance/08-governance.md) — what counts as evidence, accountability, and governance

## Contributing

Clover is an open, evidence-driven project. Corrections, disagreements, evidence, implementation experiments, and practical improvements are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
