# Runtime Context & Engineering Ledgers (`docs/runtime-context/`)

This directory organizes all 85 engineering context documents, system audits, and measurement ledgers for the Clover AI runtime across root release/audit ledgers and six domain subfolders:

## 0. Root Release, Architecture & Audit Contexts

| Document | Scope |
|---|---|
| [v6-opus55-full-audit-2026-11-26.md](v6-opus55-full-audit-2026-11-26.md) | Full-repository Clover `v6.0.0` adversarial audit, multi-round debate, and disposition ledger with `Claude Opus 5.5` |
| [one-brain-five-senses-research-context.md](one-brain-five-senses-research-context.md) | **Next Milestone (Paused Roadmap):** 7-Cycle research & 7-Horizon master plan (`Cycles 1 → 1,000,000+`) uniting 5 deterministic senses, foveated `JEPA`, `Freeze-Omni` converters, Dual-Channel `Tool-NCM`, and `Clover-Seed` |
| [multi-node-architecture-2026-11-12.md](multi-node-architecture-2026-11-12.md) | Multi-node coordinator & stateless GPU worker topology |
| [full-system-test-2026-11-11.md](full-system-test-2026-11-11.md) | End-to-end multi-device system verification ledger |
| [studio-ui-504-walk-2026-11-10.md](studio-ui-504-walk-2026-11-10.md) | Studio UI walkthrough and voice/multimodal integration notes |
| [studio-unified-ui-2026-11-10.md](studio-unified-ui-2026-11-10.md) | Unified Studio single-HTML architecture |
| [security-and-code-review-2026-11-08.md](security-and-code-review-2026-11-08.md) | Historical v5 security and code review ledger |
| [v5-goals-and-cycles-2026-11-08.md](v5-goals-and-cycles-2026-11-08.md) | Historical v5 goals and ablation cycles |

## 1. Engine & Runtime Architecture (`engine-and-architecture/`)

| Document | Scope |
|---|---|
| [clover-minimal-4part-harness-context.md](engine-and-architecture/clover-minimal-4part-harness-context.md) | Clover v6.0 minimal 4-Part Harness (`AGENTS.md` + `mcp.json` + `<task_id>.md` + Todo), `8 × RTX 3090` VRAM budget, LoRA folding, and `Opus 5.5` Stage-3 audit |
| [clover-vscode-studio-integration-context.md](engine-and-architecture/clover-vscode-studio-integration-context.md) | Two-way VS Code extension and Clover Studio workspace integration |
| [clover-kimi-k3-cpu-context.md](engine-and-architecture/clover-kimi-k3-cpu-context.md) | 93-layer CPU runtime derivation, AVX-512/OpenMP kernels, and single-machine status log |
| [clover-kimi-k3-gpu-context.md](engine-and-architecture/clover-kimi-k3-gpu-context.md) | 8x RTX 3090 GPU runtime, Rust host-path crates, and PCIe/NVMe slot management |
| [clover-hybridagent-research-context.md](engine-and-architecture/clover-hybridagent-research-context.md) | Continuous `<latent>` planning (`HybridAgent-8B`) + 11-pass Next-Concept Modeling (`NCM`) |
| [clover-memory-graph-context.md](engine-and-architecture/clover-memory-graph-context.md) | File-backed Markdown Memory Tree (`memory_tree/`), staging gate, and `CSC` graph traversal |
| [clover-ontology-graphrag-research-context.md](engine-and-architecture/clover-ontology-graphrag-research-context.md) | Deterministic Horn-rule Ontology GraphRAG and McCulloch-Pitts symbolic claim verifier |
| [clover-generative-compilation-research-context.md](engine-and-architecture/clover-generative-compilation-research-context.md) | On-the-fly Python & Rust partial-program sealing and statement-boundary compiler feedback |
| [clover-vscode-agent-loop-context.md](engine-and-architecture/clover-vscode-agent-loop-context.md) | Autonomous multi-turn VS Code agent loop (`decide_vscode_agent_step`) and tool dispatch |
| [clover-foundations-to-clover-gpu-context.md](engine-and-architecture/clover-foundations-to-clover-gpu-context.md) | 70-task standardized AI benchmark ablation (`33/70 -> 57/70 -> 70/70`) and SWE-Bench Pro V2 |
| [clover-data-movement-direction-b-context.md](engine-and-architecture/clover-data-movement-direction-b-context.md) | Direction B data-movement synthesis (`SpecMD Least-Stale + R4 Clean Drop`, `148.5 tok/s`) |
| [clover-lfm3-1.7b-stage-study-context.md](engine-and-architecture/clover-lfm3-1.7b-stage-study-context.md) | Liquid AI `LFM3-1.7B` 24-layer tri-hybrid stage-by-stage verification and Clover K3 comparison |
| [clover-nemotron-3.5-super-vl-stage-study-context.md](engine-and-architecture/clover-nemotron-3.5-super-vl-stage-study-context.md) | NVIDIA `Nemotron 3.5 Super VL (120B-A12B)` 88-layer Mamba-2 + Latent `ReLU²` MoE + NoPE GQA + C-RADIO + MTP stage study |
| [clover-ai-system-switchable-models-context.md](engine-and-architecture/clover-ai-system-switchable-models-context.md) | `clover-ai-system/` switchable Layer-1 model architecture (`kimi-k3`, `nemotron-3.5-super-vl`, `lfm3-1.7b`) and Nemotron precomputed trunk/operator/expert engine |
| [clover-hg4-opencode-fusion-egm-lean-context.md](engine-and-architecture/clover-hg4-opencode-fusion-egm-lean-context.md) | Pure-`C1` `HG4-16R-OP` `/v1` Fusion Gateway + `OpenCode` + Google `Elephant–Goldfish (EGM)` + Certified Lean 4 `RiemannHypothesis` verification roadmap & ledger |
| [clover-studio-capabilities-ablation-context.md](engine-and-architecture/clover-studio-capabilities-ablation-context.md) | Sequential `AGENTS.md` phased ablation (`Phases S0 → S7`) of Clover AI Studio subsystems (`Semantics/CSI`, `Outcome Cycle`, `x-cruncher`, `Ontology`, `Epistemic Memory`, `JEPA/RPRE`) |
| [clover-website-hg-format-guide-context.md](engine-and-architecture/clover-website-hg-format-guide-context.md) | Dedicated `cloverframework.com/hg-format/` engineering guide covering `WHT-16` `HG4-16R-OP` / `K31H` conversion, ~98–99% quality retention, setup, benchmarks, and reproducible verification |
| [one-brain-five-senses-research-context.md](one-brain-five-senses-research-context.md) | **Next Milestone (Paused Roadmap):** 7-Cycle research & 7-Horizon master plan (`Cycles 1 → 1,000,000+`) uniting 5 deterministic senses, foveated `JEPA`, `Freeze-Omni` converters, Dual-Channel `Tool-NCM`, and `Clover-Seed` |
| [layer0-verdict.md](engine-and-architecture/layer0-verdict.md) | Layer 0 input-to-embedding and dense KDA stage measurements |

## 2. Foundational Papers 1943–2017 (`foundational-papers-1943-to-2017/`)

| Document | Paper & Year |
|---|---|
| [foundations-11-papers-human-guide.md](foundational-papers-1943-to-2017/foundations-11-papers-human-guide.md) | Plain-language guide to all 11 foundational AI papers and how each upgrades Clover |
| [mcculloch-pitts-1943-context.md](foundational-papers-1943-to-2017/mcculloch-pitts-1943-context.md) | McCulloch & Pitts (1943) — All-or-none threshold logic & inhibitory vetoes |
| [turing-1950-context.md](foundational-papers-1943-to-2017/turing-1950-context.md) | Turing (1950) — Computability, imitation game, and diagonal self-reference guards |
| [newell-simon-1956-context.md](foundational-papers-1943-to-2017/newell-simon-1956-context.md) | Newell & Simon (1956) — Logic Theorist heuristic proof search & subgoal chaining |
| [rosenblatt-1958-context.md](foundational-papers-1943-to-2017/rosenblatt-1958-context.md) | Rosenblatt (1958) — Perceptron $\gamma$-system minority-branch competition |
| [minsky-papert-1969-context.md](foundational-papers-1943-to-2017/minsky-papert-1969-context.md) | Minsky & Papert (1969) — Perceptrons order-of-predicates & One-in-a-Box theorem |
| [hopfield-1982-context.md](foundational-papers-1943-to-2017/hopfield-1982-context.md) | Hopfield (1982) — Content-addressable energy relaxation over memory constellations |
| [rumelhart-hinton-williams-1986-context.md](foundational-papers-1943-to-2017/rumelhart-hinton-williams-1986-context.md) | Rumelhart, Hinton & Williams (1986) — Backpropagation & hidden-unit symmetry breaking |
| [sutton-1988-context.md](foundational-papers-1943-to-2017/sutton-1988-context.md) | Sutton (1988) — $\text{TD}(\lambda)$ temporal-difference statement boundary prediction |
| [lecun-1998-context.md](foundational-papers-1943-to-2017/lecun-1998-context.md) | LeCun et al. (1998) — Graph Transformer Networks & $\text{logadd}$ multi-path scoring |
| [krizhevsky-2012-context.md](foundational-papers-1943-to-2017/krizhevsky-2012-context.md) | Krizhevsky, Sutskever & Hinton (2012) — Cross-channel Local Response Normalization (LRN) |
| [vaswani-2017-context.md](foundational-papers-1943-to-2017/vaswani-2017-context.md) | Vaswani et al. (2017) — Scaled dot-product & $h=3$ orthogonal subspace multi-head attention |

## 3. Hardware Data-Movement Studies (`hardware-data-movement/`)

| Document | Study |
|---|---|
| [red-blue-pebbling-context.md](hardware-data-movement/red-blue-pebbling-context.md) | Hong & Kung (1981) — Red-Blue Pebble Game I/O lower bounds & R4 clean-eviction rule |
| [eyeriss-context.md](hardware-data-movement/eyeriss-context.md) | Chen et al. (2016) — Eyeriss Row-Stationary dataflow & 4-tier energy hierarchy |
| [eyeriss-v2-context.md](hardware-data-movement/eyeriss-v2-context.md) | Chen et al. (2019) — Eyeriss v2 Hierarchical Mesh NoC & CSC sparse weight compression |
| [flashattention-context.md](hardware-data-movement/flashattention-context.md) | Dao et al. (2022) — IO-aware online softmax tiling & exact KDA/MLA HBM accounting |
| [specmd-context.md](hardware-data-movement/specmd-context.md) | SpecMD — Cross-layer lookahead prefetching & least-stale expert slot eviction |

## 4. SuperAgent, OpenJev & Dory Decision Engine Studies (`superagent-jev-dory/`)

- Index & 6 study documents: [superagent-jev-dory/README.md](superagent-jev-dory/README.md) ([01-deepseek-superagent-harness-context.md](superagent-jev-dory/01-deepseek-superagent-harness-context.md), [02-openjev-decision-engine-context.md](superagent-jev-dory/02-openjev-decision-engine-context.md), [03-nvidia-dory-recurrent-loop-context.md](superagent-jev-dory/03-nvidia-dory-recurrent-loop-context.md), [04-clover-three-capability-integration-context.md](superagent-jev-dory/04-clover-three-capability-integration-context.md), [05-review-and-held-out-retest.md](superagent-jev-dory/05-review-and-held-out-retest.md), [06-follow-up-items-1-4.md](superagent-jev-dory/06-follow-up-items-1-4.md)).

## 5. Curiosity & Developmental Learning (`curiosity-and-developmental-learning/`)

- Plain-language guide & 14 study documents (`01`–`14`, Oudeyer, Gottlieb, Schmidhuber, Santucci): [curiosity-13-papers-human-guide.md](curiosity-and-developmental-learning/curiosity-13-papers-human-guide.md) and [14-clover-curiosity-foundation-architecture-context.md](curiosity-and-developmental-learning/14-clover-curiosity-foundation-architecture-context.md).

## 6. Consciousness, Interoception & Indicator Audits (`consciousness-and-subjective-experience/`)

- Plain-language guide & 18 study documents (`01`–`18`, Nagel, Chalmers, Dehaene, Seth, Butlin, Damasio, Varela): [consciousness-17-papers-human-guide.md](consciousness-and-subjective-experience/consciousness-17-papers-human-guide.md) and [18-clover-integration-context.md](consciousness-and-subjective-experience/18-clover-integration-context.md).
