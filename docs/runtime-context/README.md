# Runtime Context & Engineering Ledgers (`docs/runtime-context/`)

This directory organizes all 27 engineering context documents and measurement ledgers for the Clover AI runtime into three clear domain subfolders:

## 1. Engine & 7-Layer Runtime Architecture (`engine-and-architecture/`)

| Document | Scope |
|---|---|
| [clover-kimi-k3-cpu-context.md](engine-and-architecture/clover-kimi-k3-cpu-context.md) | 93-layer CPU runtime derivation, AVX-512/OpenMP kernels, and single-machine status log |
| [clover-kimi-k3-gpu-context.md](engine-and-architecture/clover-kimi-k3-gpu-context.md) | 8x RTX 3090 GPU runtime, Rust host-path crates, and PCIe/NVMe slot management |
| [clover-hybridagent-research-context.md](engine-and-architecture/clover-hybridagent-research-context.md) | Continuous `<latent>` planning (`HybridAgent-8B`) + 11-pass Next-Concept Modeling (`NCM`) |
| [clover-memory-graph-context.md](engine-and-architecture/clover-memory-graph-context.md) | File-backed Markdown Memory Tree (`memory_tree/`), staging gate, and `CSC` graph traversal |
| [clover-ontology-graphrag-research-context.md](engine-and-architecture/clover-ontology-graphrag-research-context.md) | Deterministic Horn-rule Ontology GraphRAG and McCulloch-Pitts symbolic claim verifier |
| [clover-generative-compilation-research-context.md](engine-and-architecture/clover-generative-compilation-research-context.md) | On-the-fly Python & Rust partial-program sealing and statement-boundary compiler feedback |
| [clover-vscode-agent-loop-context.md](engine-and-architecture/clover-vscode-agent-loop-context.md) | Autonomous multi-turn VS Code agent loop (`decide_vscode_agent_step`) and tool dispatch |
| [clover-foundations-to-clover-gpu-context.md](engine-and-architecture/clover-foundations-to-clover-gpu-context.md) | 70-task standardized AI benchmark ablation (`33/70 -> 57/70 -> 70/70`) and SWE-Bench Pro V2 |
| [clover-data-movement-direction-b-context.md](engine-and-architecture/clover-data-movement-direction-b-context.md) | Direction B data-movement synthesis (`SpecMD Least-Stale + R4 Clean Drop`, `148.5 tok/s`) |
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
