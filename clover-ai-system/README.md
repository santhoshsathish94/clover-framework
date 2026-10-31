# Clover AI System (`clover-ai-system/`)

**Clover AI System** is organized around **3 Core Pipeline Components (`Inference → Plan → Decode`)** supported by **3 Shared Supporting Components (`Memory · Relationships · Tools`)** — see [CLOVER_FRAMEWORK_ARCHITECTURE.md](CLOVER_FRAMEWORK_ARCHITECTURE.md) for the complete architectural specification:

- **3 Core Pipeline Components (`Inference → Plan → Decode`)**:
  1. **`C1` · Inference (`Base Model` + Built-In Direction B Hardware Engine)**: Switchable across `Kimi K3` (`93` layers, `2.78T` MoE), `Nemotron 3.5 Super VL` (`52` layers, `36B-A3.6B`), and `LFM3-Omni-2B` / `LFM3-1.7B` (`24` Tri-Hybrid layers), executing precomputed `trunk-L/` + `expert-L/` (or `ffn.4bit.bin`) with Direction B hardware acceleration (`Adreno 740 OpenCL dot_acc_sat`, `Kryo ARMv9 i8mm SMMLA`, `FlashAttention-2`, `LFM3-SoC-Q4`, `SpecMD Policy 3 madvise`, and `LCP` cache reuse) built directly into the engine.
  2. **`C2` · Plan (`HybridAgent` & `HybridLink` Post-Inference Planner)**: Operates on post-inference hidden states $h_t \to r_t = R(h_t)$ using direct resident weights, stepping continuous `<latent>` vectors into `C1`'s cache and running the user-triggered Clover Retry Cycle (`clover_cycle.py`).
  3. **`C3` · Decode (`11-Level NCM` Concept Renderer)**: Maps `(h_t, r_t)` through `ComboBridge` + `ExpertBridge` into `313` hierarchical VQ concept codes across 11 levels and decodes via `LexicalPointerDecoder` using direct resident weights + active-vocabulary `mmap` slicing.
- **3 Shared Supporting Components (`Memory · Relationships · Tools`)**:
  4. **`S1` · Memory (`Local Git Repository & Multi-Repo Workspaces` — `clover_git_memory_repo.py`)**: Root `AGENTS.md` onboarding gate, per-folder `<folder>-context.md`, root `<repo>-context-index.md`, `sessions/`, `tasks/`, `scripts/`, `notes/`, staged `lessons/` (Keep/Discard), selective low-token retrieval (`<= 420` tokens), and private personal GitHub sync.
  5. **`S2` · Relationships (`Knowledge & Ontology Graph` + 11 Foundational Papers — `k3_ontology_graph.py`)**: Multi-hop causal/entity GraphRAG, Horn-rule inference, source grounding audit, and all 11 Foundational Papers (`1943–2017`).
  6. **`S3` · Tools (`External Capabilities, Execution & Validation`)**: Syntax & Generative Compiler Validator (`k3_generative_compiler.py` + Sutton `TD(0)`), MCP connections & VS Code bridge, terminal/Python runner, self-hosted web search & URL reader (`live-web-search-tool.py`), and PNG/MP4/WAV media generators.

## The final version

`src/scripts/run-model.sh` with no options. Measured on the host, recorded 1,000-token prompt (745 words):

| | result |
|---|---|
| Whole request, words to next token | 297.5 s and 301.0 s in two measurements (layers 2-92: 296.5 s, layer 0: 1.1 s, layer 1: 3.3 s, head: 0.06 s) |
| Against the pod's own chain | embeddings 1000 of 1000 rows, layer 92 output 1000 of 1000 rows, 8 snapshots 8000 of 8000 rows bit-identical |
| Decode, exact, 16 cached experts per layer | 3.1 - 3.5 s per token |
| Decode, `USE_4BIT_TRUNK=1` | 2.5 - 2.8 s per token (not bit-identical; 8 of 34 recorded prompts give the same 12 tokens as exact, the rest are fluent but differ) |
| Speculative decoding, prompt lookup (`SPECULATIVE_DRAFTER=` empty) | same tokens as plain decoding; 2.5x faster on a repeating prompt, 21% slower on ordinary code (10% of draft tokens kept) |
| Speculative decoding, DSpark draft model (`SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8`; the width, 0 to 8 draft tokens, is chosen per pass from how many were right recently), 48 new tokens | same ids as plain decoding on all 3 prompts, both trunks. Exact trunk, s per token plain -> DSpark: code 3.463 -> 2.028, repeating numbers 3.132 -> 0.902, prose 3.212 -> 2.972. With `USE_4BIT_TRUNK=1` (against 4-bit plain): code 2.823 -> 1.924, repeating 2.510 -> 1.214, prose 2.609 -> 2.314. A fixed width of 8 was faster on code (1.793) and 1.5x slower than plain on prose (4.766). One run each |
| Prompt pass with 8-bit activations (experimental, not bit-identical) | experts only 222.0 s; experts and trunk 203.0 s; the 4-bit trunk is not used here (it makes the prompt pass slower). Quality of the trunk part not yet checked |

## Folder map

```
clover-kimi-k3/
├── README.md
├── docs/                              clover-kimi-k3-cpu-context.md (the dated log of everything measured, every negative result), clover-kimi-k3-gpu-context.md (the same for the GPU work), layer0-verdict.md
└── src/
    ├── code/                          all the code: common/ (shared by every machine), cpu/ (the workstation engine), gpu/ (a GPU server: probes only so far), third-party/ (see below)
    │   └── third-party/kimi-k3-in-c/  the tokenizer headers (Apache-2.0, with licence and notice, github.com/FareedKhan-dev/kimi-k3-in-c at commit a2ad8e5, unmodified)
    ├── scripts/                       run-model.sh (build and run), settings.sh (every path, one place), checks, chat, new-machine setup, the two file lists
    ├── tools/                         compare-outputs.py, compare-34-prompts-with-recorded.py, expert-route-analysis.py, prune-trunk.py, check-open-weights.c
    ├── prompts/                       the 34 recorded prompts and their recorded results, expected-exact-ids.tsv, the recorded 1000-token prompt, small prompts
    ├── datasets/                      host only, never committed
    │   ├── third-party/               made by others
    │   │   ├── kimi-k3-open-weights/  three original files of moonshotai/Kimi-K3 under their own names (4.4 GiB)
    │   │   └── kimi-k3-dspark/        the DSpark draft checkpoint, Inferact/Kimi-K3-DSpark (7.1 GB)
    │   └── computed/                  computed for this engine
    │       ├── expert-1 ... expert-92 the experts of each layer (15.7 GB each)
    │       └── trunk-0 ... trunk-92   the other weights of each layer: only the records the engine reads (33.25 GiB for layers 1-92), and operator.bin, the Q/K/V weights (18.3 GiB in all)
    └── work/                          host only: build/ (compiled program), inputs/ (recorded prompt, the pod's chain values), results/ (every log and output)
```

The code in `src/code/`, by what it does. It is split by kind of machine: `common/` has no vector instructions and no OpenMP kernels and includes nothing from `cpu/`; `cpu/` is the workstation engine (AVX-512, OpenMP, huge pages). A different kind of machine gets its own folder beside `cpu/` and shares `common/`; `gpu/` is the first, and holds probes only, no engine yet (see [its README](src/code/gpu/README.md)).

| file | what it is |
|---|---|
| `common/whole-model.c` | the driver: setup, prompt pass, decode, speculative decoding, chat mode, the final stage loaded as a library |
| `common/backend.h` | the boundary: everything the driver calls from a backend (layer 0, the two layer kinds, DSpark, expert memory), written down once; a backend defines these |
| `cpu/cpu-memory.c` | the CPU backend's expert memory: anonymous memory on huge pages, touched at once |
| `common/words-to-embeddings.c` | stage 1: text to token ids to embedding vectors |
| `cpu/layer0-batch.c` | stage 2: layer 0 for a batch of tokens, built on `layer0.c` |
| `cpu/layers-1-to-92-batch.c` | stage 3: layers 1-92 for a batch of tokens, built twice (KDA layers 1-92 except the MLA ones, from `layer-kda.c`; MLA layers, from `layer-mla.c`): batched trunk, router, experts, expert cache, state checkpoints (`lbk_*`, `lbm_*`) |
| `cpu/layer0.c`, `cpu/layer-kda.c`, `cpu/layer-mla.c` | one token through one layer: layer 0, the source of the KDA layers 1-92 (written for layer 2; the layer number is a run-time value), the source of the MLA layers (written for layer 3); the batch files are built on these |
| `cpu/dspark-draft.c` | the DSpark draft model (5 dense MLA layers, Markov head), drafts 8 tokens from the target's hidden states after layers 2, 23, 47, 71, 89 |
| `common/ids-to-text.c` | small tool: token ids to text |
| `cpu/trunk-4bit.h` | the 4-bit trunk: packing and kernels, shared by `layer0-batch.c` and `layers-1-to-92-batch.c` |
| `common/final-stage.c` (with `final-normalization.c`, `token-ends.h`) | the last stage, built as `pipeline-stage.so`: final normalisation, vocabulary head, detokenise |
| `common/`: `embedding-and-head-tables.h`, `safetensors.h`, `sha256.h`, `inflate-simple.h` | the embedding table and output head read from the open weights, the safetensors reader, hashing, the decoder for the table containers |
| `cpu/expert-reader.h`; `common/`: `expert-metadata.h`, `expert-validation.h`, `inflate-fast.h` | reading and checking the experts (`expert-N`) |
| `common/`: `stage-routing.h`, `stage-trace.h`, `generation-request.h`, `deliver-answer.h`, `json-reader.h` | small shared pieces of the earlier pod design |

Most of these files were written for an earlier design with one pod per layer; they are the same code except for reading the open weights directly, for trunk records that may be pruned, and for the names (`root` is now `expert`). Their old names and places are in `git log --follow`.

On the host the engine lives in `/opt/clover-k3/clover-kimi-k3`. Nothing outside this folder is needed to build or run it. The data under `src/datasets` can be linked in from a bigger filesystem (`DATASET_DIR`, `OPEN_WEIGHTS_DIR`, `DSPARK_DIR`).

## Setting up a new machine

```
curl -fsSL https://raw.githubusercontent.com/santhoshsathish94/clover-framework/main/clover-kimi-k3/src/scripts/set-up-new-machine.sh -o set-up-new-machine.sh
sudo bash set-up-new-machine.sh                  # every step; or name steps: preflight packages clone download open-weights check-dataset inputs build dspark verify
```

It checks the machine (Linux x86_64, AVX-512 advised, 124 GB RAM was what it was measured on, about 1.4 TB of local NVMe with O_DIRECT, memory-lock limit), installs the tools, clones only this folder from GitHub into `/opt/clover-k3/clover-framework`, downloads the precomputed dataset from Hugging Face (`HF_REPO`, default `Santhoshsathish94/clover-ai-studio`, type `dataset`, organized under `kimi/`, `nemotron/`, and `liquid-lfm3/`, resumable, `HF_TOKEN` for a private repo) and, from the original `moonshotai/Kimi-K3` at a pinned commit, the three open-weights files the engine reads (4.4 GiB), checks every file against `src/scripts/dataset-files.tsv` and `src/scripts/open-weights-files.tsv`, builds, fetches the DSpark checkpoint and runs `src/scripts/verify-setup.sh`: recorded prompts must give the recorded 12 token ids, DSpark speculative decoding must equal plain decoding, the recorded 1000-token prompt must match the pod's chain bit for bit (when `engine-inputs/refs` is in the dataset repo), and a chat must remember a name from the turn before. Every step can be re-run; the report is `/opt/clover-k3/set-up-new-machine.report`.

The same steps run as a GitHub Actions workflow, `.github/workflows/set-up-clover-kimi-k3.yml` (start it from the Actions tab), on the new machine registered as a self-hosted runner with the label `clover-k3` (its service needs `LimitMEMLOCK=infinity`).

What the Hugging Face repo must contain (its root is the dataset folder): `expert-1` to `expert-92`, `trunk-0` to `trunk-92` (each with its `operator.bin`; `trunk-1` to `trunk-92` pruned with `tools/prune-trunk.py`), and optionally `engine-inputs/refs/` with `embed.f32`, `final.resid.f32`, `final.snaps.f32` (286 MB, the pod's chain for the recorded prompt). Nothing that is in the open weights is in it: the embedding table, the output head, the normalisation tensors and the tokenizer files come from `moonshotai/Kimi-K3`, the DSpark draft model from `Inferact/Kimi-K3-DSpark`.

Publishing that dataset to the repository (the maintainer's side): `src/scripts/upload-dataset-git.sh` pushes it over ssh with git-lfs, in batches small enough for a nearly full disk (one expert layer, or ten trunks, at a time), and can be run again; `src/scripts/keep-uploading.sh` starts it again until every listed file is on the remote and then writes a `COMPLETE` marker; `src/scripts/upload-status.sh` shows how far it is. `src/scripts/upload-dataset.sh` is the same upload with `hf upload-large-folder` for someone who has a token. How it was done and measured is in `docs/clover-kimi-k3-cpu-context.md`.

## Running

```
src/scripts/run-model.sh recorded                          # the 1000-token prompt, compared with the pod's chain
src/scripts/run-model.sh text src/prompts/code-is-prime.txt    # any text file
PF_DECODE=48 PF_CACHE=16 src/scripts/run-model.sh text src/prompts/code-is-prime.txt                      # 48 more tokens, plain
PF_DECODE=48 PF_CACHE=16 USE_4BIT_TRUNK=1 src/scripts/run-model.sh text src/prompts/code-is-prime.txt     # 4-bit trunk in decode
PF_DECODE=48 PF_CACHE=16 SPECULATIVE_DRAFT_TOKENS=7 src/scripts/run-model.sh text src/prompts/repeating-numbers.txt
PF_DECODE=48 PF_CACHE=16 SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8 src/scripts/run-model.sh text src/prompts/code-is-prime.txt   # DSpark draft model
src/scripts/check-dspark-drafter.sh 48                     # plain against DSpark on three prompts, ids must be identical
```

### Chat with several turns

One process keeps the model loaded and the conversation in memory, so a new turn only runs its own tokens through the prompt pass (a short follow-up costs about 30 s plus its reply).

```
src/scripts/chat-start.sh mychat "Answer briefly."     # starts it detached; session-mychat/ready appears when the model is loaded
src/scripts/chat-say.sh mychat "What is a KDA layer?"  # sends one message and waits (WAIT=seconds, 60 by default) for the reply
src/scripts/chat-stop.sh mychat
```

Settings: `PF_CONTEXT` positions the layers hold (8192, about 19 GB of attention cache), `PF_DECODE` longest reply (2500), `PF_CHUNK` tokens per prompt pass (1024), `PF_REPLY_RESERVE` room kept for a reply (1024), `PF_INPUT_MAX` longest message before it is summarised first (half the context), `PF_SUMMARY_TOKENS` longest summary (320). When a new message would leave less than `PF_REPLY_RESERVE` positions for the reply, the oldest turns are replaced by a summary the model writes (kept in `session-NAME/conversation-summary-NNNN.txt`) and the state is rebuilt, which costs a prompt pass over the kept text. `src/scripts/check-chat-session.sh` runs these paths with a small window; its measured results are in `docs/clover-kimi-k3-cpu-context.md`.

Settings (environment): `USE_4BIT_TRUNK` (0/1), `ACTIVATION_8BIT_EXPERT` and `ACTIVATION_8BIT_TRUNK` (0/1, experimental, off by default, prompt pass), `PF_DECODE` (new tokens), `PF_CACHE` (experts kept per layer), `SPECULATIVE_DRAFT_TOKENS` (0-15), `PF_CHAT` (1, low, high, max) with `PF_SYSTEM`,
`PF_SAVE_IDS` (write generated ids), `PF_VERBOSE`, `BUILD=0` (skip build), `CLOVER_LOCK_TRUNK` (0 = do not lock the trunk files in RAM; needed with the 4-bit trunk).
Long runs on the host are started detached (`setsid nohup ... &`) and their log read afterwards.

## 8 × RTX 3090 GPU 7-Layer Stack (`src/tools/hybrid-ncm-k3/`)

In `src/tools/hybrid-ncm-k3/`, Clover runs interactively on an `8 × NVIDIA RTX 3090` server (`192 GiB` VRAM, NVLink pairs `0↔2, 1↔3, 4↔6, 5↔7` @ `51.9 GB/s` one-way, `376 GiB` RAM) from a clean GitHub (`santhoshsathish94/clover-framework`) and Hugging Face (`Santhoshsathish94/clover-ai-studio`, with top-level folders `kimi/`, `nemotron/`, and `liquid-lfm3/`) bootstrap:

1. **93-Layer Kimi K3 MoE + Validated Rust Host Path + Direction B Data Movement (`libk3_gpu_experts.so` across `cuda:0..5`):**
   - Loads all 93 Kimi K3 trunks (`52.7 GiB`) and routes across all **896 experts per layer** (`2,576` permanent VRAM experts + `1,229` dynamic VRAM `CardPool` slots + `2,912` pinned RAM slots = `51.1 GB`) alongside 8 validated Rust host crates (`slotcache`, `poolinit`, `supply`, `copyq`, `routetable`, `planner`, `loader`, `engcfg` — `213/213` unit tests, `max_abs_diff = 0.0`).
   - Direction B data-movement upgrades ([docs/clover-data-movement-direction-b-context.md](../docs/runtime-context/engine-and-architecture/clover-data-movement-direction-b-context.md)): Red-Blue Pebble `R4` clean-drop admission (`0` PCIe D2H writebacks on cold/warm reads), SpecMD Policy 3 Least-Stale (`LS`) + Cross-Layer Lookahead (`95.22%–95.29%` overall VRAM hit rate), FlashAttention-2 Split-KV MLA (`2.57×–6.84×` faster), and Eyeriss v1/v2 broadcast coalescing (`148.5 K3 tok/s` across 93 GPU layers).
2. **HybridAgent Continuous `<latent>` Reasoning & Live Tools (`k3_agent_engine.py` on `cuda:6`):**
   - 14 continuous `<latent>` reasoning steps (`HybridAgent-8B-Agentic` + `K3NativeHybridLink(7168 -> 7168)`), live `<search>` (arXiv, Wikipedia, DuckDuckGo, URL fetch), sandboxed `<python>` (math, unit tests, `matplotlib` / SVG charts), and `MoonViT3d` vision (`cuda:7`).
3. **On-the-Fly Python & Rust Generative Compilation (`k3_generative_compiler.py`):**
   - Seals partial Python and Rust code prefixes at statement boundaries during generation and rolls back the KV cache when Python AST/doctest or `rustc --edition=2021` diagnostics fail ([docs/clover-generative-compilation-research-context.md](../docs/runtime-context/engine-and-architecture/clover-generative-compilation-research-context.md)).
4. **Ontology-Guided Neuro-Symbolic GraphRAG & Persistent Memory Graph (`k3_ontology_graph.py`, `k3_memory_graph.py`):**
   - Deterministic ontology graph + 4 Horn inference rules + `[E#]`/`[P#]` provenance ([docs/clover-ontology-graphrag-research-context.md](../docs/runtime-context/engine-and-architecture/clover-ontology-graphrag-research-context.md)), paired with a 32-entry, 65-edge file-based Markdown Memory Graph with a `staging/` promotion gate ([docs/clover-memory-graph-context.md](../docs/runtime-context/engine-and-architecture/clover-memory-graph-context.md)) and six mechanisms from the 11 foundational AI papers ([docs/clover-foundations-to-clover-gpu-context.md](../docs/runtime-context/engine-and-architecture/clover-foundations-to-clover-gpu-context.md)).
5. **11-Pass Parallel Next-Concept Modeling (NCM) (`e12_clover_k3_true_ha_ncm.pt`):**
   - `NextConceptTransformer` (`119.96M` params) + `K3-Native VQ-VAE` (`144.00M` params) + `K3LexicalPointerDecoder` emit 64-token windows in 11 parallel forward passes (`17,869.8 emitted K3 tok/s` across 8 GPUs in DDP mode).

### What the 70-Task Evaluation & Head-to-Head Comparison Show

Running `src/tools/hybrid-ncm-k3/run_real_ai_benchmark_suite.py` and `src/models/benchmark_raw_models_without_clover_70_tasks.py` evaluates all three switchable Layer-1 models (`kimi-k3`, `nemotron-3.5-super-vl`, and `lfm3-1.7b`) across 70 standardized tasks in 7 categories both **without Clover (`Arm 0`, raw model alone)** and **with the full 7-Layer Clover Cycle (`Arm 2`)** ([`../research/clover_3_models_without_vs_with_clover_70_task_report.json`](../research/clover_3_models_without_vs_with_clover_70_task_report.json)):

| Category (`N=70` Standardized Tasks) | `kimi-k3` Alone *(Without Clover)* | `nemotron-3.5` Alone *(Without Clover)* | `lfm3-1.7b` Alone *(Without Clover)* | **`kimi-k3` + Clover** *(Full 7-Layer Cycle)* | **`nemotron-3.5` + Clover** *(Full 7-Layer Cycle)* | **`lfm3-1.7b` + Clover** *(Full 7-Layer Cycle)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. MMLU (`N=10`)** | `6 / 10` (`60.0%`) | `2 / 10` (`20.0%`) | `3 / 10` (`30.0%`) | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** |
| **2. GSM8K Math (`N=10`)** | `2 / 10` (`20.0%`) | `0 / 10` (`0.0%`) | `2 / 10` (`20.0%`) | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** |
| **3. Coding (`N=15`: Python-10 + Rust-5)** | `12 / 15` (`80.0%`) | `3 / 15` (`20.0%`) | `3 / 15` (`20.0%`) | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** |
| **4. Structured Output (`N=15`)** | `6 / 15` (`40.0%`) | `7 / 15` (`46.7%`) | `7 / 15` (`46.7%`) | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** |
| **5. Ontology GraphRAG (`N=6`)** | `0 / 6` (`0.0%`) | `0 / 6` (`0.0%`) | `0 / 6` (`0.0%`) | **`6 / 6` (`100.0%`)** | **`6 / 6` (`100.0%`)** | **`6 / 6` (`100.0%`)** |
| **6. Persistent Memory (`N=6`)** | `3 / 6` (`50.0%`) | `2 / 6` (`33.3%`) | `3 / 6` (`50.0%`) | **`6 / 6` (`100.0%`)** | **`6 / 6` (`100.0%`)** | **`6 / 6` (`100.0%`)** |
| **7. Agent Multi-Hop QA (`N=8`)** | `4 / 8` (`50.0%`) | `2 / 8` (`25.0%`) | `1 / 8` (`12.5%`) | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** |
| **OVERALL PASS RATE (`70` Tasks)** | **`33 / 70` (`47.14%`)** | **`16 / 70` (`22.86%`)** | **`19 / 70` (`27.14%`)** | **`70 / 70` (`100.00%`)** | **`70 / 70` (`100.00%`)** | **`70 / 70` (`100.00%`)** |
| **Clover Cycle Lift Over Model Alone** | Baseline | Baseline | Baseline | **`+37 tasks` (`+52.86 pp`)** | **`+54 tasks` (`+77.14 pp`)** | **`+51 tasks` (`+72.86 pp`)** |

To avoid conflating a 70-task mixed suite with full-benchmark leaderboards, we also ran the **exact same 70 prompt strings** through **GPT-6 Astra**, **Claude Opus 5.5**, and **Gemini 4 Argon** (where our harness called each model directly without attaching Clover's tools, compiler, ontology, or memory layers—the same raw-model setup as `Arm 0`) and through **three controlled arms of Clover's own underlying model** on `8 × RTX 3090`, scoring every output with the exact same frozen scoring harness ([raw artifacts in `../benchmarks/results/1936d4eb0d53/gpu/`](../benchmarks/verified-runs/run-baseline-before-foundations/gpu/)):

| Category / Metric (`N=70`) | **Arm 0: Underlying Model Alone** *(Clover Layers Off)* | **Arm 1: Underlying + Tools & GC** *(No Mem / Onto)* | **Arm 2: Full 7-Layer Clover** *(All 6 Surrounding Layers)* | **GPT-6 Astra** *(Clover Layers Not Attached)* | **Claude Opus 5.5** *(Clover Layers Not Attached)* | **Gemini 4 Argon** *(Clover Layers Not Attached)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. MMLU (`N=10`)** | `6 / 10` (`60.0%`) | `7 / 10` (`70.0%`) | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** |
| **2. GSM8K (`N=10`)** | `2 / 10` (`20.0%`) | **`10 / 10` (`100.0%`)** | **`10 / 10` (`100.0%`)** | `8 / 10` (`10/10` math*) | `4 / 10` (`10/10` math*) | **`10 / 10` (`100.0%`)** |
| **3. Coding (`N=15`: Python-10 + Rust-5)** | `12 / 15` (`80.0%`) | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** | **`15 / 15` (`100.0%`)** |
| **4. Structured Output (`N=15`)** | `6 / 15` (`40.0%`) | `11 / 15` (`73.3%`) | **`15 / 15` (`100.0%`)** | `13 / 15` (`86.7%`) | `14 / 15` (`93.3%`) | `7 / 15` (`46.7%`)† |
| **5. Ontology GraphRAG (`N=6`)** | `0 / 6` (`0.0%`) | `2 / 6` (`33.3%`) | **`6 / 6` (`100.0%`)** | `3 / 6` (`50.0%`)‡ | `4 / 6` (`66.7%`)‡ | `3 / 6` (`50.0%`)‡ |
| **6. Persistent Memory (`N=6`)** | `3 / 6` (`50.0%`) | `4 / 6` (`66.7%`) | **`6 / 6` (`100.0%`)** | `4 / 6` (`66.7%`)‡ | `4 / 6` (`66.7%`)‡ | `4 / 6` (`66.7%`)‡ |
| **7. Agent Multi-Hop QA (`N=8`)** | `4 / 8` (`50.0%`) | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** | **`8 / 8` (`100.0%`)** |
| **Public Subset (`58` tasks: Cats 1–4 & 7)** | `30 / 58` (`51.72%`) | `51 / 58` (`87.93%`) | **`58 / 58` (`100.00%`)** | `54 / 58` (`56/58 = 96.55%`*) | `51 / 58` (`57/58 = 98.28%`*) | `50 / 58` (`86.21%`) |
| **ALL 70 TASKS (`Strict / Normalized`)** | **`33 / 70` (`47.14%`)** | **`57 / 70` (`81.43%`)** | **`70 / 70` (`100.00%`)** | `61 / 70` (**`63/70 = 90.00%`***) | `59 / 70` (**`65/70 = 92.86%`***) | **`57 / 70` (`81.43%`)** |
| **Total Time & Cost / Solved Task** | `965.4 s` (`234.0 GPU-s`) | `1,015.4 s` (`142.5 GPU-s`) | `1,002.3 s` (**`114.5 GPU-s`**) | `333.6 s` (Cloud API) | `263.2 s` (Cloud API) | `206.4 s` (Cloud API) |

*\*All three frontier models solved `10/10` GSM8K math problems correctly; unmodified regex matched intermediate numbers when `#### <num>` was omitted. †Gemini 4 Argon scored `0/5` on 9×9 Sudoku without a solver. ‡Standalone models rightly abstained on private ontology and memory records not present in the prompt.*

### Running the GPU Server & Standardized Benchmark Suite

```bash
# Download all 32 HybridAgent/NCM/Vision checkpoints & open-weights from Hugging Face
bash src/scripts/download-hybrid-ncm-datasets.sh

# Start the persistent GPU Chat Server (Web UI at http://127.0.0.1:8765/ + /api/chat + /api/health)
/opt/clover-k3/venv/bin/python src/tools/hybrid-ncm-k3/k3_chat_server.py --host 127.0.0.1 --port 8765

# Run Phase A preflight & Phase B full 70-task benchmark suite
cd src/tools/hybrid-ncm-k3
/opt/clover-k3/venv/bin/python run_real_ai_benchmark_suite.py --mode preflight --target gpu
/opt/clover-k3/venv/bin/python run_real_ai_benchmark_suite.py --mode full --target gpu --run-id run_01
```

## Not in this folder

Everything else tried in the month before the final version (micro-benchmarks of expert kernels and lookup tables, the one-layer harnesses, the layer 0 probes, the first three-file
pipeline, the 8-bit-activation evaluation scripts, float dumps) is in `clover-kimi-k3-archive-20261018.zip` next to the repository, 53 files. Their results, all of them negative
or superseded, are recorded in `docs/clover-kimi-k3-cpu-context.md`.
