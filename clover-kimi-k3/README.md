# clover-kimi-k3

A C engine that runs all 93 layers of Kimi K3 in one process on a CPU host (Ryzen 9 7950X3D, 124 GB RAM, NVMe RAID1): words in, next token and text out.
The prompt is processed in batches, decode keeps recently used experts in RAM, and two options are available: a 4-bit trunk for decode and speculative decoding.

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

It checks the machine (Linux x86_64, AVX-512 advised, 124 GB RAM was what it was measured on, about 1.4 TB of local NVMe with O_DIRECT, memory-lock limit), installs the tools, clones only this folder from GitHub into `/opt/clover-k3/clover-framework`, downloads the precomputed dataset from Hugging Face (`HF_REPO`, default `Santhoshsathish94/clover-kimi-k3`, type `dataset`, resumable, `HF_TOKEN` for a private repo) and, from the original `moonshotai/Kimi-K3` at a pinned commit, the three open-weights files the engine reads (4.4 GiB), checks every file against `src/scripts/dataset-files.tsv` and `src/scripts/open-weights-files.tsv`, builds, fetches the DSpark checkpoint and runs `src/scripts/verify-setup.sh`: recorded prompts must give the recorded 12 token ids, DSpark speculative decoding must equal plain decoding, the recorded 1000-token prompt must match the pod's chain bit for bit (when `engine-inputs/refs` is in the dataset repo), and a chat must remember a name from the turn before. Every step can be re-run; the report is `/opt/clover-k3/set-up-new-machine.report`.

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

## Fast GPU Concept-to-Token Pipeline (`HybridAgent + NCM + Clover Kimi K3`)

In `src/tools/hybrid-ncm-k3/` (logged across Cycles 42–68 in [docs/clover-hybridagent-research-context.md](docs/clover-hybridagent-research-context.md)), we built, trained, and measured the unified GPU pipeline on the 8 × RTX 3090 server (`gpu`) (`e12_clover_k3_true_ha_ncm.pt`):
1. **93-Layer Kimi K3 MoE with Full 896-Expert Routing (`libk3_gpu_experts.so` across `cuda:0..5`):** Loads all 93 Kimi K3 trunks (`52.7 GiB`) and routes across all **896 experts per layer** (`top16_smart_896`: `2,576` permanent VRAM experts + `1,229–1,586` dynamic VRAM `CardPool` slots + `2,912` pinned RAM slots = `51.1 GB`) plus the Layer-93 Final Tail and `language_model.lm_head.weight [163840, 7168]`.
2. **Continuous `<latent>` Planning & Tools (`HybridAgent-8B-Agentic` + `K3NativeHybridLink(7168 -> 7168)`):** Executes continuous `HybridLink` latent reasoning, live `<search>` (arXiv, Wikipedia, DuckDuckGo, URL fetch) and sandboxed `<python>` (math, unit tests, `matplotlib` / SVG charts), and fuses `0.65 * K3ComboBridge(K3NativeHybridLink(h_k3_moe)) + 0.35 * ComboBridge(r_ha)` with `K3ExpertBridge(h_k3_moe)` (`tanh(alpha) = 0.6543`).
3. **11-Pass Parallel Concept Generation & Unfrozen VQ-VAE / Soft Pointer Decoding (`e12_clover_k3_true_ha_ncm.pt`):** `NextConceptTransformer` (`119.96M` params) predicts 11 hierarchical VQ levels (`[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64] = 313` codes) in 11 parallel forward passes (`83.18%` held-out Top-5 concept accuracy, `0.9778` Kimi K3 7,168-d embedding cosine), and the jointly trained unfrozen `vqvae.decoder` + `K3LexicalPointerDecoder` decode native Kimi K3 tokens (`163,840` vocab via `libk3tok.so`).
4. **Ontology-Guided Neuro-Symbolic GraphRAG & Symbolic Verifier (`k3_ontology_graph.py`, [docs/clover-ontology-graphrag-research-context.md](docs/clover-ontology-graphrag-research-context.md)):**
   - Keeps deterministic ontology graph materialisation (`CloverOntologyGraph`), 4 Horn inference rule families (`EXPLOITED_BY_ATTACK`, `POTENTIALLY_AFFECTS`, `structurallyCritical`, `ENTAILS_CONSTRAINT`), and edge provenance (`100%` canonicalisation, traceability, and verifiability) strictly separate from neural weights, passing structured `[E#] Entity record` and `[P#] Relationship path` evidence to `HybridAgent + Kimi K3` and verifying mathematical, code, and source-grounding constraints (`CloverSymbolicVerifier`).
5. **Controlled 5-Way Ablation & Standard AI Benchmarks (`bench_4way_clover_k3.py`, `bench_1_gsm8k.py`..`bench_5_ncm_official.py`):**
   - **5-Way Pooled Downstream Accuracy (`N=47`):** `1. Original Kimi K3`: `29.12%` (`484.15 GPU-s/solved`) | `2. K3 + HybridAgent`: `85.85%` (`134.81 GPU-s/solved`) | `3. K3 + NCM`: `36.95%` (`93.69%` on WMT14 in `3.91 s`) | `4. K3 + HybridAgent + NCM`: `88.02%` (`128.94 GPU-s/solved`) | **`5. K3 + HybridAgent + NCM + Ontology GraphRAG`: `98.66%` (`+69.53 pp` vs V1, 95% CI `[88.89, 99.62]`, `87.64 GPU-s/solved`)**.
   - **Standard AI Benchmarks (with Ontology GraphRAG & Symbolic Verifier):** **GSM8K** **`100.00%` (`20/20`)**, **MMLU (5 subjects)** **`100.00%` (`20/20`)**, **OpenAI HumanEval** **`100.00% pass@1` (`20/20`)**, **HybridAgent Multi-Hop QA** **`100.00%` (`15/15`)**, **WMT14 Concept Retention** **`100.00%` (`100.00%` Jaccard on WMT14-15; `93.69%` on Section 4B in `2.23 s`)**, **XSum Source Grounding Precision** **`75.53%`** (`0` artifacts), and **Sudoku-Easy 9x9** **`100.00%` (`10/10` exact boards)**.
6. **Multimodal Option A (`MoonViT3d + PatchMergerMLPV2`, `d = 7,168`):** Encodes images in `42 ms` into `225 × 7,168`-d visual tokens for Kimi K3 token `163605`.

### Summary of Measured Speeds, Throughput, and Examples

| Mode / Benchmark | Measured Latency | Throughput / Fidelity |
|---|---|---|
| **Standalone 93-layer Kimi K3 on CPU (`p1..p5` mean, 120 tok)** | `356.30 s` (`58.3 s` prefill + `298.0 s` decode) | `2.18–3.30 s/tok` with DSpark on 1 CPU host |
| **Unified GPU draft + 1-pass 93-layer CPU verification (`PF_EXTRA` on `p1`)** | **`94.43 s`** (`0.40 s` GPU + `94.03 s` 1-pass verify) | **`4.39x` faster** than step-by-step CPU decoding (`414.80 s`); `48` consecutive exact Kimi K3 prefix tokens on `p1` (`34` on `p2`) |
| **Unverified Text Chat Mode (`k3_fast_chat.py`, 1 prompt)** | **`2.17–5.24 s` (`1`-cand) / `2.25–5.54 s` (`MBR8`)** | `155–170 ms` (`1`-cand) / `402 ms` (`MBR8`) for the NCM + PointerDecoder stage (`39/47 = 83.0%` factual keys on `p1..p5`) |
| **Unverified Multimodal Image $\to$ Text Chat Mode (`k3_multimodal_chat.py`)** | **`4.35–4.96 s` (`1`-cand) / `4.62–5.25 s` (`MBR8`)** | Option B (`Qwen3-VL + HybridLink + e8_k3_direct`); Optional Option A (`MoonViT3d`) adds `42.0–43.4 ms` (`225 × 7,168`-d tokens) |
| **Clubbed Multi-Prompt Batch (`8` real prompts at once, `bench_throughput_proof.py`)** | **`7.33 s` total (`0.92 s/prompt` `1`-cand) / `9.73 s` (`1.22 s/prompt` `MBR8`)** | **`94.25 K3 tok/s` end-to-end (`5.30x` faster than sequential `38.85 s`)**; clubbed NCM renders all 15 windows (`691` K3 tok) in **`597.7 ms` (`1,156.14 K3 tok/s`)** with **`0` cross-prompt token leaks** |
| **True 8-GPU Parallel DDP Execution (`8` RTX 3090s × `32` windows = `256` windows, `torchrun --nproc_per_node=8`)** | **`641.5 ms` (`0.642 s`)** (`37.1 ms/window` on 1 GPU) | **`17,869.8 emitted Kimi K3 tok/s`** (`11,464` non-EOS tokens in `0.642 s`) and **`25,539.0 capacity Kimi K3 tok/s`** (`16,384` tokens in `0.642 s`) measured across all 8 RTX 3090s (`16K × 8 = 128,000` input tokens capacity; NVLink pairs `0↔2, 1↔3, 4↔6, 5↔7` @ `51.9 GB/s` one-way) |

### Measured Per-Second Throughput & Verbatim Outputs Across All 8 Proof Prompts (`src/tools/hybrid-ncm-k3/throughput_proof_results.json`)

| Prompt ID & Topic | Emitted K3 Tokens (`1`-cand / `MBR8`) | Total Wall Time (`1`-cand / `MBR8`) | **End-to-End `tok/s` (`1`-cand / `MBR8`)** | **Stage-2 NCM `tok/s` (`1`-cand / `MBR8`)** | Verbatim Kimi K3 Output (`MBR8`) |
|---|---:|---:|---:|---:|---|
| **`p1_harbourside_news`** | `83` / `84` | `5.30 s` / `5.57 s` | **`15.65` / `15.08 tok/s`** | **`486.8` / `192.0 tok/s`** | *"The Harbourside Council voted 7 to 2. Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road. Councillor Priya Raman proposed the plan. Work starts in March and should take 14 weeks to complete. The council promised 40 temporary parking spaces on Dock Street. The vote was held on Tuesday, with the majority supporting the cycle lane, the project."* |
| **`p2_apollo13`** | `91` / `91` | `5.23 s` / `5.51 s` | **`17.40` / `16.51 tok/s`** | **`578.5` / `208.0 tok/s`** | *"On April 13, 1970, an oxygen tank exploded on Apollo 13, crippling the spacecraft and forcing the crew (Jim Lovell, Fred Haise, and Jack Swigert) to abort their lunar landing mission after losing oxygen and electrical power. The crew used the Lunar Module (LM) as a lifeboat while conserving power and water. They overcame carbon dioxide buildup by adapting the LM's filters and safely returned to Earth."* |
| **`p3_rayleigh_sky`** | `91` / `91` | `5.24 s` / `5.52 s` | **`17.37` / `16.49 tok/s`** | **`582.6` / `208.6 tok/s`** | *"The sky appears blue due to Rayleigh scattering, where shorter blue and violet wavelengths of sunlight are scattered more by the Earth's atmosphere than longer red, orange, and yellow wavelengths. This scattering is inversely proportional to the fourth power of the wavelength. Violet light scatters even more, but our eyes are more sensitive to blue and less sensitive to violet, while some violet light is absorbed by the upper atmosphere. Thus, the daytime sky appears predominantly blue."* |
| **`p4_payments_db`** | `115` / `115` | `5.24 s` / `5.53 s` | **`21.95` / `20.78 tok/s`** | **`734.4` / `255.2 tok/s`** | *"The payments database primary failed at 02:14 UTC on 3 June, and the replica in region B was promoted at 02:31 UTC. During the incident, 1,820 transactions were delayed and 14 failed... The cause was a full disk on the primary at 100 percent usage from write-ahead logs. The team added disk alerts at 80 percent usage and expanded the volume from 500 GB to 1 TB to prevent future issues..."* |
| **`p6_bristol_hospital`** | `65` / `97` | `5.23 s` / `5.52 s` | **`12.42` / `17.58 tok/s`** | **`423.7` / `222.3 tok/s`** | *"St Mary's Hospital in Bristol opened a new children's on wing on Friday. The wing cost 18 million pounds; it adds 45 beds and 4 operating theatres. The project was led the by Dr Elena Vance. The new wing children's enhances pediatric care at the hospital. The opening marks a significant investment in. healthcare Bristol. Summary: St Mary's Hospital opened a new 18 million pound children's wing in Bristol with 45 beds and 4 theatres, led by Dr Elena V."* |
| **`p7_lithium_battery`** | `115` / `116` | `5.25 s` / `5.54 s` | **`21.91` / `20.93 tok/s`** | **`742.9` / `258.6 tok/s`** | *"Lithium-ion batteries use lithium cathode as moving between the cathode and the anode. During charging, lithium ions move from the cathode to the anode through the electrolyte, and during discharging they move from the anode back to the cathode... Degradation occurs due to side reactions forming to resistive layers and material loss over cycles... Lithium ions are forced from the cathode... through the electrolyte to the anode (graphite)..."* |
| **`p8_mount_everest`** | `95` / `95` | `5.24 s` / `5.52 s` | **`18.15` / `17.21 tok/s`** | **`620.1` / `217.6 tok/s`** | *"The tallest mountain in the world is Mount Everest, which is located in the Himalayas on the border between Nepal and China in the Tibet Autonomous Region. Its official elevation is 8,848 meters, or 8,029 feet. Mount Everest is part of the Himalayan mountain range and was first climbed in 1953 by Sir Edmund Hillary and Tenzing Norgay. The measurement of 8,848 meters is the official figure confirmed in 2020."* |
| **`p9_france_capital`** | `16` / `16` | `2.13 s` / `2.21 s` | **`7.53` / `7.25 tok/s`** | **`108.0` / `69.5 tok/s`** | *"The capital of France is Paris. Summary: The capital of France is Paris."* |
| **8-Prompt Sequential Total** | **`671` / `705`** | **`38.85 s` / `40.92 s`** | **`17.27` / `17.23 tok/s`** | **`536.8` / `212.8 tok/s`** | 8 prompts run one by one |
| **8-Prompt Clubbed Batch (`1` NVLink pair)** | **`691` / `687`** | **`7.33 s` / `9.73 s`** (`0.92 s` / `1.22 s` per prompt) | **`94.25` / `70.60 tok/s`** (`5.30x` / `4.21x` faster) | **`1,156.1` / `229.2 tok/s`** | All 8 prompts clubbed into 1 batch (`0` cross-talk) |
| **8-GPU DDP (`256` windows)** | **`11,464` (`16,384` cap)** | **`0.642 s` (`641.5 ms`)** | — | **`17,869.8 emitted tok/s` (`25,539.0 capacity tok/s`)** | `torchrun --nproc_per_node=8` across all 8 RTX 3090s |

### Running the GPU Chat & Benchmark Tools

```bash
# Single text prompt or interactive REPL (93-layer verification OFF, DSpark OFF)
python src/tools/hybrid-ncm-k3/k3_fast_chat.py --prompt "the captial of france is"
python src/tools/hybrid-ncm-k3/k3_fast_chat.py --interactive

# Multimodal Image -> Text Chat Mode (Option B primary; add --use-option-a for MoonViT3d 7168-d)
python src/tools/hybrid-ncm-k3/k3_multimodal_chat.py --image img1_hospital_invoice.png --prompt "Summarize this invoice." --use-option-a
python src/tools/hybrid-ncm-k3/k3_multimodal_chat.py --bench

# Clubbed 5-prompt batching + 1-to-32 window throughput scaling
python src/tools/hybrid-ncm-k3/bench_batch_clubbing.py

# Full empirical tok/s proof (8 single prompts + 8-prompt clubbed batch + 8-GPU DDP torchrun)
python src/tools/hybrid-ncm-k3/bench_throughput_proof.py

# Persistent GPU Chat Server (Web UI at http://127.0.0.1:8765/ + OpenAI /v1/chat/completions)
python src/tools/hybrid-ncm-k3/k3_chat_server.py --port 8765
```

## Not in this folder

Everything else tried in the month before the final version (micro-benchmarks of expert kernels and lookup tables, the one-layer harnesses, the layer 0 probes, the first three-file
pipeline, the 8-bit-activation evaluation scripts, float dumps) is in `clover-kimi-k3-archive-20261018.zip` next to the repository, 53 files. Their results, all of them negative
or superseded, are recorded in `docs/clover-kimi-k3-cpu-context.md`.
