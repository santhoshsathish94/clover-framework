# Context: Multi-Node Clover Architecture (`Laptop/Phone` ↔ `k3 Coordinator + Kimi K3.1 Pro` ↔ `GX10 Stateless Nemotron 3.5 Super VL 4-Bit`)

- **Date:** 2026-11-12
- **Branch / Worktree:** `feature/multi-node-clover-architecture` (`C:\personal\oss\clover-multi-node-wt`, branched from `origin/main` `4d1c8ac5` / `v5.0.6`)
- **Governing Specification:** [AGENTS.md](../../AGENTS.md) (`Context → Direction → Execution → Outcome → Growth`)

---

## 1. Human Direction

Build and verify the multi-node Clover architecture one step at a time, starting with basic setup and verification of each node using the `k3` versions of the full Clover system on each model, then integrating them:

| Node | Runs | Job |
|---|---|---|
| **Laptop, phone** (clients) | **LFM3-Omni-2B, locally** (full Clover system) | Answer simple requests themselves; send larger ones to `k3`; send image/audio/video requests for Nemotron through `k3`; work offline |
| **k3** (CPU server, the real source of data) | **Clover-35B (coordinator) + Kimi K3.1 Pro (Clover's own engine, already running) + web and research tools** (full Clover system) | Split tasks, coordinate, reason deeply, and hold all conversations, memory and learning |
| **GX10** (GPU server, `ssh -p 40067 root@192.54.244.15`) | **Nemotron 3.5 Super VL, 4-bit only** (full Clover C1/C2/C3 + multimodal vision/audio/video) | Vision, audio, video. **Keeps no personal data** |
| **Later: a Kimi machine** | **the same Kimi engine** | Either rental shown has more RAM than `k3` and enough disk (test before committing) |

### Key Boundaries from Direction
1. **Use the `k3` versions of the full Clover system (`C1 + C2 HybridAgent/JEPA + C3 11-level NCM + S1/S2/S3`) on each model** rather than bare unintegrated model servers.
2. **`k3` is the real source of data**: holds all conversations, memory (`S1`), ontology (`S2`), and learning (`Growth`).
3. **`GX10` keeps no personal data**: stateless multimodal worker for vision, audio, and video (`Nemotron 3.5 Super VL, 4-bit only`); clients never send personal data directly to `GX10`—all Nemotron requests route **through `k3`**, and `GX10` persists zero conversations, memory, or media after a request completes.
4. **Clients (`Laptop, phone`) work offline**: local `LFM3-Omni-2B` answers simple requests locally and continues working when `k3` is unreachable.
5. **Step-by-step execution**: start with basic setup test of each node, observe what reality shows, update this context file after every cycle, and only then integrate.

---

## 2. Initial Context Established (Read-Only Survey Before Action)

### 2.1 Findings from Multi-Agent Architecture Research in Sibling Branches
1. **`research/superagent-jev-dory` ([01](superagent-jev-dory/01-deepseek-superagent-harness-context.md), [04](superagent-jev-dory/04-clover-three-capability-integration-context.md), [05](superagent-jev-dory/05-review-and-held-out-retest.md), [06](superagent-jev-dory/06-follow-up-items-1-4.md))**:
   - **What held and is in `main` (`clover-ai-system/studio/clover/superagent_jev_dory.py`, `tool_tasks.py`, `caution.py`)**:
     - `CloverSuperAgent`: 7-state task lifecycle (`created -> routed -> running -> waiting -> verifying -> completed | failed`), dependency-ready auto-routing, active-only `write_scope_warnings`, repeat-tool loop guard, and mandatory verification gate (`verify_and_complete_task` forbids worker self-certification).
   - **What failed held-out testing and must NOT be repeated**:
     - Using 1-pass logprob routing (`OpenJev`) on plain user messages failed the pre-registered 200-message gate (`0/50` on `clarify`, `0/50` on `verify`, and confident `act` routings on ambiguous requests). Therefore, client-to-`k3` escalation and `k3` coordinator task splitting must use explicit, inspectable criteria and `Clover-35B`'s structured task decomposition with `CloverSuperAgent` verification rather than raw 1-token logprob `act` authorization.
2. **`feature/work-board` (`scripts/work-board.py`)**:
   - Wraps `CloverSuperAgent` with file locking (`board.lock`) and replay-before-write so multiple processes/workers coordinate safely on a shared JSONL task board without duplicate task IDs or overlapping write scopes.
3. **`feature/personal-mesh` (`clover_events.py`, `clover_sync.py`)**:
   - Append-only per-device JSONL event logs (`events/<device>/<YYYY-MM>.jsonl`) ordered by `(lamport, device, seq)` and prefix-only sync (`sync(a, b)`) so offline client turns sync cleanly with `k3` without git merge losses.

### 2.2 Live State of `k3` (`root@k3`, AMD Ryzen 9 7950X3D, 16c/32t, 124 GiB RAM, 554 GiB free on `/dev/md2`)
- **Running Services**:
  - `clover-k3-chat.service` on `127.0.0.1:8765` (`/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py`): `status: "ready"`, `k3_trunks_resident: 93`, `k3_moe_layers_resident: 92`, `k3_experts_resident: 2576` (`Kimi K3.1 Pro` C/OpenMP engine + `HybridAgent` + `11-level NCM` + `S1/S2/S3`).
  - `clover-web-studio.service` on `127.0.0.1:8780` (`/opt/clover-k3/web-studio/clover/server.py`, `v5.0.6`): `status: "ready"`, serving `LFM3-Omni-2B` + `K3HostEngine` (`omni`, `clover-35b-a3b`, `nemotron-3.5-super-vl`, `kimi-k3`).
- **Precomputed Full-Clover Model Assets on `k3` & `Santhoshsathish94/clover-ai-studio`**:
  - **Kimi K3.1 Pro (`kimi-k3`)**: `/opt/clover-k3/kimi-k3.1-pro-computed/` (`3.1 GiB`: `pruned/`, `hybridagent-8gpu/kimi_k31_hybridagent_final.pt`, `hybrid-ncm/kimi_k31_hybrid_ncm_bridges.pt`, `tokenizer/`) + `/opt/clover-k3/clover-framework/clover-ai-system/src/datasets/computed` (`93` trunks + `92` expert layers).
  - **Clover-35B (`clover-35b-a3b`)**:
    - Code: `clover-ai-system/src/models/clover-35b-a3b/clover_35b_hybrid_ncm.py` (`Clover35BPrecomputedEngine`, `Clover35BNativeHybridLink` $d=2048$, `Clover35BComboBridge`, `Clover35BExpertBridge`) and `porting-models/clover_35b_gpu_backbone.py`.
    - Data on `k3`: `/opt/clover-k3/clover-35b-computed/hf-stage/` (`experts/vqvae-expert-0..10`, `ncm/*`), `/opt/clover-k3/four-node-20261112/Diffuspace_Clover-35B-A3B-Preview-Q4_K_M.gguf` (`20.78 GiB`, sha256 verified, measured `236.21 tok/s` pp512 and `12.75 tok/s` tg128 on 16 threads with `llama.cpp` `dd6b945`), plus `Santhoshsathish94/clover-ai-studio/clover-35b/` (`trunks/`, `experts/expert-0,3`, `hybridagent/clover_35b_hybridagent_final.pt`, `bridges/clover_35b_hybrid_ncm_bridges.pt`, `jepa/jepa-trunk-0`).
    - **Gap found in `k3` `clover_request_pipeline.py` (line 24)**: `CANONICAL_MODELS = ("nemotron-3.5-super-vl", "lfm3-1.7b", "kimi-k3")` omits `"clover-35b-a3b"`, causing `/api/chat` requests for `clover-35b-a3b` on `127.0.0.1:8765` to resolve to `nemotron-3.5-super-vl` instead of running `Clover-35B`!
  - **Nemotron 3.5 Super VL (`nemotron-3.5-super-vl`, 4-bit)**:
    - Code: `clover-ai-system/src/models/nemotron-3.5-super-vl/` (`nemotron_cpu_engine.c`, `nemotron_precomputed_engine.py`, `nemotron_hybrid_ncm.py`, `nemotron_multimodal.py`, `nemotron_workstation_clover.py`).
    - Data on `k3`: `/opt/clover-k3/nemotron-computed/pruned/` (`expert-1/experts.4bit.direct` `1.4 GiB`, `expert-mtp-1/experts.4bit.direct` `1.4 GiB`, `global-0/` `2.0 GiB`, `trunk-0,1,2,6,7,mtp-0,mtp-1` `~0.8 GiB`), `/opt/clover-k3/nemotron-computed/hybrid-ncm/` (`nemotron_hybrid_ncm_bridges.pt` `1.28 GiB` + `precomputed/` 4-bit NCM trunks), `/opt/clover-k3/nemotron-computed/hybridagent-8gpu/nemotron_hybridagent_final.pt` (`161 MB`), and `/opt/clover-k3/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/` (`tokenizer.json`, C-RADIO vision/video configs & weights).

### 2.3 Live State of `GX10` (`ssh -p 40067 root@192.54.244.15`, NVIDIA GB10 Grace Blackwell)
- **Hardware & OS**: `Linux aarch64` (20 Cortex-A725 cores), NVIDIA GB10 GPU (`CUDA 13.0`, driver `580.178.04`, `sm_121`), `121 GiB` unified RAM (`60 GiB` `/dev/shm` tmpfs, `13 GiB` free on `/` overlay).
- **What was left running on `GX10` by an earlier session**:
  - Three `llama-server` supervisor processes (`nemotron` Q2_K `53.7 GiB`, `clover35b` `21.9 GiB`, `lfm3omni` `2.6 GiB`) occupying `78.2 GiB` of unified memory, with raw GGUF weights instead of the `k3` full Clover system (`C1 + C2 HybridAgent + C3 11-level NCM + C-RADIO/EVS/Audio-Mamba multimodal engine`).
  - Per the human's Direction table, `GX10` runs **only `Nemotron 3.5 Super VL, 4-bit only`** (for vision, audio, video, keeping no personal data), while `LFM3-Omni-2B` runs on clients (`Laptop, phone`) and `Clover-35B + Kimi K3.1 Pro` run on `k3`.

---

## 3. Execution & Measured Outcomes

### 3.1 Cycle 1 — Precomputed `C1 (Inference) -> C2 (Planning) -> C3 (Decoding)` Across All 4 Models
Per the human's explicit requirement (*"every model should run on our complete efficient setup of c1 for inference c2 for planning and c3 for decoding and all precomputed datasets"*), we removed bare GGUF `llama-server` units (`clover-35b-llama.service` on `k3` and the old `clover35b`/`lfm3omni`/`nemotron` GGUF processes on `GX10`) and unified `OmniC2C3Runtime` ([clover-ai-system/studio/clover/omni_c2_c3.py](../../clover-ai-system/studio/clover/omni_c2_c3.py)), `CloverJEPARuntime` ([clover-ai-system/studio/clover/jepa_engine.py](../../clover-ai-system/studio/clover/jepa_engine.py)), and `LayerKit` ([clover-ai-system/studio/clover/lfm3_layers.py](../../clover-ai-system/studio/clover/lfm3_layers.py)) across all 4 model families:

| Model | Node | Vocabulary (`V`) | Hidden (`D`) | C1 Precomputed Inference | C2 Precomputed Plan (`plan-trunk-0` + `JEPA`) | C3 Precomputed Decode (`bridge-trunk-0` + `11L NCM`) | Measured `C1 / C2 / C3` (`k3` / `GX10`) |
|---|---|---|---|---|---|---|---|
| **`LFM3-Omni-2B`** (`omni`) | Laptop, phone (clients) & `k3` | `128,000` (`tokenizer.json`) | `2048` | `liblfm3_cpu_engine.so` (`trunk-0..23` + 4-bit FFN + `embed_tokens.bf16.bin`) | `Lfm3NativeHybridLink(2048->512->2048)` + `HybridLink2048` + `jepa-trunk-0` | `ComboBridge` & `ExpertBridge` (`2048->384`) + `11L VQ-VAE` + `12L NCM` + `3L Pointer` | `0.8 ms` (embed) / `1988 ms` (24L C1) · `C2: 49–80 ms` · `C3: 30–100 ms` |
| **`Clover-35B-A3B`** (`clover-35b-a3b`) | `k3` (Coordinator) | `248,320` (`tokenizer.json`) | `2048` | `Clover35BPrecomputedEngine` (`trunk-0,3..39` BD-GDN/GQA + 4-bit `experts.4bit.direct`, coreset `top_p=0.9`) | `Clover35BNativeHybridLink(2048->512->2048)` + `HybridLink2048` + `jepa-trunk-0` | `Clover35BComboBridge` & `ExpertBridge` (`2048->384`) + `11L NCM` (`vqvae_dec` std-calibrated) | `C1: 117.2 ms` · `C2: 57.0 ms` · `C3: 30.0 ms` |
| **`Kimi K3.1 Pro`** (`kimi-k3`) | `k3` (Deep reasoning; later dedicated Kimi node) | `163,840` (`tiktoken.model`, `PureTiktokenTokenizer`) | `7168` | `libk3_cpu_experts.so` + `libslotcache.so` (`93` trunks + `24` Gated DSA indexers + `embed_tokens.bf16.bin` + `norm.bf16.bin`) | `KimiNativeHybridLink(7168->2048->7168)` + `hl2048` + `jepa-trunk-0` | `KimiComboBridge` & `ExpertBridge` (`7168->384`) + `11L NCM` | `C1: 0.4 ms` (normed) · `C2: 41.2 ms` · `C3: 30.0 ms` |
| **`Nemotron 3.5 Super VL, 4-bit only`** (`nemotron-3.5-super-vl`) | `GX10` (Stateless GPU multimodal worker) | `131,072` (`tokenizer.json`) | `4096` | `libnemotron_cpu_engine.so` (`experts.4bit.direct` 4-bit only) + 32L `C-RADIO` CUDA (`vision-trunk-0..31`) + `EVS` video pruning + `AudioMambaBridge` | `NemotronNativeHybridLink(4096->1024->4096)` + `hl2048` + `jepa-trunk-0` | `NemotronComboBridge` & `ExpertBridge` (`4096->384`) + `11L NCM` (`2-layer copy_gate`) | `C1: 846.9 ms` (with 32L C-RADIO GPU + EVS + Audio) · `C2: 122.9 ms` · `C3: 100.0 ms` |

### 3.2 Cycle 2 — Stateless `GX10` 4-Bit `Nemotron 3.5 Super VL` Worker & `k3 -> GX10` Tunnel
1. **4-Bit-Only Precomputed Staging on `GX10` (`/opt/clover-gx10/nemotron-computed/hf-stage` in unified LPDDR5X `/dev/shm`, `8.3 GiB`)**:
   - Updated `nemo_engine_init` in [clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c](../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c) so 4-bit-only deployments (`experts.4bit.direct` present, `experts.direct` BF16 absent) initialize cleanly and default `use_4bit_experts = 1`.
   - Updated `NemotronCRadioVisionVideoEncoder` in [clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_multimodal.py](../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_multimodal.py) to load and execute all 32 ViT-Huge layers (`ncm/vision-global-0` + `ncm/vision-trunk-0..31`), `PixelShuffle(0.5)`, `ReLU^2` `mlp1` (`[1, 256, 4096]`), and `EVS` video token pruning directly from the precomputed `CLVRTRK1` binary trunks on the NVIDIA GB10 CUDA GPU (`61.1 ms` warm image encode, `134.2 ms` 4-frame video + EVS encode).
   - Built and deployed [clover-ai-system/src/models/nemotron-3.5-super-vl/gx10_nemotron_worker.py](../../clover-ai-system/src/models/nemotron-3.5-super-vl/gx10_nemotron_worker.py) under supervisor (`nemotron` on port `18082`), processing all images, video frames, and audio strictly in-memory (`io.BytesIO` + stdin-piped OCR) with `stateless_zero_personal_data: True` and `persisted_user_files: 0`.
2. **Persistent `k3 -> GX10` Tunnel (`clover-gx10-tunnel.service`)**:
   - Configured and started `clover-gx10-tunnel.service` on `k3`, forwarding `127.0.0.1:18082` on `k3` to `127.0.0.1:18082` on `GX10`.

### 3.3 Cycle 3 — Multi-Node Integration (`Laptop/Phone` ↔ `k3` Coordinator ↔ `GX10` Worker & Later Kimi Node Gate)
- Implemented [clover-ai-system/studio/clover/multi_node_coordinator.py](../../clover-ai-system/studio/clover/multi_node_coordinator.py) and wired `/api/multi-node/status`, `/api/multi-node/turn`, `/api/multi-node/sync`, and `/api/multi-node/client-turn` into [clover-ai-system/studio/clover/server.py](../../clover-ai-system/studio/clover/server.py):
  - **`ClientNodeRouter`**: runs simple requests locally on `LFM3-Omni-2B` (`C1 -> C2 -> C3`) and syncs conversation records to `k3`; works offline when `k3_reachable=False` by queuing turns in `client_offline_sync_queue.json` and flushing them to `k3` when back online; escalates larger/research requests to `k3`; routes image/audio/video requests for Nemotron through `k3` to `GX10`.
  - **`K3MultiNodeCoordinator`**: splits tasks across `GX10` (`vision_audio_video`), `k3` (`web_and_research_tools`), `Kimi K3.1 Pro` (`deep_reasoning`), and `Clover-35B-A3B` (`coordinator_synthesis`), and persists all conversations, memory, and learning on `k3`.
  - **`probe_kimi_rental_node`**: readiness gate verifying `RAM > 124 GiB` (more RAM than `k3`), `free_disk_gib >= 80 GiB`, and verified `Kimi K3.1 Pro` `C1 -> C2 -> C3` execution before committing to a dedicated Kimi rental node.
- Verified with [clover-ai-system/studio/tests/test_multi_node_clover_architecture.py](../../clover-ai-system/studio/tests/test_multi_node_clover_architecture.py) (`22/22 PASS` on `k3` with live `GX10` GPU worker; `13/13 PASS` on Windows laptop client including live `Laptop -> k3 -> GX10` Vision/Video/Audio), [clover-ai-system/studio/tests/test_c1_inference_only.py](../../clover-ai-system/studio/tests/test_c1_inference_only.py) (`36/36 PASS`), and [clover-ai-system/studio/tests/test_strict_c1_c2_c3.py](../../clover-ai-system/studio/tests/test_strict_c1_c2_c3.py) (`34/34 PASS`).

### 3.4 Cycle 4 — Real-World AI User Requests Testing & Fixes Across All 4 Models
We tested the live Studio Web App (`http://127.0.0.1:8765/`) across 10 real-world categories of what people use AI for (polite email writing, Python coding with examples, compound-interest math, multi-turn travel planning & recall, receipt/UI image inspection on `GX10` Nemotron, SQLite vs PostgreSQL comparison on `Clover-35B-A3B`, 4-week Rust study plan on `Kimi K3.1 Pro`, unit conversion, live web search on Python 3.13 free-threaded GIL, and laptop RAM/disk status). Testing exposed 5 concrete issues, each fixed and re-verified:

1. **Compound Interest & Financial Math (`$5,000 at 6% compounded annually for 3 years`)**:
   - *Observed before fix*: `k3:kimi-k3` guessed `$5,916.12` (`$916.12` interest) with `TOOLS: []`.
   - *Fix*: Added compound-interest routing in [semantics.py](../../clover-ai-system/studio/clover/semantics.py) & [mobile/semantics.py](../../clover-ai-system/mobile/semantics.py), deterministic S3 `calculate` pre-calls in [omni_native.py](../../clover-ai-system/studio/clover/omni_native.py), and `compound_interest_findings` verification in [clover_cycle.py](../../clover-ai-system/studio/clover/clover_cycle.py).
   - *Verified after fix*: Runs `calculate` twice (`5955.08` and `955.08`) and replies: `"You will have $5,955.08 at the end of 3 years, and the total interest earned is $955.08."` (`5.28 s`).
2. **Multi-Word & Multi-Target Unit Conversion (`Convert 72 miles per hour to kilometers per hour and meters per second`)**:
   - *Observed before fix*: Replied `115.8 km/h and 10.3 m/s` (`10.3 m/s` was wrong; `72 mph = 32.18688 m/s`) because `semantics.py` only matched single-word unit abbreviations (`mph`, `km/h`) and `local_tool_packs.py` lacked spelled-out `miles_per_hour` / `kilometers_per_hour` / `meters_per_second` keys.
   - *Fix*: Added spelled-out speed unit aliases in [local_tool_packs.py](../../clover-ai-system/studio/clover/local_tool_packs.py), expanded `("units",)` in `semantics.py`, and wired multi-target `convert_units` S3 pre-calls in `omni_native.py`.
   - *Verified after fix*: `72 mph = 115.872768 km/h; 72 mph = 32.18688 m/s.` (`0.24 s`, `TOOLS: ['convert_units', 'convert_units']`).
3. **Markdown Structure, Semicolon Alignment & Latency on `k3` Hosted Models (`Clover-35B-A3B`, `Nemotron 3.5`, `Kimi K3.1 Pro`)**:
   - *Observed before fix*: `/api/worker/chat` on `k3` forwarded `[VERIFIED_STUDIO_DRAFT]` to the legacy 48-token splitter, which stripped blank lines, turned `; pick` into `. pick` (because `has_semi` in `omni_c2_c3.py` disabled the `-45.0` 1-to-1 diagonal alignment), and took 25–32 seconds.
   - *Fix*: Wired `/api/worker/chat` in [server.py](../../clover-ai-system/studio/clover/server.py) to execute `OmniC2C3Runtime` (`512`-token window) on `k3` for `clover-35b-a3b` / `kimi-k3` and the stateless `GX10` 4-bit worker for `nemotron-3.5-super-vl`, and kept 1-to-1 diagonal pointer alignment in [omni_c2_c3.py](../../clover-ai-system/studio/clover/omni_c2_c3.py).
   - *Verified after fix*: `4-Week Rust Study Plan` (`k3:kimi-k3`) dropped from `32.29 s` to `12.62 s` with crisp bold headings and bullets; `SQLite vs PostgreSQL` (`k3:clover-35b-a3b`) dropped from `9.57 s` to `6.56 s` with semicolons and capitalization preserved (`C1=3.13ms | C2=19.46ms | C3=120.0ms`).
4. **Live Web Search Keyword & Wikipedia Snippet Preservation (`Python 3.13 GIL free-threaded mode`)**:
   - *Observed before fix*: `keywords()` in [web_search.py](../../clover-ai-system/studio/clover/web_search.py) split `3.13` at the dot and kept `"Search web"`, while `_search_wikipedia()` in [live-web-search-tool.py](../../clover-ai-system/studio/clover/live-web-search-tool.py) overwrote the matching section snippet (`since 3.13 ... optional "free-threaded mode"`) with the generic article intro.
   - *Fix*: Preserved dotted version numbers (`\d+\.\d+`) and stripped search command verbs in `keywords()`, required multi-term match in Bing Web RSS fallback, and preserved `search_snip` alongside `sum_txt` in `_search_wikipedia()`.
   - *Verified after fix*: Returns LWN/InfoWorld/Wikipedia free-threaded Python 3.13 sources and answers accurately how Python 3.13's optional free-threaded mode removes the GIL for parallel threads.
5. **Vision Snapshot Header Leak (`k3:nemotron-3.5-super-vl`)**:
   - *Observed before fix*: Reply started with `"The image is a Playwright-Style visual/UI snapshot..."`.
   - *Fix*: Clarified the S3 system prompt in `omni_native.py` and filtered echoed snapshot headers in `k3_host_engine.py`.
   - *Verified after fix*: Cleanly describes the receipt banner (`"CLOVERCAFE RECEIPT - TOTAL $42.50"`), red box (`"2x Espresso $8.00"`), green box (`"1x Sandwich $14.50"`), and palette in `12.74 s`.

### 3.5 Cycle 5 — All 4 Models Running Natively on `k3`, Smart Model Switching (`Auto (Smart)`), & App Builds (`Windows` + `Android`)
Per the human's direction (*"can you buld the current version into the apps so I can test in my actual device"* and *"have all the 4 models run in k3 and have all the models available for me in the app and it can smartly switch between models based on what the question is asked"*):
1. **All 4 Models Running Natively on `k3` (`C1 -> C2 -> C3`)**:
   - Enabled and started `clover-nemotron-k3.service` on `k3` (`127.0.0.1:18082`, backed by `/opt/clover-k3/nemotron-computed/hf-stage` with `libnemotron_cpu_engine.so` 4-bit `experts.4bit.direct` + 32-layer `C-RADIO` `vision-trunk-0..31` + `EVS` video pruning + `AudioMambaBridge`), alongside `LFM3-Omni-2B`, `Clover-35B-A3B`, and `Kimi K3.1 Pro` on `k3`.
2. **All Models Available Out-of-the-Box + Smart Model Switching (`select_smart_model_for_turn`)**:
   - Added `Auto (Smart)` (`id: "auto"`) and unlocked all 4 `k3` models (`LFM3-Omni (k3)`, `Clover-35B-A3B`, `Nemotron 3.5`, `Kimi K3.1 Pro`) plus local `LFM3-Omni` in `model_catalog()` ([server.py](../../clover-ai-system/studio/clover/server.py), [k3_host_engine.py](../../clover-ai-system/studio/clover/k3_host_engine.py)).
   - Implemented `select_smart_model_for_turn()` in `k3_host_engine.py` and wired it into `lfm3_turn()` in `server.py`:
     - **Images, video, audio & visual layout questions** $\rightarrow$ **`Nemotron 3.5`** (`k3:nemotron-3.5-super-vl`)
     - **Coding, debugging, algorithms, compound-interest/math proofs & study/research plans** $\rightarrow$ **`Kimi K3.1 Pro`** (`k3:kimi-k3`)
     - **Comparisons (`X vs Y`), task splitting, coordination, workflows & multi-step synthesis** $\rightarrow$ **`Clover-35B-A3B`** (`k3:clover-35b-a3b`)
     - **Everyday chat, short writing, quick arithmetic, unit conversion & local device tools** $\rightarrow$ **`LFM3-Omni`** (`omni`)
3. **Built & Verified Native App Packages (`Windows` & `Android`)**:
   - **Windows Laptop Installer**: [download/laptop/CloverAIStudio-Setup.exe](../../download/laptop/CloverAIStudio-Setup.exe) (`9,592,280` bytes, `SHA-256: 1ac5091885360d4fcd3eff0f0209fd25de7b4c556eda306928ebcff74ca2603b`, Authenticode Valid) + updated `$HOME\CloverAI` in place.
   - **Android Phone APK & AAB (`versionCode 20`, `targetSdk 36`, `5.0.6`)**: Built on `k3` (`build_apk.sh phone release`), verified byte-identical `clover.zip` inside `assets/chaquopy/app.imy`, and staged at [download/mobile/CloverAIStudio.apk](../../download/mobile/CloverAIStudio.apk) (`82,639,595` bytes, `SHA-256: 4568fca5d957ed5eaf4c789879e8224dae820393b97a03e7ccab3596e1083d3a`) and `https://api.cloverframework.com/download/mobile/CloverAIStudio.apk`.

---

## 4. Growth (Lessons Recorded)

1. **Precomputed `CLVRTRK1` / `CLVR4B01` uniformity across model dimensions (`D = 2048, 4096, 7168`)**:
   - All 4 models share identical tensor schemas in `plan-trunk-0`, `bridge-trunk-0`, and `ncm/*`, differing only in `(vocab_size, d_model)`, `copy_gate` depth (`copy_gate.0/2` 2-layer MLP in Nemotron vs 1-layer in LFM3/Clover-35B/Kimi), and `Clover-35B`'s unit-layer-normed `vqvae_dec.bf16.bin` (`std = 0.9835` vs `0.055–0.085`, calibrated on slice lookup at zero extra RAM cost).
2. **`/dev/shm` `noexec` vs `PROT_READ` data mmaps on container GPU nodes (`GX10`)**:
   - On `GX10`, `/dev/shm` has `60 GiB` of fast LPDDR5X tmpfs but is mounted `noexec`. Placing data trunks (`nemotron-computed/hf-stage`, mapped with `PROT_READ`) in `/dev/shm` while keeping compiled `.so` libraries (`/opt/clover-gx10/app/nemotron-3.5-super-vl/libnemotron_cpu_engine.so`, mapped with `PROT_EXEC` by `dlopen`) on `/` avoids `failed to map segment from shared object` while keeping all `8.3 GiB` of 4-bit weights in RAM.
3. **Hugging Face `/api/datasets/.../tree/main/<prefix>?recursive=true` 1,000-item pagination limit**:
   - Recursive tree listings on large model folders (`kimi/` `1,712` files, `nemotron/` `1,150+` files) truncate at 1,000 entries (`trunk-6`, `trunk-7`, `trunk-mtp-*`, `global-0` were omitted in a top-level query). Always query explicit subpaths (`nemotron/trunks/trunk-7`, `kimi/trunks/global-0`, `kimi/ncm`) when staging specific trunks.
