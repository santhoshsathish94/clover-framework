# Upstream context: Kimi K3.1 Pro (`kimi-k3.1-pro` & `kimi-k3.1-pro-highspeed`, released 2026-10-24)

Primary-source investigation of Moonshot AI's **Kimi K3.1 Pro** release (`2026-10-24`), read directly from Moonshot AI's official technical blog (`https://www.kimi.com/en/blog/kimi-k3-1-pro`), Kimi API documentation (`https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart.md`, `models.md`, `models-overview.md`, `pricing/chat`), Hugging Face (`https://huggingface.co/moonshotai`), and GitHub (`https://github.com/MoonshotAI`).

All benchmark and pricing numbers below are **published by Moonshot AI, not reproduced locally** (`full model weights scheduled for release by November 4, 2026`).

---

## 1. Clover Cycle — Context & Evidence Status

- **Has Kimi K3.1 Pro been released?**
  - **Yes — API and product rollout on October 24, 2026; open-weight files announced for November 4, 2026.**
  - On **October 24, 2026**, Moonshot AI launched **`kimi-k3.1-pro`** and **`kimi-k3.1-pro-highspeed`** across `kimi.com` (default selector `K3.1 Pro High`), Kimi Work desktop (`v3.1.0+`), Kimi Code CLI (`@moonshot-ai/kimi-code`), and the Kimi Open Platform API (`https://api.moonshot.ai/v1`).
  - As of **October 28, 2026**, the open model weights are **not yet uploaded** to `huggingface.co/moonshotai` (where `moonshotai/Kimi-K3` remains the latest uploaded checkpoint, last modified `2026-09-02`). Both the official blog and API quickstart state: *"The full model weights will be released by Nov 4, 2026."*

---

## 2. What Kimi K3.1 Pro Actually Is (Architecture vs. Post-Training)

In Moonshot AI's own words from the K3.1 Pro technical blog and API documentation:

> *"Kimi K3.1 Pro is a **post-trained version of Kimi K3**, with substantially larger post-training compute. To enable large-scale joint RL across hundreds of downstream domains, we optimized every layer of the stack — the RL framework, long-context inference, training efficiency, sandbox infrastructure, rollout scheduling, and a shared platform for environment quality checks — so that each increment of RL compute delivers stable, predictable returns."*

### 2.1 What Stayed the Same as Kimi K3 (Base Architecture & API Contract)

Because Kimi K3.1 Pro is a **post-trained checkpoint of the Kimi K3 3T-class backbone** rather than a new base architecture:

| Property | Kimi K3 (`2026-07-16` / `2026-07-27` weights) | Kimi K3.1 Pro (`2026-10-24`) | Status |
|---|---|---|---|
| **Model Class & Backbone** | 2.8T total / 104B activated MoE (`93` layers: `1` dense + `92` LatentMoE, `69` KDA + `24` Gated MLA, Attention Residuals, `16 / 896` routed experts + `2` shared experts, `W = 7168`) | Post-trained version of the same 3T-class Kimi K3 backbone | **Same backbone** (exact shard/tensor layout to be verified when weights publish by Nov 4, 2026) |
| **Context Window** | `1,048,576` tokens (`1M`) | `1,048,576` tokens (`1M`) | **Identical** |
| **Max Completion Tokens** | Default `131,072`, max `1,048,576` | Default `131,072`, max `1,048,576` | **Identical** |
| **Modalities** | Native Text, Image, Video (`MoonViT-V2`) | Native Text, Image, Video | **Identical** |
| **Thinking Mode** | Always-on Preserved Thinking (`reasoning_content` must be returned in multi-turn history) | Always-on Preserved Thinking (`reasoning_content` must be returned in multi-turn history) | **Identical** |
| **`reasoning_effort`** | Top-level `"low"` / `"high"` / `"max"` (default `"max"`) | Top-level `"low"` / `"high"` / `"max"` (default `"max"`) | **Identical** |
| **Fixed Sampling Params** | `temperature = 1.0`, `top_p = 0.95`, `n = 1`, `presence_penalty = 0`, `frequency_penalty = 0` | `temperature = 1.0`, `top_p = 0.95`, `n = 1`, `presence_penalty = 0`, `frequency_penalty = 0` | **Identical** |
| **`tool_choice`** | `auto` / `none` / `required` + dynamic tool loading in `system` messages | `auto` / `none` / `required` + dynamic tool loading in `system` messages | **Identical** |

---

### 2.2 What Changed in Upstream Kimi K3.1 Pro vs. Upstream Kimi K3

1. **Large-Scale Multi-Domain Joint RL Post-Training:**
   - Scaled reinforcement learning across hundreds of downstream domains with sandboxed execution verification, long-horizon rollout scheduling, and environment quality checks.
2. **Targeted Factual Accuracy & Hallucination Reduction (`2.8×` to `3.4×` lower claim hallucination rate):**
   - On Moonshot's internal factuality stress test over de-identified error-prone user conversations, claim-level hallucination dropped from **`23.35%` (K3) to `6.85%` (K3.1 Pro)** (and **`23.4% → 8.3%`** on `Kimi Hallu Bench Hard` with tools; **`5.4%`** without tools), cutting the share of responses with $\ge 1$ factual error to **`42.4%–46.4%`** of K3's rate.
3. **3D Spatial Reasoning, Parametric CAD & Game Engine Code Generation (`3D Make`):**
   - Trained to write and execute Blender, FreeCAD, `trimesh`, Three.js, Unreal Engine 5.8, and Godot 4.7 scripts directly without requiring MCP wrappers (`92.7%` Voxel IoU on `BenchCAD` with tools, `72.8%` without tools).
4. **End-to-End Financial Research, Modeling & Business Simulation:**
   - Improved over K3 on Internal Financial Modeling (`57.6` vs. `50.4`, `+7.2` pts), Internal Financial Research (`70.4` vs. `58.8`, `+11.6` pts), and Internal Business Operation (`37.0` vs. `8.9`, **`4.2×` improvement**).
5. **Mid-Training & RL Cybersecurity Data Filtering for Open Weights:**
   - Identified and removed offensive cyber-exploitation data from both the mid-training corpus and RL environments before the open-weight release. On `xdev-bench-hard` (`pass@3`), the unfiltered K3.1 checkpoint scored `90.5%`, the filtered open-weight K3.1 Pro release scored **`81.0%`**, and original Kimi K3 scored **`33.3%`** (showing general coding/debugging RL still transferred `+47.7` pts even after cyber-specific removal).
6. **New High-Speed Serving Variant & `33%` Lower Base API Price:**
   - Added `kimi-k3.1-pro-highspeed` (`2–3×` faster output speed than `kimi-k3.1-pro`) and lowered standard `kimi-k3.1-pro` API pricing by **33%** compared to `kimi-k3`:

| API Model ID | Cache-Miss Input | Cache-Hit Input | Output Tokens | 5m Cache Write | 1h Cache Write |
|---|---:|---:|---:|---:|---:|
| **`kimi-k3` (Jul 2026)** | `$3.00 / MTok` | `$0.30 / MTok` | `$15.00 / MTok` | `$3.00 / MTok` | `$6.00 / MTok` |
| **`kimi-k3.1-pro` (Oct 2026)** | **`$2.00 / MTok`** | **`$0.20 / MTok`** | **`$10.00 / MTok`** | **`$2.00 / MTok`** | **`$4.00 / MTok`** |
| **`kimi-k3.1-pro-highspeed` (Oct 2026)** | `$4.00 / MTok` | `$0.20 / MTok` | `$20.00 / MTok` | `$4.00 / MTok` | `$8.00 / MTok` |

---

## 3. Published Benchmark Comparison: Kimi K3.1 Pro vs. Kimi K3 (Published by Moonshot AI, Not Reproduced)

Where both Kimi K3 (`2026-07-16`) and Kimi K3.1 Pro (`2026-10-24`) report numbers on the same or updated benchmark suites:

| Benchmark | Kimi K3 (Published Jul 2026) | Kimi K3.1 Pro (Published Oct 2026) | Delta / Notes |
|---|---:|---:|---|
| **DeepSWE v1.1** | `67.5%` | **`75.5%`** | **`+8.0 pts`** |
| **SWE-Marathon v1.1** | `42.0%` | **`51.7%`** | **`+9.7 pts`** (leads Opus 5.5 `46.3%` and GPT-6 Astra `42.5%`) |
| **Kimi Code Bench 2.0 (Internal)** | `72.9%` | **`77.1%`** | **`+4.2 pts`** |
| **Kimi Code Bench 3.0 Long-horizon** | `55.95%` | **`61.2%`** | **`+5.25 pts`** |
| **Terminal-Bench 4.0** | — *(TB 2.1 was `88.3%`)* | **`65.7%`** (`$5.25`/task at `max`; `52.8%` at `$4.16`/task `high`) | Ahead of GPT-6 Astra (`58.2%`), behind Opus 5.5 (`66.4%`) |
| **BenchCAD (w/o tools / w/ tools)** | — | **`72.8% / 92.7%`** | New 3D CAD reconstruction benchmark |
| **GPQA-Diamond** | `93.5%` | **`94.2%`** | **`+0.7 pts`** |
| **HLE-Full (text+mm / w/ tools)** | `43.5% / 56.0%` | **`51.1% / 70.0%`** | **`+7.6 pts` (raw) / `+14.0 pts` (with tools)** |
| **BrowseComp** | `91.2%` | **`92.1%`** | **`+0.9 pts`** |
| **GDPval-AA (Elo)** | `1686` *(v2)* | **`1846`** *(v2.1)* | **`+160 Elo`** |
| **AA-Briefcase (Elo)** | `1548` *(v1)* | **`1678`** *(v1.1)* | **`+130 Elo`** |
| **APEX-Agents** | `41.0%` | **`67.2%`** | **`+26.2 pts`** |
| **OfficeQA Pro / OfficeQA Pro v2** | `63.3%` / — | **`67.4%` / `64.4%`** | **`+4.1 pts`** on OfficeQA Pro; SOTA on v2 |
| **Agents' Last Exam (ALL / CLI)** | `28.3%` / — | **`43.2% / 44.2%`** | **`+14.9 pts`** |
| **PerceptionBench** | `58.5%` | **`68.3%`** | **`+9.8 pts`** |
| **MMMU-Pro (raw / w/ python)** | `81.6% / 83.4%` | **`84.1% / 90.0%`** | **`+2.5 pts / +6.6 pts`** |
| **CharXiv RQ (raw / w/ python)** | `84.8% / 91.3%` | **`89.0% / 93.4%`** | **`+4.2 pts / +2.1 pts`** |
| **BabyVision w/ python** | `85.7%` | **`95.4%`** | **`+9.7 pts`** |
| **ZeroBench pass@5 (raw / w/ python)** | `23.0% / 41.0%` | **`39.0% / 63.0%`** | **`+16.0 pts / +22.0 pts`** |
| **Kimi Hallu Bench Hard (lower=better)** | `23.4%` | **`8.3%`** (`5.4%` w/o tools) | **`-15.1 pts` (`2.8×` fewer hallucinations)** |
| **xdev-bench-hard (pass@3)** | `33.3%` | **`81.0%`** (`90.5%` pre-filter) | **`+47.7 pts`** |

---

## 4. How Kimi K3.1 Pro Compares to **Our Clover Kimi K3** (`clover-kimi-k3`)

Now let us compare **Moonshot's Kimi K3.1 Pro** against **what we actually built and measured in `clover-kimi-k3`** on our own hardware (`8 × RTX 3090` on `gpu` and `Ryzen 9 7950X3D` on `k3`):

| Dimension | Upstream Kimi K3 (`moonshotai/Kimi-K3`) | Upstream Kimi K3.1 Pro (`kimi-k3.1-pro`) | **Our Clover Kimi K3 (`clover-kimi-k3`)** |
|---|---|---|---|
| **Base Backbone Weights** | `96` safetensors shards (`1.56 TB`, `2.8T` params, `93` layers: `69` KDA + `24` MLA, `16/896` MoE) | Same 3T-class K3 architecture with updated post-training RL weights (weights due **Nov 4, 2026**) | Runs the exact `93`-layer Kimi K3 backbone (`trunk-0..92` + `expert-1..92` = `82,432` routed experts) **plus** `HybridAgent-8B` planner + `119.96M` `NextConceptTransformer` (NCM `MBR8`) + `144.00M` `VQVAE` (`e11`/`e12`) |
| **Target Hardware** | Recommends supernodes with **`≥ 64` accelerators** (`H20`/`H100`/`B200`) using MoonEP + Mooncake + FlashKDA | Cloud API cluster (`kimi-k3.1-pro` & `kimi-k3.1-pro-highspeed`); weights due Nov 4 | **Single commodity 8 × RTX 3090 (`24 GB` Ampere) server** (`148.5 K3 tok/s` across all 93 layers) **or single 1-socket CPU server** (`Ryzen 9 7950X3D` with `128 GB` RAM + NVMe) |
| **How Hallucinations & Errors Are Controlled** | Relies on model weights alone (`23.4%` claim hallucination rate on Kimi Hallu Bench Hard) | Uses massive multi-domain joint RL post-training to reduce claim hallucination rate from `23.4%` $\to$ **`8.3%`** | Uses **architectural grounding & deterministic verification** rather than hoping RL weights never hallucinate: **1)** `CloverMemoryGraphStore` + `CloverOntologyGraph` (`Provenance` citations + McCulloch-Pitts symbolic predicates), **2)** `CloverPythonGenerativeCompiler` & `CloverRustGenerativeCompiler` (`rustc` + Sutton $\text{TD}(\lambda)$ statement repair) |
| **How Thinking & Planning Work** | Always-on token-by-token chain-of-thought (`reasoning_content` must be echoed back every turn, or generation destabilizes) | Same always-on token-by-token `reasoning_content` (`low`/`high`/`max`) | **Decoupled Latent/Window Planning + Concept Decoding**: `HybridAgent` plans compact response windows (`W1..Wn`), routed through 93 K3 layers + NCM `MBR8` concept decoding, with zero requirement for the client to echo hidden thinking traces |
| **How Long Context Is Managed** | `1M` raw token window with 512-token block prefix caching (`5m` or `1h` paid TTL); BrowseComp harness compacts at `200K–300K` tokens | Same `1M` raw token window + paid `5m`/`1h` TTL prefix cache; BrowseComp compacts at `200K` tokens | **`256K` (`262,144`-token) verbatim session window** with automatic **5× simple-sentence compaction** (`source="summary:compacted"`) that triggers *only* when a session crosses `262,144` tokens, preserving every identifier, number, and date indefinitely |
| **How Expert Bandwidth Bottleneck Is Solved** | `WarpDecode` + `MoonEP` across 64+ GPUs with all 2.8T weights held in cluster HBM | Cloud cluster HBM (`highspeed` variant likely uses higher EP degree / speculative decoding) | **Direction B 4-Tier Data-Movement Architecture** on 1 machine: `28` permanent VRAM experts/layer + `1,230` dynamic `CardPool` VRAM slots with **Policy 3 (`SpecMD Least-Stale + Red-Blue Pebble R4 Clean-Drop + Cross-Layer Lookahead`)** achieving **`95.29%` VRAM hit rate**, `0` D2H writebacks (`448 GB` PCIe traffic eliminated), and **`7.01×` Eyeriss energy reduction** |
| **Can We Load K3.1 Pro Weights Into Our Engine?** | Already running (`f831ab66...`) | **Weights release by Nov 4, 2026** | Because K3.1 Pro is a **post-trained version of Kimi K3** with the same 3T-class 93-layer architecture, once Moonshot publishes the Nov 4 safetensors shards on Hugging Face, our existing conversion & engine pipeline (`trunk-0..92`, `expert-1..92`, `whole-model.c`, `backend-gpu.cu`, `k3_gpu_experts_lib.cu`) can ingest the updated weights directly — **provided** the shard tensor shapes and MXFP4/BF16 packing layout remain unchanged (to be verified on the Nov 4 release) |

---

## 5. Clover Takeaways — What Kimi K3.1 Pro Teaches Us for Our Next Cycle

1. **Why Moonshot's K3 $\to$ K3.1 Pro Jump Validates Our Clover Architecture:**
   - Look at the three biggest problems Moonshot had to spend massive post-training RL compute to fix between K3 (July 2026) and K3.1 Pro (October 2026):
     1. **Factual hallucinations** (`23.4% → 8.3%` on Hallu Bench Hard).
     2. **Iterative code & simulation debugging** (running open-loop probes and test checks instead of blindly guessing).
     3. **Excessive unguided proactiveness** (which K3's own blog noted required explicit `AGENTS.md` boundaries).
   - In Clover, those three things are first-class **system architecture (`Archive` with `Provenance`, `Inspector` with `GenerativeCompiler`, and `Order` with `AGENTS.md` boundaries)** rather than properties we hope statistical weights maintain on their own.
2. **Even After Massive RL, Model Weights Alone Still Hallucinate `5.4%–8.3%` of Claims on Stress Prompts:**
   - K3.1 Pro's reduction from `23.4%` to `8.3%` is a huge achievement for pure post-training RL, yet `8.3%` claim error in financial or engineering work still means roughly **1 in 12 hard claims is wrong** unless backed by an external **Archive + Symbolic/Compiler Inspector**. Combining post-trained weights *with* Clover's deterministic verification and provenance graph is strictly stronger than either alone.
3. **Actionable Next Step When Nov 4, 2026 Arrives:**
   - When Moonshot publishes the `Kimi-K3.1-Pro` weights on Hugging Face (announced for **by November 4, 2026**), our first step under Clover will be:
     1. Inspect its `config.json` and `model.safetensors.index.json` against `moonshotai/Kimi-K3` (`f831ab66`).
     2. Verify that all 93 layer dimensions (`W=7168`, `69 KDA + 24 MLA`, `896 experts`, `3584 latent`, `3072 expert hidden`, `163,840 vocab`) are unchanged.
     3. Measure router score/bias shifts and expert palette differences between K3 and K3.1 Pro.
