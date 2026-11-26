# Clover Research Context: 5 Deterministic Senses + Thin Aligned Converters + 1 Brain + Tools

- **Created:** 2026-11-26
- **Status:** **RECORDED & PAUSED — Next Major Milestone for Clover AI** *(7-Horizon Roadmap & Cycles 1–7 preserved here; active execution focused on HG Format & 6-Myth Public Proof Release)*
- **Governing Contract:** [AGENTS.md](../../AGENTS.md) (`Context → Direction → Execution → Outcome → Growth`)
- **Human Direction:**
  1. *"What must be computed belongs in prebuilt tools (`x-cruncher` class); never train neurons to predict what an algorithm can compute."*
  2. *"Humans have 5 senses and 1 brain that processes all senses and learns to use tools — and those 5 senses can be deterministic libraries + thin converters into 1 Brain (`AGENTS.md` + `mcp.json`)."*
  3. *"Even before starting implementation, research more, argue with `Claude Opus 5.5`, verify existing published results on the internet, maintain a Context File, and run Clover cycles so each cycle puts us in a better state before starting work."*

---

## Baseline State Before Cycle 1 (What We Believed & What Was Unverified)

- **Prior tentative hypothesis (from Debate Turns 7–8):**
  - Use `Qwen/Qwen3-Omni-30B-A3B-Instruct` as the single 1-GPU Coordinating Brain based on secondary summaries of its abstract claiming *"zero modality degradation"* across text, vision, audio, and tool calling.
  - Known repo bugs already caught in Turn 3 & Turn 8:
    1. [lfm3_multimodal.py](../../clover-ai-system/src/models/lfm3-1.7b/lfm3_multimodal.py) (`Lfm3ShortConvAudioVoiceAdapter`, `Lfm3ShortConvVisionVideoAdapter`) and [nemotron_multimodal.py](../../clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_multimodal.py) (`NemotronAudioMambaBridge`) initialize converters with **untrained `manual_seed(42)` random weights** (valid tensor shape, zero semantic alignment).
    2. [studio_epistemic_memory.py](../../clover-ai-system/src/tools/hg4-fusion-gateway/studio_epistemic_memory.py) lacks a `rejected` lifecycle state and lets `unless` exceptions overwrite base rules (`0.286` vs `1.000` on `Probe 7A` until replaced by a 30-line `0B` defeasible rule resolver).
    3. Deployed C `HG4` gateways returned `sees_image_pixels = false` on `Probe 5` (`0/3` engines had pixel converters wired in).
- **Open risks entering Cycle 1:**
  - We had **not** inspected the primary-source numeric tables inside `Qwen3-Omni` (`arXiv:2509.17765v1`), `Voxtral` (`arXiv:2507.13264v1`), `SLAM-ASR` (`arXiv:2610.20844`), `Qwen3.6-35B-A3B`, or `Gemma 4` (`arXiv:2607.02770`).

---

## Cycle 1 — Primary-Source Audit: Does an All-in-One Omni Brain Degrade Reasoning & Tool Use?

### 1.1 Context & Execution
Fetched primary HTML tables directly from `arXiv:2509.17765v1` (*Qwen3-Omni Technical Report*, Tables 1–15), `arXiv:2507.13264v1` (*Mistral Voxtral*, Table 1 & Section 3), `arXiv:2610.20844` (*EMNLP 2026 Zero-Shot SLU with SLAM-ASR*), plus 2026 technical specifications and benchmarks for `Qwen/Qwen3.6-35B-A3B` and `Google Gemma 4` (`arXiv:2607.02770`, April 2026, Apache-2.0).

### 1.2 Outcome — Four Assumptions Overturned by Primary Data

1. **`Qwen3-Omni-30B-A3B` DOES suffer severe reasoning, logic, and tool-calling degradation in its own Tables 4 & 5 (`arXiv:2509.17765v1`):**

   | Benchmark (Capability) | `Qwen3-30B-A3B-2507` (Text Brain) | `Qwen3-Omni-30B-A3B` (Unfrozen Joint Omni) | Delta |
   | :--- | :---: | :---: | :---: |
   | **`BFCL-v3` (Agent Tool Calling, Thinking — Table 5)** | **`72.4`** | `63.2` | **`-9.2` pts** |
   | **`AIME25` (Math Reasoning, Thinking — Table 5)** | **`85.0`** | `73.7` | **`-11.3` pts** |
   | **`ZebraLogic` (Constraint Logic, Instruct — Table 4)** | **`90.0`** | `76.0` | **`-14.0` pts** |
   | **`LiveBench` (Reasoning/Coding — Table 5)** | **`76.8`** | `71.8` | **`-5.0` pts** |
   | **`PolyMATH` (Multilingual Math — Table 5)** | **`52.6`** | `47.1` | **`-5.5` pts** |
   | **Pretrained Context Window (Section 3, Stage S3)** | `262,144` (`256K`) | **`32,768` (`32K` S3; `65K` max)** | **`4–8×` shorter** |
   | **Spoken South Asian Languages (Table 3, 19 ASR / 10 TTS)** | — | `ur` (Urdu ASR only); **`0` Tamil, Hindi, Kannada, Telugu, Malayalam, Bengali** | **Missing major Indic languages** |

   *Evidence note (`[Primary-Paper]` `arXiv:2509.17765v1` Tables 3–5, refined in Cycle 3):* Because `Qwen3-30B-A3B-2507` received a later upgraded post-training recipe than `Qwen3-Omni`'s base, the `-9.2` / `-11.3` / `-14.0` gap mixes multimodal trade-offs with `2507` post-training gains (cause unattributed). Regardless of cause, `Qwen3-Omni-30B-A3B` trails stronger 2026 brains on Clover's core agentic tool & reasoning work.

2. **2026 Single-GPU Brains (`Qwen/Qwen3.6-35B-A3B` & `Google Gemma 4 26B-A4B / 31B Dense`, Apache-2.0) Are Vastly Stronger Reasoning + Vision Coordinators:**
   - **`Qwen/Qwen3.6-35B-A3B`** (`Apache-2.0`, `35B` total / `3B` active, `40` layers `3:1` Gated-DeltaNet + Gated-GQA, `262K` native -> `1M` extensible): Already ships on Hugging Face as `AutoModelForImageTextToText` with a **co-trained native 3D-RoPE Vision Encoder + Projector** (`MMMU-Pro = 75.1%`, `TAU2-Bench Agent Tool Use = 81.2`)! Our earlier `Diffuspace/Clover-35B-A3B-Preview` port only extracted the text trunk, leaving its native trained vision converter unmounted.
   - **`Google Gemma 4` (`arXiv:2607.02770`, April 2026, `Apache-2.0`):**
     - `Gemma 4 26B-A4B MoE` (`25.2B` / `3.8B` active, `~16 GiB` 4-bit, `256K` ctx, Text+Vision+Tools): `AIME 2026 = 88.3%`, `LiveCodeBench v6 = 77.1%`, `GPQA Diamond = 82.3%`, `TAU2-Bench = 85.5%`, `MMMU-Pro = 73.8%`.
     - `Gemma 4 31B Dense` (`30.7B` dense, `~18 GiB` 4-bit — **fits 1× 24 GiB RTX 3090!**, `256K` ctx): `AIME 2026 = 89.2%`, `LiveCodeBench v6 = 80.0%`, `GPQA Diamond = 84.3%`, `TAU2-Bench = 86.4%`, `MMMU-Pro = 76.9%` (**`8×–10×` more active parameters per token than `3B`-active MoE on hard single-pass derivations!**).
     - `Gemma 4 E4B` (`4.5B`, `~3.6 GiB` 4-bit, `128K` ctx): Native Text + Image + **Raw Audio** + Video + Tools for mobile/edge.

3. **Exact Physics of Thin Audio Converters (`Voxtral` `arXiv:2507.13264v1` vs `SLAM-ASR` `arXiv:2610.20844`):**
   - Training a linear projector *only* on ASR transcription (`SLAM-ASR` `<repeat>`) creates a **phonetic bottleneck**: `3–5%` WER in-domain, but **`23%` vs `71%` collapse on zero-shot Spoken Language Understanding (`arXiv:2610.20844`)**!
   - `Voxtral` (`arXiv:2507.13264v1` Table 1) solved this with `128`-Mel DSP (`0B`) → `640M` encoder → **`25M–52M` `4×` MLP Downsampler (`12.5 Hz` = `450` tokens/min)** trained on `50% <repeat>` + `50% <next>` cross-modal continuation.

### 1.3 Growth From Cycle 1 (How Our State Improved)
- **Rejected naive swap to `Qwen3-Omni-30B-A3B` as sole Brain:** Losing `-9.2` `BFCL-v3` tool points, `-14.0` `ZebraLogic` points, `256K -> 32K` context, and all Indian languages would have **weakened** Clover's core Brain!
- **Better Architectural State Entering Cycle 2:** Keep the **Brain** (`Qwen3.6-35B-A3B` MoE for speed OR `Gemma 4 31B Dense` / `26B-A4B MoE` — both Apache-2.0, `256K` context, native Vision+Tools, `81–86+` `TAU2` tool score!) **100% uncompromised on reasoning & tool-use**, mount its **official co-trained Vision Converter**, and audit in **Cycle 2** how each of the 5 Senses (`See`, `Hear`, `Speak`, `Multilingual Script`, `Act/Judge`) should connect via `0B` deterministic libraries + thin converters without degrading the Brain!

---

## Cycle 2 — Sense-by-Sense Audit of Published 2025–2026 Leaderboards (`0B` Libraries vs Thin Converters)

### 2.1 Context & Execution
Audited published 2025–2026 leaderboards across **Document & Scene OCR** (`OmniDocBench v1.5` & `Real5-OmniDocBench`), **High-Res GUI Grounding** (`ScreenSpot-Pro` `arXiv:2504.07981` & `ScreenSpot v2`), **Multilingual & Indian Speech** (`AI4Bharat IndicVoices / IndicSUPERB / Lahaja` `arXiv:2403.01926`), and **Multilingual Morphology** (`SIGMORPHON` & Universal Dependencies `HFST` vs `Stanza`/`UDPipe`).

### 2.2 Outcome — Exact Published Numbers Across All 5 Senses

1. **Sense 1 — Vision (`See`: Layout, GUI & Text-in-Images):**
   - **GUI / Screen Grounding (`ScreenSpot-Pro` 4K professional CAD/IDE apps, `arXiv:2504.07981`):** Even `Qwen3-VL-235B-A22B` drops to **`~62.0%`** from raw pixels alone (`ScreenSpot v2` standard GUI = `95.8%` on `Qwen3-VL-32B`), whereas **`0B` `Playwright` DOM / OS Accessibility (`UIAutomation` / `AT-SPI`) yields `100.0%` exact element coordinates** whenever a DOM/a11y tree exists!
   - **Warped / Photographed / Low-Light Text-in-Images (`Real5-OmniDocBench`, 2026):** **`PaddleOCR-VL-1.6` (`~0.9B`, Apache-2.0) scores `93.19`**, **beating `250×` larger `Qwen3-VL-235B-A22B` (`88.90`) and `MinerU2.5-Pro` (`88.94`) by `+4.3` points**!
   - **Born-digital PDFs / SVGs / QR:** `0B` `PyMuPDF` + `zbar` = `100%` lossless in `<5 ms`.
   - **Visual Scene / Chart Reasoning:** Handled directly by the Brain's own co-trained Vision Converter (`Gemma 4 31B` `MMMU-Pro = 76.9%`, `Qwen3.6-35B-A3B` `MMMU-Pro = 75.1%`).

2. **Sense 2 — Hearing & Voice (`Hear & Speak`):**
   - **All 22 Official Indian Languages (`[HF-Card]` `ai4bharat/indic-conformer-600m-multilingual`):** Exact verified checkpoint is **`600M` parameters, `MIT` license** (correcting a secondary blog's `~30M` claim), trained across `IndicVoices / Kathbath / Shrutilipi`, outperforming `Whisper-large-v3` across Tamil, Hindi, Kannada, Telugu, Malayalam, etc.
   - **Global 99-Language Speech + Prosody:** `0B` `Silero VAD` (MIT) + `0B` **Deterministic Acoustic Prosody Extractor** (`F0` contour, pause durations `ms`, speaking rate, RMS energy dB) + `indic-conformer-600m-multilingual` (`0.6B`, MIT) / `Whisper-large-v3-turbo` (99 langs) + `Kokoro-82M` (Apache-2.0) / `IndicParler-TTS`.

3. **Sense 3 — Multilingual Script & Morphology (`Read & Write`):**
   - `0B` `HFST` / `Apertium` finite-state analyzers give exact zero-hallucination morphological splits on canonical vocabulary, while **`Stanza` / `UDPipe 2` (`~50–100M`, Apache-2.0)** resolves context ambiguity & code-mixed Tanglish/Hinglish (`>95%` UD accuracy on high-resource, top SIGMORPHON neural disambiguator) before entering the Brain's `248K–256K` multilingual vocabulary.

4. **Senses 4 & 5 — Motor Action & Proprioception (`Act` & `Judge`):**
   - `0B` `TokenEconomizer` + `mcp.json` (`x-cruncher`, `0B` `StatsForecast`/Periodogram `MASE 0.3084` `[Measured-on-k3]`, `0B` `Lean 4` tactic cascade `6/12` MiniF2F `[Measured-on-k3]`, compilers, `OpenGL/Playwright`, `ripgrep/SQLite FTS5`) + `0B` Defeasible Rule & Verbatim Quote Verifier (`1.000` on `Probe 7A` `[Measured-in-Memory]`).

### 2.3 Growth From Cycle 2 (How Our State Improved)
- Discovered that **even inside Vision and Hearing, `0B` deterministic libraries (`Playwright a11y`, `PyMuPDF`, `STFT/F0 prosody`) + compact MIT/Apache-2.0 transducers (`PaddleOCR-VL`, `indic-conformer-600m-multilingual`)** outperform giant end-to-end multimodal models while keeping the Central Brain free of Omni regressions.

---

## Cycle 3 — Adversarial Stress-Test With `Claude Opus 5.5` & Official HF Card Verification

### 3.1 Context & Execution
Submitted the Cycle 1 & Cycle 2 Context File to `Claude Opus 5.5` for adversarial falsification and cross-checked official Hugging Face model cards (`Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-27B`, `google/gemma-4-31B-it`, `google/gemma-4` `12B`/`E4B`, and `ai4bharat/indic-conformer-600m-multilingual`).

### 3.2 Outcome — Critical Flaws Caught Before Implementation

1. **`Gemma-4-31B` vs `Qwen3.6-35B-A3B` & `Qwen3.5-27B` on Agentic Coding & MCP Tool Use (`[Vendor-Reported HF Card]`):**
   - Although `Gemma-4-31B` (`30.7B` dense) posts high math/QA numbers (`AIME26 = 89.2%`, `MMMU-Pro = 76.9%`), Qwen's published agentic evaluations report a huge gap on Clover's core repo + `mcp.json` loop:

     | Agentic Benchmark | `google/gemma-4-31B-it` | `Qwen/Qwen3.6-35B-A3B` (`3B` MoE) | `Qwen/Qwen3.5-27B` (`27B` Dense VLM) |
     | :--- | :---: | :---: | :---: |
     | **`SWE-bench Verified`** | `52.0%` | **`73.4%`** | **`75.0%`** |
     | **`Terminal-Bench 2.0`** | `42.9%` | **`51.5%`** | — |
     | **`MCPMark` (`mcp.json` tools)** | `18.1%` | **`37.0%` (`2.04×`)** | — |
     | **`TAU3`** | **`67.5%`** | `67.2%` | — |

   - **Two major missing Brain candidates uncovered in Cycle 3:**
     1. **`Qwen/Qwen3.5-27B` (`Apache-2.0`, `27B` Dense Unified Vision+Text, `262K` ctx):** Activates **`9×` more parameters per token (`27B` vs `3B`)** than `Qwen3.6-35B-A3B`, hits **`75.0%` SWE-bench Verified**, and takes only **`~14 GiB` in `HG4`**, leaving **`10 GiB` free VRAM** on a single `24 GiB` RTX 3090 for Vision + KV cache!
     2. **`google/gemma-4` `12B Unified` / `E4B` (`Apache-2.0`, `256K`/`128K` ctx, Text + Image + Raw Audio):** Uses **lightweight linear patch & audio converters directly into the decoder** (no heavy separate encoder models) — the exact literal implementation of "thin converters + 1 Brain" for edge/single-GPU native audio+vision testing!

2. **Four Sensory Edge-Case Traps Caught & Fixed in Design:**
   - **Trap A — Broken PDF Text Layers:** `PyMuPDF` returns garbled `ToUnicode` mojibake (`\ufffd`, control chars) on corrupt PDFs. **Fix:** Add a `0B` printable-character & token-entropy gate; route failed pages to `PaddleOCR-VL`.
   - **Trap B — Canvas / CAD / Remote GUIs (`ScreenSpot-Pro`):** Have **no** DOM/accessibility tree (`Playwright` cannot see inside `<canvas>` or OpenGL viewports). **Fix:** Route `has_a11y_tree -> 0B DOM/UIA`, else `Brain Native Vision Converter + OCR coordinate merge`.
   - **Trap C — Tonal & Particle-Question Languages + Non-Speech Sounds:** Rising $F_0$ pitch does **not** mean "question" in tonal languages (Mandarin/Thai/Vietnamese) or particle-question languages like **Tamil (`-ஆ`, e.g. `வந்தாயா?`)** and Japanese (`か`), and ASR transcripts drop coughs/sirens/machine knocks. **Fix:** Pass `0B` pause/energy contours as neutral acoustic telemetry (never hardcoded to "question"), and attach a tiny audio-event classifier (`YAMNet`/`BEATs`) or native audio converter (`Gemma-4-12B/E4B`) when non-speech sound matters.
   - **Trap D — Honest 1× 24 GiB GPU Context & End-to-End Task Time:** `18–20 GiB` weights (`35B-A3B` / `31B`) leave only `~4 GiB` for Vision + KV cache (`~32K–64K` practical single-GPU context, not `256K`), whereas `Qwen3.5-27B` (`~14 GiB` HG4) or `Gemma-4-12B` (`~7 GiB` HG4) leaves `10–17 GiB` for long KV cache. Also score models by **end-to-end task completion wall-time**, not raw `tok/s`.

### 3.3 Final Converged Architecture & Pre-Implementation Gate (Ready State)

1. **Architecture ("5 Deterministic Senses + Aligned Converters + 1 Brain + `mcp.json` Tools"):**
   - **Sense 1 (See):** `0B` `PyMuPDF` (with `ToUnicode` entropy check) + `Playwright`/OS a11y → `PaddleOCR-VL` (dense/corrupt/Indic scans) → **Brain's Co-Trained Vision Converter** (charts, UI canvas, photos).
   - **Sense 2 (Hear & Speak):** `0B` `Silero VAD` + `0B` neutral `pause/energy` DSP + `ai4bharat/indic-conformer-600m-multilingual` (`0.6B`, MIT, 22 Indic langs) / `Whisper-large-v3-turbo` + `Kokoro-82M` (Apache-2.0) / `IndicParler-TTS` *(plus `Gemma-4-12B/E4B` native linear audio converter tested on tone/non-speech gate)*.
   - **Sense 3 (Read & Write):** `0B` `UTF-8` + `HFST`/`Apertium` FSTs + `Stanza` (`Apache-2.0`) disambiguation → Brain's `248K+` multilingual embedding table.
   - **Sense 4 (Act):** `0B` `TokenEconomizer` (session-start tool pruning so the Brain is never flooded) + `mcp.json` (`x-cruncher`, `0B` `StatsForecast`, `0B` `Lean 4` tactic cascade, compilers, `OpenGL/Playwright`, `ripgrep/SQLite FTS5`).
   - **Sense 5 (Judge):** `0B` Defeasible Rule Resolver (`rejected` filter + `unless` specificity = `1.000` on `Probe 7A`) + `0B` Verbatim Quote Gate + `Clover-LoRA` (`PROCEED / ASK / ESCALATE / "I don't know"`).

2. **Three 1-GPU Brain Contenders to Benchmark on the Same Pre-Registered 6-Part Gate Before Selecting Winner:**
   1. **`Qwen/Qwen3.6-35B-A3B`** (`3B` active MoE, Apache-2.0 — mount its official co-trained Vision Converter onto our incumbent `Clover-LoRA` stack).
   2. **`Qwen/Qwen3.5-27B`** (`27B` dense VLM, Apache-2.0, `~14 GiB` HG4 — tests whether `9×` active neurons per token solves `Probes 1–3` single-pass derivations on **1 GPU**, eliminating multi-GPU `120B`/`2.8T` escalation!).
   3. **`google/gemma-4` (`12B Unified` / `26B-A4B`)** (Apache-2.0 — tests native linear image+audio converters vs modular sense transducers).

---

## Cycle 4 — Training Our Own Clover 5-Sense Model (`Freeze-Omni` Paradigm + Native `model.visual` + Dynamic `Clover-LoRA`)

### 4.1 Human Direction & Context
> *"no if we have the five sense we can itself train the clover model than expecting a model with those capabilites already available"*

Inspected our existing distributed trainer ([train_clover_harness_lora.py](../../clover-ai-system/src/training/train_clover_harness_lora.py)), `Freeze-Omni` (`[Primary-Paper]` ICML 2025, `arXiv:2411.00774v2`), and ran Cycle 4 adversarial audit with `Claude Opus 5.5` on compute math, failure modes, and what to train vs. reuse.

### 4.2 Outcome — Four Key Engineering Truths Established

1. **Smoking Gun Found in Our Repo (`train_clover_harness_lora.py:148–150`):**
   - `Qwen/Qwen3.6-35B-A3B` **already** contains its trillions-of-tokens co-trained Vision Tower (`model.visual`), and our own script explicitly deleted it (`full_model.model.visual = None`)!
   - **Rule for Sense 1 (Vision):** **Do NOT retrain Vision from scratch on 1M samples** (which takes days under sequential MoE sharding and loses to Qwen's trillion-token `model.visual` on `ScreenSpot` / `ChartQA`). **Re-enable native `model.visual`** (`0` training hours) + route digital/corrupt PDFs through `0B` `PyMuPDF` / `PaddleOCR-VL`.

2. **Why `Freeze-Omni` (`arXiv:2411.00774v2`) + Our Multi-Server `RTX 3090` Cluster Is Ideal for Training the Missing Senses:**
   - Because the `35B` Brain trunk is **100% frozen**, **only the `20M–80M` Converter parameters (`~40–160 MB` gradients) synchronize across nodes**!
   - Even over modest WAN bandwidth, multiple `4× RTX 3090` servers (`17.3 GiB/GPU` BF16 shard, `26 GiB` free activation headroom in `train_clover_harness_lora.py`) can run **data-parallel converter replicas** (`~8–16 hours` for `~2,000` hours of `12.5 Hz` speech = `90M` audio tokens; `~2–4 hours` for `Clover-LoRA`).
   - *Case I Warning:* Training raw-pixel or raw-waveform encoders from scratch *without* frozen pretrained feature front-ends (`AuT` needed `20M` hours) is infeasible; always project from frozen `indic-conformer-600m` + `Whisper-v3-turbo` features!

3. **Exact Condition for True `0.00%` Degradation:**
   - Keeping the Brain frozen guarantees **bit-identical `0.00%` base logits** *only* when no sensory tokens are present **and `Clover-LoRA` is dynamically toggled (`scale = 0.0` for raw base/vision tasks, `scale = 1.0` for `AGENTS.md` judgment, tool loops & relational extraction)**.

4. **What We Train Ourselves to Create `Clover-1` (Capabilities No Pre-Baked Checkpoint Has):**
   - **Module A — Clover Dual-Ear Hearing Converter (`~35M–60M` Conv-4× + MLP into Frozen Brain `d=2048`):**
     - Front-end: `0B` `128`-Mel DSP → frozen `ai4bharat/indic-conformer-600m-multilingual` (MIT, 22 Indian languages) + `Whisper-large-v3-turbo` (global).
     - Loss: **Self-Distillation KL** (`KL(Brain(Converter(audio)) || Brain(transcript))` — requires **zero** scarce Tamil speech-QA labels!) + `Voxtral` `<next>` continuation + `Freeze-Omni` Stage-3 prompt embedding (`60K` multi-round dialogues) + paralinguistic tone/event clips.
     - *Simplicity Gate:* Must **match or beat** the `0`-training cascade (`IndicConformer/Whisper -> text + 0B pause/energy -> Brain`) on WER + tone/hesitation tasks, or we ship the cascade!
   - **Module B — Clover Streaming Voice Converter (`Freeze-Omni` `NAR Prefix + AR Codec`):**
     - Maps frozen Brain hidden states + streamed text chunks → speech codec frames; gated directly against `Kokoro-82M` / `IndicParler-TTS` tool latency & MOS/WER.
   - **Module C — Dynamic `Clover-LoRA` (`rank=32`, Trained via Assistant-Masked SFT + `GRPO` Verifier RL in `train_clover_harness_lora.py`):**
     - Trains the 3 faculties missing from raw models: (i) 4-way `PROCEED / ASK / ESCALATE / REFUSE / "I don't know"` discernment (`11/19 -> 19/19` target), (ii) multi-tool `mcp.json` selection without tool-flooding regression, and (iii) `{subj, rel, obj, cond, polarity, status, span}` extraction into Clover's `0B` Defeasible Rule Resolver (`1.000` on `Probe 7A`).

### 4.3 Mandatory 6-Point Fail-Closed Training Gate (Prevents `manual_seed(42)` Fake-Converter Bugs Forever)
1. **Logged Convergence Curve:** Held-out validation loss must drop monotonically well below random initialization.
2. **Shuffled-Pair Negative Control:** Pairing `audio_i` / `image_i` with mismatched `text_j` must sharply spike loss (proving the Brain genuinely attends to sensory tokens).
3. **Must Beat `0`-Training Cascade:** Trained Audio/Voice Converters ship **only** if they beat `IndicConformer/Whisper + Kokoro` tool cascade on latency or paralinguistic accuracy.
4. **Bit-Identical Frozen Trunk Check:** With `LoRA` scale `0.0` and text-only input, output logit SHA-256 must match base checkpoint `100%`.
5. **Agentic Non-Regression:** With `Clover-LoRA` active, `HumanEval` (5/20/50 tools), `Probes 1–3` (`max_tokens=4096`), and `Probe 6` (19+40 discernment cases) must tie or beat base within Wilson CI.
6. **Fail-Closed Loader:** Remove all `torch.manual_seed(42)` random converter fallbacks in `lfm3_multimodal.py` & `nemotron_multimodal.py`; missing or ungated converter weights raise an immediate error.

---

## Cycle 5 — Training a Clover Brain From Scratch vs. `Minitron` Prune+Distill vs. `Freeze-Omni` Adapter

### 5.1 Human Direction & Context
> *"we can even train a model from scratch if needed"*

Audited exact `6 × N_active × D` FLOPs, Adam optimizer VRAM (`16–18 B/param`), WAN gradient synchronization (`DiLoCo`), primary-source `Phi-1` (`[Primary-Paper]` `arXiv:2306.11644`: `1.3B` params, `7B` unique tokens × `~7` epochs ≈ `51B` tokens seen, `768` A100-hours = `50.6%` HumanEval) and `NVIDIA Minitron` (`[Primary-Paper]` `arXiv:2407.14679`: structural pruning + logit distillation in `24–200` A100-hours), stress-tested with `Claude Opus 5.5`.

### 5.2 Outcome — Exact RTX 3090 Compute, VRAM & Reality Trade-Offs

1. **Exact From-Scratch Compute & Memory Table on Our `RTX 3090` Hardware (`~28` achieved TFLOPS/GPU @ ~40% MFU; `DiLoCo` outer-sync every 500 steps across WAN):**

   | From-Scratch Target | Tokens Seen ($D$) | Total 3090 GPU-Hours | `1 × 8×3090` Same-Box | `48 × 3090` (`DiLoCo` @ 70% eff.) | Fits Adam (`16–18 B/p`)? |
   | :--- | :---: | :---: | :---: | :---: | :--- |
   | **`Clover-Seed-0.5B` Dense** | `10B` | **`298` h** | `1.6` days | **`9 hours`** | Yes (`~9 GB`/GPU, 1 replica/GPU) |
   | **`Clover-Seed-1.0B` Dense** | `20B` | **`1,190` h** | `6.2` days | **`1.5 days`** | Yes (`~18 GB`/GPU, 1 replica/GPU) |
   | **`Phi-1` Scale (`1.3B` Dense)** | `51B` (`7B` × 7 ep) | **`3,968` h** | `21` days | **`4.9 days`** | Tight (`ZeRO-2` on 2 GPUs) |
   | **`3B` Dense** | `100B / 300B` | `17.9K / 53.6K` h | `93 / 279` days | `22 / 66` days | Sharded inside `8×3090` PCIe box |
   | **`35B-A3B` MoE From Scratch** | `2T+` (`Qwen` uses `36T`) | `357,143+` h | `5.1+` years | **`443+` days** | **No** (`~560 GB` Adam > 192 GB box) |

2. **Three Hidden Real-World Realities Caught in Cycle 5:**
   - **Pragmatic Commonsense ≠ Trivia:** Knowing that *"kill zombie process 4412"* is benign while *"history starts from today"* destroys git history (`Probe 6` discernment, where `LFM3-2B` got `12/19` vs `35B`'s `19/19`) comes from broad world/language exposure, not Python textbooks alone (`Phi-1` hit `50.6%` HumanEval on `7B` tokens, yet `Phi-1.5` & `Phi-2` needed `150B` and `1.4T` tokens to gain general language pragmatics).
   - **Where From-Scratch *Does* Have a Unique Scientific Edge:** Baking **calibrated abstention (`<i_dont_know>`), `<ask_human>`, `<tool_call>`, and native `HG4` Quantization-Aware Training (QAT)** into the objective from **Step 0** at the `0.5B–1.0B` edge scale (`9–36 hours` on `48 × 3090`!).
   - **`Minitron` (`arXiv:2407.14679`) Beats Random-Init at `1B–4B`:** Structurally pruning + logit-distilling `Qwen3.6-35B` / `Qwen3.5-27B` down to a compact custom Clover student retains pragmatic world knowledge in `1B–7B` tokens (`10–50×` fewer tokens than random init!).

### 5.3 Growth — The Two-Track Clover Plan (Best of Both Worlds, Zero Wasted GPU Hours)

1. **Track 1 — Production `Clover-35B` 5-Sense Brain (Cycle 4 `Freeze-Omni` Recipe, Ships in Hours/Days):**
   - **Keep `Qwen3.6-35B-A3B` frozen** + **re-enable its native `model.visual`** (fixing `train_clover_harness_lora.py:148` where we accidentally set `model.visual = None`!) + train **Dual-Ear Hearing Converter** (`indic-conformer-600m` + `Whisper-turbo` → `2048-d`), **Voice Converter**, and **Dynamic `Clover-LoRA`** (`scale=1.0` on agent loops, `scale=0.0` on raw base tasks), gated against the `0`-training tool cascade.
2. **Track 2 — Ground-Up `Clover-Seed-0.5B / 1B` From-Scratch & `Minitron` Edge Brain (`9–36 hours` via `DiLoCo`):**
   - Train a **`0.5B` Clover-Seed from random init** (`10B` open textbook + multilingual + tool + abstention tokens + top-32 teacher logits stored on `k3` NVMe, native `HG4` QAT from Step 0) alongside a **`Minitron` pruned+distilled arm**.
   - **Strict Scaling Gate (`0.5B → 1.5B → 3B`):** Only scale up if `Clover-Seed-0.5B` beats a pretrained `0.6B + LoRA` control outside the Wilson CI on `>= 2` of: (i) 4-way discernment, (ii) zero tool-flooding drop on HumanEval, (iii) `Probe 7` relational edge recall, and (iv) abstention calibration!

---

## Cycle 6 — Modular Sense Cortices (Library + Neural Cortex per Sense) & Dual-Channel `Tool-NCM`

### 6.1 Human Direction & Context
> *"we can have each model trained for different purpose one model to understand visuals using the libaries which means it become neurons that understand the eye and each 5 sense can have it's own version of models and even the mcp and tools we can try using ncm approach"*

Audited our existing `NCM` codebase ([next_concept.py](../../clover-ai-system/src/tools/hybrid-ncm-k3/models/next_concept.py), `vqvae.py`, `eval_both_tracks.py`, `benchmark-next-concept-modeling.py`, [clover_35b_hybrid_ncm.py](../../clover-ai-system/src/models/clover-35b-a3b/clover_35b_hybrid_ncm.py)) against `Meta LCM` (`[Primary-Paper]` `arXiv:2412.08821`), `SONAR-LLM` (`[Primary-Paper]` `arXiv:2508.05305`), and `ToolkenGPT`/`ToolGen`, stress-tested with `Claude Opus 5.5`.

### 6.2 Outcome — Honest Repo Audit & Three Laws That Make Cortical `NCM` Work

1. **What Our Existing `NCM` (`hybrid-ncm-k3`) Actually Is Today (`[Repo-Verified]`):**
   - Built on **Hazy Research's `TinyStories` VQ-VAE** (`598` story validation pairs + `48` story requests; `ncm_stage2.py` notes level-0 codebook norms `~4e10`), while `benchmark-next-concept-modeling.py` tested `25` WMT14 / `10` XSum / `10` Sudoku examples routed through the Kimi chat API, and `clover_35b_hybrid_ncm.py`'s `2048 → 384` bridges have no logged convergence gate on code, tools, vision, or audio.
   - **Lesson:** Do **not** build 5-sense / MCP cortices on the `TinyStories` VQ-VAE! Anchor the shared concept bus to a **frozen multilingual/multimodal embedding or Brain `d_model` codebook**.

2. **Three Engineering Laws Required for "1 Cortex Model Per Sense (On Top of Libraries) + `Tool-NCM` + 1 Brain":**
   - **Law 1 — Frozen Shared Cortical Bus (Independent Versioning Without Drift):**
     - Each Sense Cortex (`Eye Cortex` over `OpenCV/PyMuPDF/zbar` + frozen visual features; `Ear/Voice Cortex` over `STFT/128-Mel/VAD` + frozen acoustic features; `Language Cortex` over `UTF-8/HFST`; `Tool-NCM Cortex` over `mcp.json`) **must** map into a **frozen shared coordinate space**. Otherwise upgrading Eye Cortex `v2` shifts coordinates and breaks the Brain and other cortices!
   - **Law 2 — Dual-Channel Bus (`Concept Vector z_c` + `Exact Symbol Tape`):**
     - By the Data Processing Inequality, a `384-d` or `11-level` VQ concept vector **cannot** losslessly carry `987654321987654321`, `/mnt/nvme1/...`, or exact variable names (the exact reason pure `Meta LCM` lagged token LLMs on exact details until `SONAR-LLM` restored token-level cross-entropy!).
     - Every cortex must emit **two channels**:
       1. **`Concept Channel` (`z_c` / hierarchical VQ codes):** Semantic meaning, scene layout, acoustic tone, intent, and tool ID (`0` prompt-schema bloat!).
       2. **`Exact Symbol Tape` (`⟦lit:0⟧ = "987654321987654321"`, `⟦span:file:line⟧`, `⟦box:(x,y,w,h)⟧`):** Extracted deterministically by the sense's `0B` libraries so tool arguments and code identifiers are **copied bit-exact from the tape**, never decoded from a lossy vector, then validated against `mcp.json` JSON Schema (`0B`).
   - **Law 3 — `Tool-NCM` Must Beat `0`-Training Top-K Schema Retrieval (`bge-small` / `BM25`):**
     - Stuffing `150+` `mcp.json` schemas (`~22K` tokens) floods small brains (`LFM3-2B` dropped `39.0% → 18.9%` with just 5 tools). Both **`0`-training Top-5 `bge-small`/`BM25` schema retrieval** and **`Dual-Channel Tool-NCM` (`ToolkenGPT`-style learned tool concepts + Exact Tape argument copy)** eliminate prompt flooding — so `Tool-NCM` must prove its worth head-to-head against Top-5 retrieval!

### 6.3 Concrete 2-Hour Cycle-6 Validation Experiment (Before Full Multi-Cortex Build)
- **Benchmark:** `150` real-style `mcp.json` tools (including `30` near-duplicate pairs + `25` unseen tools hot-added after training), `600` queries containing 18-digit integers and exact file paths (`450` train / `150` test).
- **4 Paired Arms on 1 GPU:**
  - **Arm A:** All `150` JSON schemas in prompt (`~22K` tokens) on `2B` & `35B`.
  - **Arm B (`0`-Training Baseline):** Top-5 schemas retrieved via `bge-small` (`0` training, zero-cost hot-add of the `25` new tools).
  - **Arm C1 (`Pure Concept NCM`, negative control):** Tool selected & arguments decoded purely from `384-d` concept vector (verifies literal corruption without tape).
  - **Arm C2 (`Dual-Channel Tool-NCM`):** Learned `NCM` tool concept selection (`0` schema tokens in Brain, `~20 min` training) + **Exact Symbol Tape copy** + `0B` JSON Schema check.
- **Decision Gate:** `Dual-Channel Tool-NCM` (`C2`) earns adoption if and only if it beats `Arm B` on tool-selection accuracy/latency outside Wilson CI while achieving `100%` bit-exact literal preservation.

---

## Cycle 7 — Audit of Initial Clover Research (`JEPA`, `Curiosity`, `Emergent Concepts`) & Gradual 7-Horizon Master Plan (`Cycles 1 → 1,000,000`)

### 7.1 Human Direction & Context
> *"and jeppa since even though eyes look at so much details it only focus on few things to understand it jeppa will be useful and so many things were part of clover inital research improve those research and see how we should plan this system a detailed plan step by step rather than thinking everything will be done in one attempt it might need 1000s of clover cycle even millions but we should plan it gradually"*

Audited [jepa_engine.py](../../clover-ai-system/studio/clover/jepa_engine.py), [curiosity_foundation.py](../../clover-ai-system/studio/clover/curiosity_foundation.py), [emergent_concepts.py](../../clover-ai-system/studio/clover/emergent_concepts.py), `personal.py`, and [experiment-design.md](../experiments/meaning-development/experiment-design.md) alongside `Claude Opus 5.5`.

### 7.2 Honest Audit of Clover's Initial Research Modules (Keep the Architecture, Replace the Prototype Stubs)

| Initial Clover Module | Enduring Design Genius to **KEEP** | Prototype Stub / Flaw to **UPGRADE** |
| :--- | :--- | :--- |
| **`jepa_engine.py` (`ijepa`, `audio_jepa`, `eb_jepa`, `mjepa`, `clover_rpre`)** | Unified routing across visual foveation, audio regime detection, latent MPC planning, and `OutcomeResidualAdapter` with Anomaly + Non-Regression Rollback Gates. | Falls back to `_build_default_weights` (`64-d` random orthogonal init + regex/hash features) when trained trunks are absent — random predictors emit noise, not surprise! Replace with real trained `V-JEPA 2.1` / `Audio-JEPA` weights. |
| **Foveated Attention Principle** | Discarding `90–98%` of irrelevant pixel/acoustic detail before hitting the Brain. | **Surprise alone misses static task-critical items** (e.g., a motionless error dialog!). True foveation must take the **union of**: (1) High `JEPA` prediction error (bottom-up salience), (2) Top-down Brain query relevance, and (3) **Exact Symbol Tape** (OCR/literals never dropped), gated against `0B` `EVS` frame-differencing. |
| **`emergent_concepts.py` (`Observe → Associate → Hypothesize → Validate → Grow`)** | **Crown jewel lifecycle:** holds unverified hypotheses `under_review` in `staging/`, induces Minsky-Papert Order-3 boxes + inhibitory vetoes, and promotes **0%** of unvalidated claims. | Replace hand-written `CONDITION_FAMILIES` / `SYMPTOM_FAMILIES` word lists with learned relational extraction (`GLiREL` / `NuExtract-2B` / `Clover-LoRA`). |
| **`curiosity_foundation.py` (`LPM/CPM` + Controllability Gate + `ZPDES`)** | Defeats the "noisy TV / rustling leaves" spectator trap via Controllability Gate, POMDP Value-of-Information (`VoI`), and `ZPDES` proximal-difficulty selection. | `encode_text_to_64d` uses **SHA-256 word hashing** rather than semantic embeddings; upgrade to real `JEPA` / Brain embeddings. |
| **`memory-knowledge-graph.py` & `symbolic-ontology-graph.py`** | `staging/` quarantine gate, provenance chains, Horn deductive closure, Rosenblatt $\gamma$-balance, Minsky-Papert Order-$k$ boxes. | Add `rejected` status, `unless` specificity rule resolution (`T2 = 1.000`), and `context_scope` regime filtering. |
| **`personal.py` (Personal Habit Learning)** | Safety asymmetry: learned user habits *"can only make Clover more careful"*, never bypass a safety ask or offer a declined tool. | Validate beyond simulated personas on real longitudinal human sessions. |

### 7.3 Why `1,000,000` Verifier-Gated Clover Cycles Compound Instead of Collapsing
- **Unverified loops suffer Model Collapse** (Shumailov et al., *Nature* 2024) and memory pollution because hallucinations re-enter memory and training.
- **Clover cycles only promote externally verified outcomes** (`staging/ -> oracle check -> active`), making progress ratcheted and monotonic.
- **Defeating Goodhart's Law across `1,000–1,000,000` cycles:** Any single fixed benchmark eventually gets overfit. Horizon 0 therefore establishes a **Sealed Canary Vault** (never visible to selection) + **Rotating Fresh Canaries** + **Automatic Rollback** whenever any canary drops outside its confidence interval, plus **Human Direction Checkpoints** at every Horizon boundary.

### 7.4 The 7-Horizon Step-by-Step Master Plan (`Cycles 1 → 1,000,000+`)

| Horizon & Cycle Scale | Step-by-Step Deliverable | Initial Clover Research Upgraded | Atomic Unit of `1 Cycle` | Hard Exit Gate (Must Pass + Human Sign-Off) |
| :--- | :--- | :--- | :--- | :--- |
| **Horizon 0** (`Cycles 1–10`)<br>*Safe Ratchet & Truth Ledger* | Append-only `CloverMetricsLedger`, Sealed + Rotating Canaries, Auto-Rollback, Fail-Closed Loader (delete all `manual_seed(42)`, `_build_default_weights`, and SHA-hash stubs). | `clover_cycle.py`, `memory_tree/staging/`, `clover_telemetry_ledger.py` | 1 infra guard + 1 fault-injection test | Injected bad adapter/rule triggers automatic rollback `100%` of the time; ledger reproduces baseline probe scores within CI. |
| **Horizon 1** (`Cycles 10–50`)<br>*Computed `0B` Core & Memory Fix* | Wire `0B` MCP tools (`x-cruncher`, `0B` Periodogram+Conformal forecaster, `0B` Lean tactic cascade, `0B` Defeasible Rule Resolver w/ `rejected` + `unless`, `0B` Verbatim Quote Gate, `PyMuPDF` entropy check). | `studio_ontology_verifier.py` (`S4`), `studio_epistemic_memory.py` (`S5`), `xcruncher_bridge.py` | 1 `0B` tool + its deterministic oracle | `Probe 7A` resolver = `1.000` (`180/180`); `0B` TS `MASE <= 0.32`; `0B` Lean `>= 6/12` MiniF2F (`0` GPUs); `0` random fallbacks left in repo. |
| **Horizon 2** (`Cycles 50–150`)<br>*Eye Cortex & Foveated `JEPA`* | Re-mount `Qwen3.6-35B` native `model.visual` (fixing `train_clover_harness_lora.py:148`); build **3-Signal Foveated `JEPA` Filter** (`Real V-JEPA 2.1 Surprise ∪ Goal Query ∪ PaddleOCR-VL Exact Symbol Tape`). | `jepa_engine.py` (`ijepa`), `nemotron_multimodal.py` (`EVS`), `image_inspector.py` | 1 visual benchmark slice (`Probe 5`, `Cases 5–6`, GUI/docs) | `Probe 5 = True`; `Cases 5–6` pass; Foveated `JEPA` cuts visual tokens **`>= 10×`** with **`<= 1` pt** accuracy loss and beats `0B` `EVS` frame-diff. |
| **Horizon 3** (`Cycles 150–300`)<br>*Ear & Voice Cortex* | Ship `0`-training Ear/Voice cascade (`Silero VAD` + `0B` neutral pause/energy + `indic-conformer-600m` MIT 22 langs + `Whisper-turbo` + `Kokoro-82M`); train `Freeze-Omni` / `Audio-JEPA` converter **only** where cascade falls short. | `jepa_engine.py` (`audio_jepa`), `lfm3_multimodal.py` | 1 language/prosody eval batch (Tamil, Hindi, English + tone/events) | Passes WER gate per language + **Shuffled-Pair Negative Control**; trained converter ships only if it beats the `0`-training cascade. |
| **Horizon 4** (`Cycles 300–600`)<br>*Multilingual Morphology & Relational Extraction* | Wire `HFST` + `Stanza` (Apache-2.0) into `CSI`; train/wire `GLiREL` / `NuExtract-2B` / dynamic `Clover-LoRA` to extract `{subj, rel, obj, cond, polarity, status, span}` into `S4/S5` & `emergent_concepts.py`. | `csi/morphology.py`, `csi/analyze.py`, `semantics.py`, `emergent_concepts.py` | 1 language or spec-compaction batch (`Probe 7A/7B`) | Passes 20-sentence Tamil/Turkish in-word negation gate + `Probe 7A/7B` end-to-end with `0` hallucinated numbers (`0B` Quote Gate). |
| **Horizon 5** (`Cycles 600–1,000`)<br>*Dual-Channel `Tool-NCM` Bus* | Replace `TinyStories` VQ-VAE with **Frozen Shared Concept Anchor Space + Exact Symbol Tape**; run 4-arm `150`-tool + `25`-new-tool `Tool-NCM` gate. | `next_concept.py`, `vqvae.py`, `token_economizer.py` | 1 tool-domain expansion (`50` tools/batch) | Beats `bge-small` Top-5 schema retrieval outside Wilson CI with `100.0%` bit-exact literal preservation and `0` tool-flooding drop. |
| **Horizon 6** (`Cycles 1,000 → 1,000,000+`)<br>*Endogenous Curiosity, Collective Growth & `Clover-Seed`* | Upgrade `curiosity_foundation.py` to real `JEPA`/Brain embeddings; run bounded sandbox curiosity cycles (`Observe → Hypothesize in staging/ → Oracle Validate → Grow`); distill verified trajectories into `HG4` `Clover-Seed` (`0.5B → 1.5B → 3B` via `DiLoCo`/`Minitron`). | `curiosity_foundation.py`, `emergent_concepts.py`, `personal.py`, `collective.py` | 1 self-proposed sandbox experiment + oracle check | Rising rotating-canary curve across $10^3$–$10^6$ cycles with **`0` regressions** on the Sealed Canary Vault; `Clover-Seed` beats same-size pretrained+LoRA on Cycle 5 gate. |






