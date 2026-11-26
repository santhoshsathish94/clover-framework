# Clover Framework & Clover AI Studio — Master `TODO.md` (`v6.0.0`)

Governed by the canonical Clover Cycle (**`Context → Direction → Execution → Outcome → Growth`** in [`AGENTS.md`](AGENTS.md)).

---

## Shipped in `v6.0.0` (`2026-11-26`)

- [x] **1. Minimal 4-Part Steerable Clover System (`AGENTS.md` + `mcp.json` + `<task_id>.md` + Prerequisite-Gated SQLite Todo)**
  - Implemented across both the `HG4` Fusion Gateway ([`clover_4part_agent.py`](clover-ai-system/src/tools/hg4-fusion-gateway/clover_4part_agent.py)) and Clover AI Studio ([`clover_v6_system.py`](clover-ai-system/studio/clover/clover_v6_system.py)).
  - Enforces `STOP_AND_ASK` human steering on destructive/underspecified requests and Tier-0 `2B` tool-exposure gating (`43/43` gateway tests & Studio v6 tests passing under `python -W error`).

- [x] **2. Computed `0B` Specialists First (`Computed > Predicted`)**
  - Exact big-integer multiplication, factorials, BBP hexadecimal $\pi$, and binary exponentiation (`A^B` / `A**B` past `6,000+` digits) via `x-cruncher` ([`local_tool_packs.py`](clover-ai-system/studio/clover/local_tool_packs.py), [`studio_xcruncher_verifier.py`](clover-ai-system/src/tools/hg4-fusion-gateway/studio_xcruncher_verifier.py)).
  - `0B` Periodogram + Joint Harmonic OLS + Split-Conformal `P10/P50/P90` forecaster (`MASE = 0.3084`, `<50 ms` CPU) and `0B` Defeasible Epistemic Rule Resolver (`180/180` accuracy).

- [x] **3. Clover AI Studio `v6.0.0` Rebuild Across Desktop, Android Mobile, Web & VS Code**
  - Added **Clover System Setup** (`1` model, custom subset, or full Clover AI System) across local device and user-supplied GPU servers with per-GPU VRAM placement (`ready_served`, `ready_pinned`, `ready_swap`, `needs_gateway`, `skipped_not_interactive`) and multi-token live decode `tok/s` verification.
  - Small models & `0B` tools run interactively on any phone, laptop, or CPU machine; big GPU models are automatically skipped on CPU-only/undersized machines with explicit minimum server requirements.
  - Updated first-run [`setup.html`](clover-ai-system/studio/clover/web/setup.html) with **Choose models / Use server** so users connecting remote servers are never forced into a `3 GB` local download first.
  - Rebuilt and signed Windows [`CloverAIStudio-Setup.exe`](download/laptop/CloverAIStudio-Setup.exe) (`6.0.0`), [`clover-local-6.0.0.vsix`](clover-ai-system/studio/studio/dist/clover-local-6.0.0.vsix), Android [`CloverAIStudio.apk`](download/mobile/CloverAIStudio.apk) & [`CloverAIStudio.aab`](download/mobile/CloverAIStudio.aab) (`versionCode 22`), live `k3` Web Studio (`https://api.cloverframework.com`), and deterministic [`RELEASE-MANIFEST.json`](release/beta/RELEASE-MANIFEST.json) + [`SHA256SUMS.txt`](download/SHA256SUMS.txt).

- [x] **4. Six Broken AI Myths, `CloverMetricsLedger` Telemetry & Zero-Setup Public Reproducibility**
  - Published live cluster proofs in [`benchmarks/public-runs-20261126/`](benchmarks/public-runs-20261126/README.md), updated [`site/hg-format/index.html`](site/hg-format/index.html) and [`site/ai-system/index.html`](site/ai-system/index.html), and shipped [`site/hg-format/REPRODUCE_AGENTS.md`](site/hg-format/REPRODUCE_AGENTS.md) + [`site/hg-format/reproduce_hg_myths.py`](site/hg-format/reproduce_hg_myths.py) (`6/6 PASS` in `<5s`).

---

## Next Major Research Milestone (`v6.x → v7.0`, Recorded & Paused)

> **Context & 7-Horizon Step-by-Step Plan (`Cycles 1 → 1,000,000+`):** [`docs/runtime-context/one-brain-five-senses-research-context.md`](docs/runtime-context/one-brain-five-senses-research-context.md)

- [ ] **Horizon 1 (`Cycles 1–50`):** Restore native vision in `Qwen3.6-35B-A3B` LoRA training (remove `model.visual = None`), enforce fail-closed checks against untrained `manual_seed(42)` adapters, and ship deterministic OCR/AST/math preprocessors.
- [ ] **Horizon 2 (`Cycles 50–500`):** Multilingual ear & voice converters (`Freeze-Omni` adapter + `ai4bharat/indic-conformer-600m-multilingual` for Tamil/Hindi/Kannada + English).
- [ ] **Horizon 3 (`Cycles 500–5,000`):** 3-Signal Foveated `JEPA` attention cropper (`PaddleOCR-VL` + `V-JEPA 2.1` epistemic surprise patches).
- [ ] **Horizon 4–7 (`Cycles 5,000 → 1,000,000+`):** Dual-Channel `Tool-NCM` (`Concept + Exact Symbol Tape`) and `Clover-Seed` (`0.5B → 3B` coordinator brain trained over the 5 deterministic senses).
