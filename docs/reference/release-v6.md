# Clover v6.0 Release Notes (`2026-11-26`)

**Clover v6.0 — Minimal 4-Part Steerable AI Harness (`AGENTS.md` + `mcp.json` + `<task_id>.md` Context + Prerequisite-Gated Todo), Hadamard-Grouped (`HG4-16R-OP` / `K31H`) Specialist Fleet, `tsock://` `4 B/tok` Binary Mesh, `CloverMetricsLedger` Runtime Telemetry, 6 Broken AI Myths Public Proofs, and Autonomous Zero-Setup Reproducibility (`REPRODUCE_AGENTS.md`).**

---

## 1. Why v6.0 Is a Major Architectural Milestone

Clover v6.0 distills everything learned across v4 and v5 into two core principles:

1. **Computed vs. Predicted:** Never train or prompt neural weights to predict what a deterministic algorithm can compute (`x-cruncher` exact NTT/BBP math, `Lean 4` kernel & tactic cascades, `0B` Periodogram+Conformal forecasting, `0B` Defeasible Rule Resolution, compilers, `ripgrep`, and `SQLite`).
2. **Human Direction at the Steering Wheel (`Myth 6`):** AI models are prediction engines — fast autopilot on bounded sub-goals, while a named human sets the destination, holds accountability, and steers past off-map roadblocks via Clover's minimal 4-part harness:
   - [`AGENTS.md`](../../AGENTS.md) (`Context → Direction → Execution → Outcome → Growth` + `PROCEED / ASK / ESCALATE / REFUSE`)
   - `mcp.json` (`Mcp2MeshServer` + `clover_mcp_stdio.py` tool & specialist bus)
   - `<task_id>.md` (`CloverContextRecord` structured per-task memory)
   - `CloverTodoStore` (SQLite prerequisite-gated todo graph)

---

## 2. Key Engineering Additions in v6.0

- **Self-Hosted Clover v6.0 System & Interactive Hardware Gate (`clover-ai-system/studio/clover/clover_v6_system.py`, `src/tools/hg4-fusion-gateway/`):**
  - Unifies the 4-Part Steerable Harness, `0B` Computed Specialists, and `CloverMetricsLedger` for self-hosted Python/C deployments *(note: standalone Windows `.exe` and Android `.apk` GUI installers in `clover-ai-system/studio/` & `mobile/` remain **Work in Progress** and are withheld until end-to-end UX verification completes)*.
  - Added **Clover System Setup (`GET /api/clover/setup/catalog`, `POST /api/clover/setup/plan`, `POST /api/clover/setup/apply`)** so operators can deploy **1 model**, a **custom set**, or the **complete 9-component Clover AI System** across their machine and any GPU servers they provide: small models (`0B` Computed Pack, `LFM3-Omni-2B`, `Chronos-Bolt 0.2B`, `V-JEPA 2.1`) run interactively on any CPU machine, while big GPU models (`Boltz-2` `>=16 GB`, `Goedel-Prover-V2-32B` `>=24 GB`, `Clover-35B-A3B` `>=24 GB`, `Nemotron 3.5 Super VL 120B` `>=72 GB`, `Kimi K3.1 Pro 2.8T K31H` `741 GB` / `>=780 GB` same-rack) verify real single-user (`batch = 1`) decode `tok/s` and are automatically skipped on CPU-only or undersized hardware with clear minimum server specs.
- **Hadamard-Grouped (`HG4-16R-OP` & `K31H`) Open Specialist Conversions:**
  - `Qwen3.6-35B-A3B` (`68.2 GiB` BF16 → `17.2 GiB` HG4 / `17.4 GiB` w/ LoRA fold, fits `1 × 24 GB` GPU, `86/86` deterministic token match), `Nemotron-3.5-Super-VL 120B` (`224.5 GiB` / `241.0 GB` BF16 → `58.2 GiB`), `Goedel-Prover-V2-32B-HG4` (`15.8 GiB`), `Boltz-2-HG4-16R` (`4,007 MiB` → `529.55 MiB`, `7.57x` smaller, `0.995939` cosine, `+20.92 dB` SQNR; `stage45` 3D `.cif` folding `pLDDT = 0.926` in `40.6 s` & `0.908` in `39.6 s`), `V-JEPA 2.1-HG4` (`0.51 GiB`, `1408-d` rollout), `Chronos-Bolt-HG4` (`102 MiB`), and `LFM3-Omni-2B-HG4` (`0.83 GiB`, `87/87` exact tokens).
  - Standalone converter: [`site/hg-format/convert_to_hg.py`](../../site/hg-format/convert_to_hg.py).
- **Comprehensive Telemetry Ledger (`CloverMetricsLedger`):**
  - Thread-safe SQLite + in-memory telemetry (`GET /v1/clover/metrics`, `POST /v1/clover/metrics/reset`) tracking `used_vs_unused` models/tools/instances, `decode_tok_per_s` (`mean/p50/p95/max`), `solver_cycles` (`cycles_histogram`, `solved_after_retry_count`), and automatic `fine_tune_signals` (`clover-ai-system/src/tools/hg4-fusion-gateway/clover_telemetry_ledger.py`).
  - Live telemetry immediately caught and fixed two real regressions: `LFM3-2B` planning-tool flooding on single-function code prompts (`39.02% -> 18.90%`, fixed via Tier-0 Tool-Exposure Gate in `clover_4part_agent.py`) and missing `A^B` / `A**B` binary exponentiation in `studio_xcruncher_verifier.py` (`43/43` `-W error` tests passing in `test_hg4_fusion_gateway.py`).
- **6-Myth AI Mythbuster & Turnkey Public Reproducibility (`site/hg-format/`):**
  - Added **Myth 6 (`#myth-6-driver` — The Center Steering Wheel)** with golden scroll-synced center hub SVG and live 6-task proof (`0/6 = 0.0%` unsteered AI-alone vs. `5/6 = 83.33%` Human+AI in `1.17` cycles & `53%` fewer tokens).
  - Added autonomous zero-human-setup agent instructions ([`site/hg-format/REPRODUCE_AGENTS.md`](../../site/hg-format/REPRODUCE_AGENTS.md)), standalone `<5s` hardware/math/socket verifier ([`site/hg-format/reproduce_hg_myths.py`](../../site/hg-format/reproduce_hg_myths.py)), and raw cluster JSON ledgers ([`benchmarks/public-runs-20261126/README.md`](../../benchmarks/public-runs-20261126/README.md)).
- **Recorded Next Major Milestone (`v6.x -> v7.0` Roadmap):**
  - Preserved the 7-Cycle research audit & 7-Horizon step-by-step plan (`Cycles 1 -> 1,000,000+`) uniting 5 deterministic senses, 3-signal foveated `JEPA`, `Freeze-Omni` aligned converters, Dual-Channel `Tool-NCM`, and `Clover-Seed` in [`docs/runtime-context/one-brain-five-senses-research-context.md`](../runtime-context/one-brain-five-senses-research-context.md).
