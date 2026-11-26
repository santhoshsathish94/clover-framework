# Public Benchmark Ledgers — Clover v6.0 (`2026-11-26`)

This folder contains the raw JSON telemetry and outcome ledgers captured on `2026-11-26` across our `Hetzner k3` coordinator hub and `24 GB RTX 3090` `HG4-16R` GPU petals, plus the 1-command local reproducibility runner.

## Quick Autonomous Reproduction (Zero Human Setup)

Point any AI coding agent (`GitHub Copilot`, `Claude Code`, `Cursor`, `Codex`, `OpenCode`) at [`../../site/hg-format/REPRODUCE_AGENTS.md`](../../site/hg-format/REPRODUCE_AGENTS.md), or run directly:

```bash
python3 ../../site/hg-format/reproduce_hg_myths.py --verify-all --out ./local_repro_report.json
```

## Included Raw Run Ledgers

| File | What Was Measured | Key Verified Result |
|---|---|---|
| [`stage2_mesh_verify.json`](stage2_mesh_verify.json) | 14-petal `tsock://` (`4 B/tok`) cluster registration, `T=0` 4-way deterministic request clubbing, `mcp.json` stdio bridge, and `CloverMetricsLedger` `used_vs_unused` audit | `clubbing_answers_identical = true`, `clubbed_coalesced = 3`, `13` MCP tools verified |
| [`stage45_public_results.json`](stage45_public_results.json) | **Stage 4A:** `21` `Lean 4 v4.33.1 + Mathlib` theorems (`12` MiniF2F + `9` RH Tier-2) comparing `Nemotron-120B` solo vs. `Clover 4-Part + Goedel-Prover-V2-32B-HG4`<br>**Stage 4B:** `10` exact big-int obligations<br>**Stage 5:** Live `Chronos-Bolt-HG4`, `V-JEPA 2.1-HG4` (`1408-d`), and `Boltz-2-HG4-16R` (`529.55 MiB`, `7.57x` compression, `0.995939` cosine) | MiniF2F: `Goedel-32B + Clover` **`8/12` (`66.7%`, 3 Cycle-2 compiler repairs)** + `0B` Lean tactic cascade **`6/12` (`50.0%`, 0 GPUs)** vs. `Nemotron-120B` **`5/12` (`41.7%`)** |
| [`myth6_steering_results.json`](myth6_steering_results.json) | **Myth 6 (The Center Steering Wheel):** `6` underspecified domain-policy coding tasks (`AI Alone` unsteered vs. `Human Steering + AI`) | `AI Alone = 0/6 (0.0%, 595.0 mean tokens)` vs. **`Human + AI = 5/6 (83.33%, 1.17 mean cycles, 277.5 mean tokens = -53% tokens)`** |
| [`opus_reality_probes_specialists.json`](opus_reality_probes_specialists.json) | Bidirectional reality probes co-designed in adversarial review with `Claude Opus 5.5`: Vision Step-0 pixel check, `Probe 0` (`512 -> 48` time-series quantiles), `Probe 6` (`19` implicit discernment traps), and `Probes 1–4` (`yoshida4` symplectic integrator, KMP `10^18`, Peterson lock, `cube7`) | `0B` Periodogram+Conformal **`MASE = 0.3084`** & `Chronos-0.2B` **`0.4157`** vs. `2B/35B/120B` LLMs **`0/3`**; Implicit Discernment: `35B-LoRA` **`19/19` binary (`11/19` 4-way)** & `120B` **`19/19` binary (`14/19` 4-way)** vs. `Router-0.6B` `9/19` |

## Companion Runner Scripts

- [`../../site/hg-format/reproduce_hg_myths.py`](../../site/hg-format/reproduce_hg_myths.py) — Standalone `<5s` mathematical, wire-socket, and `0B` specialist verification script (no GPU or API key required; supports optional `--endpoint` live LLM A/B).
- [`../../research/benchmarks/run_stage45_public_bench.py`](../../research/benchmarks/run_stage45_public_bench.py) — Full `Lean 4 v4.33.1 + Mathlib`, `x-cruncher`, `Chronos-Bolt`, `V-JEPA 2.1`, and `Boltz-2` cluster runner.
- [`../../research/benchmarks/run_myth6_steering_bench.py`](../../research/benchmarks/run_myth6_steering_bench.py) — Live Myth-6 Human Steering vs. Unsteered Autopilot benchmark runner.
- [`../../research/benchmarks/run_opus_reality_probes.py`](../../research/benchmarks/run_opus_reality_probes.py) — Bidirectional multi-domain reality probe runner.
