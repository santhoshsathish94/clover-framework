# Clover AI System & Clover AI Studio (`v6.0.0`)

`clover-ai-system/` contains the self-hosted **Clover v6.0 HG4 Fusion & Specialist Mesh** (`src/tools/hg4-fusion-gateway/`, `studio/clover/clover_v6_system.py`) alongside the **Work in Progress (`WIP`)** standalone Desktop (`studio/`) and Android Mobile (`mobile/`) GUI wrappers — see [`CLOVER_FRAMEWORK_ARCHITECTURE.md`](CLOVER_FRAMEWORK_ARCHITECTURE.md) and [`../docs/reference/release-v6.md`](../docs/reference/release-v6.md).

> **Work in Progress Note:** Standalone Windows `.exe` and Android `.apk` GUI installers are **Work in Progress** and are not published for download ([`../download/README.md`](../download/README.md)). Run Clover v6.0 directly from Python/C source on your own Linux/Windows/GPU machine as shown below.

---

## 1. The Simpler v6.0 4-Part Steerable System

Instead of forcing every request through heavyweight post-inference neural layers, Clover v6.0 organizes every surface around four minimal, verifiable components ([`studio/clover/clover_v6_system.py`](studio/clover/clover_v6_system.py) and [`src/tools/hg4-fusion-gateway/clover_4part_agent.py`](src/tools/hg4-fusion-gateway/clover_4part_agent.py)):

1. **`AGENTS.md` (Human Steering & Epistemic Boundaries):**
   - Runs `Context → Direction → Execution → Outcome → Growth` with `STOP_AND_ASK` human approval gates before destructive or underspecified actions and a Tier-0 tool-exposure gate that prevents schema flooding on compact `2B` models.
2. **`mcp.json` (`0B` Computed Specialists First + `HG4` Specialist Mesh):**
   - **Computed > Predicted:** Built-in deterministic tools compute exact answers in `<50 ms` on any CPU with zero neural parameters:
     - `xcruncher_exact_math` / `calculate`: Exact big-integer multiplication (`A*B`), binary exponentiation (`A^B` past `6,000+` digits), factorials (`N!`), and BBP hexadecimal $\pi$ ([`studio/clover/local_tool_packs.py`](studio/clover/local_tool_packs.py)).
     - `forecast_series_0b`: `0B` Periodogram + Joint Harmonic OLS + Split-Conformal `P10/P50/P90` time-series forecaster (`MASE = 0.3084`).
     - `resolve_defeasible_rules_0b`: `0B` Defeasible Epistemic Rule Resolver (`180/180` accuracy, filtering `rejected` rules and honoring `unless` exceptions + specificity).
     - `python_ast_verify`: Deterministic Python AST syntax & structure verifier.
   - **`HG4` Specialist Mesh (`src/tools/hg4-fusion-gateway/`):** Connects `Qwen3.6-35B-A3B + Clover-Harness-LoRA`, `Goedel-Prover-V2-32B` (`Lean 4`), `Boltz-2` (3D biomolecular folding), `Chronos-Bolt`, `V-JEPA 2.1`, `Nemotron 3.5 Super VL` (`120B`, `1M` context), and `Kimi K3.1 Pro` (`2.8T`) over `4 Bytes/token` `tsock://` binary sockets and `MCP 2.0`.
3. **`<task_id>.md` (Durable Task Context & Private Git Memory):**
   - Writes human-readable per-task context files and keeps sessions, verified outcomes, and OS-encrypted credentials (`vault.py` via Windows DPAPI / Android KeyStore) on your device, with optional sync to your own **private** GitHub repository.
4. **Prerequisite-Gated SQLite Todo & `CloverMetricsLedger` Telemetry:**
   - Enforces step dependencies (`todos` + `todo_deps`) and exposes live telemetry (`GET /api/clover/metrics` / `GET /v1/clover/metrics`) tracking used vs. idle models and tools, measured decode `tok/s` (`mean`/`p50`/`p95`/`max`), cycle counts, and Cycle-2 self-repair signals.

---

## 2. Flexible Model & Server Setup (`1` Model or Full System)

In **Settings → Model → Clover System Setup** (or on first launch via *Choose models / Use server*), users can deploy **1 model**, a **custom subset**, or the **complete Clover AI System** across their own device and any GPU servers they provide (`GET /api/clover/setup/catalog`, `POST /api/clover/setup/plan`, `POST /api/clover/setup/apply`):

| Model / Specialist | `HG4` Footprint | Minimum Required Hardware / Servers | Measured Single-User Speed (`batch = 1`, Provenance) |
| :--- | :--- | :--- | :--- |
| **`0B` Computed Pack** | `0 MiB` | Any phone, laptop, or CPU machine (`0` GPUs) | **`0.02–105 ms`** exact on CPU (`0` GPUs) |
| **`LFM3-Omni-2B`** (`HG4-16R`, `87/87` exact) | `0.83 GiB` (`853.9 MiB`) | Any phone, laptop, CPU, or `1×` GPU | **`240.2 decode tok/s`** (`1× 4090`) · `35.7` e2e `3090` · `50.3` CPU |
| **`Chronos-Bolt 0.2B`** (`HG4-16R`) | `0.10 GiB` (`102 MiB`) | Any phone, laptop, CPU, or `1×` GPU | **`187.1–191.5 steps/s`** `p50–max` (`151.9` mean, `0.80 s` on `1× 3090`) · `<110 ms` `0B` CPU |
| **`V-JEPA 2.1`** (`HG4-16R`) | `0.51 GiB` | Any laptop, CPU, or `1×` GPU | `64-frame` ViT-g `1408-d` rollout verified on `1× 3090` |
| **`Boltz-2`** (`HG4-16R` 3D folding) | `0.52 GiB` (`529.6 MiB`; `~14 GiB` resident) | `1×` GPU server `>= 16 GB` VRAM (`1× RTX 3090`) | **`39.6–40.6 s`/complex** on `1× 3090` (`pLDDT 0.908 & 0.926`; *skipped on CPU*) |
| **`Goedel-Prover-V2-32B`** (`HG4-16R`) | `15.8 GiB` (`21.6 GiB` @ 16k KV) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | Fits `1× 24 GB 3090` in VRAM *(skipped on CPU: `0.93 tok/s` on `k3`)* |
| **`Clover-35B-A3B` Coordinator** (`HG4-16R-OP`) | `17.2 GiB` (`17.4 GiB` w/ LoRA) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | **`168.4 tok/s`** (`1× 4090` JSON) · `20.8–22.9` e2e `3090` JSON (`58.5` CLI*) |
| **`Nemotron 3.5 Super VL`** (`120B-A12B` `HG4-16R-OP`) | `58.2 GiB` (`4 × 14.6 GiB`) | `4× 24 GB RTX 3090` server or `1× 80 GB` GPU (`>= 72 GB` VRAM) | **`71.3 T=1 / 172.2 MTP`** (`4× 4090` JSON) · `13.4–14.5` e2e `3090` JSON (`38.4` CLI*) |
| **`Kimi K3.1 Pro`** (`2.8T MoE` `K31H` two-tier `HG4+HG2`) | `741 GB` (`~780 GB` resident) | Same-rack GPU cluster `>= 780 GB` VRAM (`5× 8-GPU 3090` or `2× 8×80 GB`) | **`7.89 tok/s`** (`8× 4090` 3-tier + `288.4 tok/s DSpark` JSON; `~6` `3090` CLI*) |

---

## 3. Directory Map

```text
clover-ai-system/
├── studio/
│   ├── clover/                        # Shared Studio v6.0 core (Windows, Android & Web)
│   │   ├── server.py                  # Studio HTTP/SSE + OpenAI /v1 + /api/clover/* server
│   │   ├── clover_v6_system.py        # 4-Part Harness, 0B Specialists, Setup Planner & Metrics Ledger
│   │   ├── local_tool_packs.py        # Deterministic Phase 1 & 2 local tools + x-cruncher math
│   │   ├── k3_host_engine.py          # Remote & k3 HG4 gateway engine client
│   │   └── web/                       # Canonical app.html & first-run setup.html UI
│   ├── studio/                        # Native C# Windows Launcher.cs, Setup.cs & build_setup.ps1
│   └── tests/                         # Studio v6, 41-check Unified UI & E2E quality suites
├── mobile/                            # Android Mobile packaging & ARMv9/Adreno C11 engines
├── vscode-extension/                  # VS Code LanguageModelChatProvider & Studio bridge (v6.0.0)
└── src/
    ├── tools/hg4-fusion-gateway/      # HG4 Fusion Gateway (:8790), tsock:// mesh & MCP 2.0 server
    ├── models/                        # HG4-16R / K31H C & CUDA kernels (LFM3, Qwen3.6-35B, Nemotron, Kimi)
    └── rust/                          # Validated Rust host-path memory & routing crates
```

---

## 4. Quick Run & Verification

```powershell
# Run Clover AI Studio locally on any port
python clover-ai-system\studio\clover\server.py --base C:\tmp\clover-studio --port 8765

# Run v6.0 Studio, Unified UI (41 checks), and HG4 Gateway (43 checks) test suites
python -W error clover-ai-system\studio\tests\test_clover_v6_studio_system.py
python clover-ai-system\studio\tests\test_studio_unified_ui.py
python -W error -m unittest discover -s clover-ai-system\src\tools\hg4-fusion-gateway -p "test_*.py"
```

*(Historical `v4–v5` `Kimi-K3` single-workstation CPU and `HybridAgent`/`NCM` experiments remain preserved in `src/code/`, `src/tools/hybrid-ncm-k3/`, and [`../research/clover-ai-system-paper/README.md`](../research/clover-ai-system-paper/README.md).)*
