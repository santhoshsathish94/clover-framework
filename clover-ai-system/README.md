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

| Model / Specialist | `HG4` Footprint | Minimum Required Hardware / Servers | Measured `RTX 3090` Single-User Decode (`batch = 1`) |
| :--- | :--- | :--- | :--- |
| **`0B` Computed Pack** | `0 MB` | Any phone, laptop, or CPU machine (`0` GPUs) | **`0.02–52 ms`** exact on CPU (`0` GPUs) |
| **`LFM3-Omni-2B` & Small Drafters** (`HG4-16R` & `5L-DSpark`) | `0.85 GB` | Any phone, laptop, CPU, or `1×` GPU | **`240–288 decode tok/s`** on `1× 3090` · `50.3 tok/s` CPU |
| **`Chronos-Bolt 0.2B`** (`HG4`) | `0.10 GB` | Any phone, laptop, CPU, or `1×` GPU | **`187–191 decode tok/s`** (`0.37 s`) · `<52 ms` `0B` Conformal |
| **`V-JEPA 2.1`** (`HG4-16R`) | `0.51 GB` | Any laptop, CPU, or `1×` GPU | **`0.12 s`/clip** (`1408-d` rollout) · `0.9 s` CPU |
| **`Boltz-2`** (`HG4-16R` 3D folding) | `0.53 GB` (`~14 GB` resident) | `1×` GPU server `>= 16 GB` VRAM (`1× RTX 3090`) | **`~12 s`/complex** on `1× 3090` *(skipped on CPU)* |
| **`Goedel-Prover-V2-32B`** (`HG4-16R`) | `15.8 GB` (`21.6 GB` @ 16k KV) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | **`18.4–28.6 decode tok/s`** *(skipped on CPU: `0.9 tok/s`)* |
| **`Clover-35B-A3B` Coordinator** (`HG4-16R-OP`) | `17.4 GB` (`20.2 GB` @ 32k KV) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | **`58.5 decode tok/s`** CUDA (`35.8 tok/s` HTTP; `4×` GPUs add `+0.0 tok/s`) |
| **`Nemotron 3.5 Super VL`** (`120B-A12B` `HG4-16R`) | `58.2 GB` (`68 GB` resident) | `4× 24 GB RTX 3090` server or `1× 80 GB` GPU (`>= 72 GB` VRAM) | **`38.4–55+ decode tok/s`** (`2L-MTP`; `8×` GPUs add `+0.0 tok/s`) |
| **`Kimi K3.1 Pro`** (`2.8T MoE` `K31H` two-tier `HG4+HG2`) | `741 GB` (`~780 GB` resident) | Same-rack GPU cluster `>= 780 GB` VRAM (`5× 8-GPU 3090` or `2× 8×80 GB`) | **`6.0–18 decode tok/s`** (`T=1` to `+DSpark`; *skipped without cluster*) |

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
