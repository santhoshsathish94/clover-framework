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

| Model / Specialist | `HG4` Footprint | Minimum Required Hardware / Servers | Interactive Gate |
| :--- | :--- | :--- | :--- |
| **`0B` Computed Pack** | `0 MB` | Any phone, laptop, or CPU machine (`0` GPUs) | **Ready on any device** (`<50 ms` CPU) |
| **`LFM3-Omni-2B`** (`HG4-16R`) | `0.98 GB` | Any phone, laptop, or CPU/GPU machine (`0` GPUs) | **Ready on any device** (`18–65 tok/s` CPU · `163 tok/s` GPU) |
| **`Chronos-Bolt 0.2B` & `V-JEPA 2.1`** (`HG4`) | `0.12–1.2 GB` | Any phone, laptop, or CPU/GPU machine (`0` GPUs) | **Ready on any device** |
| **`Boltz-2`** (`HG4` 3D folding) | `3.5 GB` (`~14 GB` resident) | `1×` GPU server `>= 16 GB` VRAM | **GPU required** *(skipped on CPU)* |
| **`Goedel-Prover-V2-32B`** (`HG4-16R`) | `16.4 GB` (`21.6 GB` @ 16k KV) | `1×` GPU server `>= 24 GB` VRAM (`RTX 3090 / 4090 / L4`) | **GPU required** (`18.4 tok/s`; *skipped on CPU*) |
| **`Clover-35B-A3B` Coordinator** (`HG4-16R-OP`) | `17.4 GB` (`20.2 GB` @ 32k KV) | `1×` GPU server `>= 24 GB` VRAM (`RTX 3090 / 4090 / L4`) | **GPU required** (`35.8 tok/s`; *skipped on CPU*) |
| **`Nemotron 3.5 Super VL`** (`120B-A12B` `HG4-16R`) | `60.2 GB` (`68 GB` resident) | `1× 80 GB` GPU or `4× 24 GB` GPU server (`>= 72 GB` VRAM) | **GPU required** (`22–38 tok/s`; *skipped on CPU*) |
| **`Kimi K3.1 Pro`** (`2.8T MoE` `K31H` two-tier `HG4+HG2`) | `741 GB` (`~780 GB` resident) | Same-rack GPU cluster `>= 780 GB` VRAM (`5× 8-GPU 24 GB` or `2× 8×80 GB`) | **In-rack cluster** (`6–18 decode tok/s`; *skipped without cluster*) |

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
