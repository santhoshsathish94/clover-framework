<p align="center">
  <a href="https://cloverframework.com/">
    <img src="assets/branding/social-preview.png" width="820"
         alt="Clover Framework — System, Human and AI working together for meaningful outcomes">
  </a>
</p>

<p align="center">
  <strong>Clover Framework</strong> · System, Human and AI working together for meaningful outcomes
</p>

<p align="center">
  <a href="https://cloverframework.com/"><strong>cloverframework.com</strong></a>
  ·
  <a href="https://cloverframework.com/ai-system/"><strong>AI Studio (v6.0)</strong></a>
  ·
  <a href="https://cloverframework.com/hg-format/"><strong>AI Mythbuster</strong></a>
  ·
  <a href="QUICKSTART.md">Quickstart</a>
  ·
  <a href="AGENTS.md">AGENTS.md</a>
  ·
  <a href="docs/README.md">Docs</a>
</p>

---

# Clover (`v6.0.0`)

**Clover** is a framework and open-source AI system for working with **System, Human, and AI** to produce meaningful outcomes through a repeating five-stage cycle:

```text
Context → Direction → Execution → Outcome → Growth
```

- **Context** — establish what is actually true in the system before deciding.
- **Direction** — define the objective, priorities, and boundaries *(owned by a named, accountable human)*.
- **Execution** — carry out the work within human Direction and system boundaries.
- **Outcome** — verify what the real system shows *(never claim unverified output as success)*.
- **Growth** — carry what the Outcome taught into the next cycle's Context.

> **The model can propose. The system decides. Deterministic tools compute. Evidence verifies. Humans remain accountable.**

---

## Start Here — Repository Navigation Map

| What you want to do | Where to go |
| :--- | :--- |
| **Understand the Clover Framework** | [**cloverframework.com**](https://cloverframework.com/) · [`QUICKSTART.md`](QUICKSTART.md) · [`docs/README.md`](docs/README.md) · [`docs/case-studies/`](docs/case-studies/) |
| **Operate any AI agent under Clover** | [`AGENTS.md`](AGENTS.md) — drop-in self-contained operating contract |
| **Self-Host Clover AI System `v6.0.0` From Source *(Desktop/Mobile GUI Apps: WIP)*** | [**cloverframework.com/ai-system/**](https://cloverframework.com/ai-system/) · [`clover-ai-system/README.md`](clover-ai-system/README.md) · [`download/README.md`](download/README.md) |
| **Read the 6 Broken AI Myths & `HG4` 4-Bit Hadamard Format** | [**AI Mythbuster (`cloverframework.com/hg-format/`)**](https://cloverframework.com/hg-format/) · [`site/hg-format/index.html`](site/hg-format/index.html) |
| **Reproduce all 6 Myths & `HG4` on your own machine (Zero Human Setup)** | [`site/hg-format/REPRODUCE_AGENTS.md`](site/hg-format/REPRODUCE_AGENTS.md) *(paste into any AI agent)* · [`site/hg-format/reproduce_hg_myths.py`](site/hg-format/reproduce_hg_myths.py) · [`site/hg-format/convert_to_hg.py`](site/hg-format/convert_to_hg.py) |
| **Inspect raw public benchmark ledgers (`2026-11-26`)** | [`benchmarks/public-runs-20261126/README.md`](benchmarks/public-runs-20261126/README.md) · [`research/benchmarks/`](research/benchmarks/) |
| **Read `v6.0.0` Release Notes & Changelog** | [`docs/reference/release-v6.md`](docs/reference/release-v6.md) · [`CHANGELOG.md`](CHANGELOG.md) |
| **Read the Next Milestone Roadmap (`v6.x → v7.0`: 5 Senses + 1 Brain)** | [`docs/runtime-context/one-brain-five-senses-research-context.md`](docs/runtime-context/one-brain-five-senses-research-context.md) |
| **Browse full repository layout & historical research** | [`docs/repository-organization.md`](docs/repository-organization.md) · [`research/README.md`](research/README.md) · [`TODO.md`](TODO.md) |

---

## Clover AI System (`v6.0.0` Self-Hosted Source · Desktop & Mobile GUI Apps in WIP)

In **Clover v6.0** ([`clover-ai-system/`](clover-ai-system/README.md), [`docs/reference/release-v6.md`](docs/reference/release-v6.md)), self-hosted deployments run one simpler **4-Part Steerable Clover System** directly from open source on your own hardware *(note: standalone Windows `.exe` and Android `.apk` GUI installers are **Work in Progress** and withheld until end-to-end UX verification completes)*:

1. **Part 1 — `AGENTS.md` Steering & Epistemic Boundaries:** Enforces `Context → Direction → Execution → Outcome → Growth`, pauses with `STOP_AND_ASK` before destructive or underspecified actions, and applies a Tier-0 tool-exposure gate so small `2B` models are never flooded with unrequested schemas.
2. **Part 2 — `mcp.json` (`0B` Computed Specialists First + `HG4` Specialist Mesh):**
   - **Computed > Predicted:** Built-in zero-parameter deterministic tools compute what neural weights should never guess: `x-cruncher` exact arbitrary-precision arithmetic (`A*B`, `A^B` past `6,000+` digits, `N!`, BBP hexadecimal $\pi$), a `0B` Periodogram + Split-Conformal `P10/P50/P90` time-series forecaster (`MASE = 0.3084` in `<110 ms` CPU, beating `2B/35B/120B` LLMs and `Chronos-0.2B`), a `0B` Defeasible Epistemic Rule Resolver (`180/180` accuracy), and Python AST verification.
   - **Hadamard-Grouped (`HG4-16R` / `K31H`) Models & `tsock://` `4 B/tok` Mesh:** Shrinks open models ~4× (`68.2 GiB` BF16 → `17.2 GiB` `HG4` / `17.4 GiB` w/ LoRA at `99.6%+` cosine fidelity) and streams binary token IDs at `4 Bytes/token` (`49.4×` smaller than HTTP JSON SSE).
3. **Part 3 — `<task_id>.md` Context Ledger & Local-First Git Memory:** Writes human-readable per-task context files and stores chats, verified outcomes, and encrypted credentials locally on your device (Windows DPAPI / Android KeyStore) with optional sync to your own **private** GitHub repository.
4. **Part 4 — Prerequisite-Gated SQLite Todo & `CloverMetricsLedger` Telemetry:** Tracks step dependencies (`todos` + `todo_deps`) and records live runtime metrics (`GET /api/clover/metrics`): used vs. idle models and tools, measured decode `tok/s` (`mean`/`p50`/`p95`/`max`), solver cycle counts, and Cycle-2 repair signals.

### Set Up One Model — or the Full Clover AI System — on Your Own Hardware

Inside **Settings → Model → Clover System Setup** (or from first launch via *Choose models / Use server*), users can configure **1 model**, a **custom subset**, or the **complete Clover AI System** across their device and any GPU servers they provide:

- **Small models & `0B` tools run interactively on *any* phone, laptop, or CPU machine (`0` GPUs required).**
- **Big models require GPUs to be interactive** — if a machine is CPU-only or lacks the required GPU VRAM, Studio automatically skips those models and displays the exact minimum server needed rather than letting a model crawl at `<1 tok/s`.

| Model / Specialist | `HG4` Footprint | Minimum Required Hardware / Servers | Measured Single-User Speed (`batch = 1`, Provenance) |
| :--- | :--- | :--- | :--- |
| **`0B` Computed Pack** (`x-cruncher` + Conformal + Defeasible) | `0 MiB` | Any phone, laptop, or CPU machine (`0` GPUs) | **`0.02–105 ms`** exact on CPU (`0` GPUs) |
| **`LFM3-Omni-2B`** (`HG4-16R`, `87/87` exact) | `0.83 GiB` (`853.9 MiB`) | Any phone, laptop, CPU, or `1×` GPU | **`240.2 decode tok/s`** (`1× 4090`) · `35.7` e2e `3090` · `50.3` CPU · `33.4` S23 |
| **`Chronos-Bolt 0.2B`** (`HG4-16R` time-series) | `0.10 GiB` (`102 MiB`) | Any phone, laptop, CPU, or `1×` GPU | **`187.1–191.5 steps/s`** `p50–max` (`151.9` mean, `0.80 s` on `1× 3090`) · `<110 ms` `0B` CPU |
| **`V-JEPA 2.1`** (`HG4-16R` physical dynamics) | `0.51 GiB` | Any laptop, CPU, or `1×` GPU | `64-frame` ViT-g `1408-d` rollout verified on `1× 3090` |
| **`Boltz-2`** (`HG4-16R` 3D biomolecular folding) | `0.52 GiB` (`529.6 MiB`; `~14 GiB` resident) | `1×` GPU server `>= 16 GB` VRAM (`1× RTX 3090`) | **`39.6–40.6 s`/complex** on `1× 3090` (`pLDDT 0.908 & 0.926`; *skipped on CPU*) |
| **`Goedel-Prover-V2-32B`** (`HG4-16R` `Lean 4` prover) | `15.8 GiB` (`21.6 GiB` @ 16k KV) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | Fits `1× 24 GB 3090` in VRAM *(skipped on CPU: `0.93 tok/s` on `k3`)* |
| **`Clover-35B-A3B` Coordinator** (`Qwen3.6-35B-A3B + LoRA`, `HG4-16R-OP`) | `17.2 GiB` (`17.4 GiB` w/ LoRA) | `1×` GPU server `>= 24 GB` VRAM (`1× RTX 3090`) | **`168.4 tok/s`** (`1× 4090` JSON) · `20.8–22.9` e2e `3090` JSON (`58.5` CLI*) |
| **`Nemotron 3.5 Super VL`** (`120B-A12B` `HG4-16R-OP`, `1M` context) | `58.2 GiB` (`4 × 14.6 GiB`) | `4× 24 GB RTX 3090` server or `1× 80 GB` GPU (`>= 72 GB` VRAM) | **`71.3 T=1 / 172.2 MTP`** (`4× 4090` JSON) · `13.4–14.5` e2e `3090` JSON (`38.4` CLI*) |
| **`Kimi K3.1 Pro`** (`2.8T MoE` `K31H` two-tier `HG4+HG2`) | `741 GB` (`~780 GB` resident) | Same-rack GPU cluster `>= 780 GB` VRAM (`5× 8-GPU 3090` or `2× 8×80 GB`) | **`7.89 tok/s`** (`8× 4090` 3-tier + `288.4 tok/s DSpark` JSON; `~6` `3090` CLI*) |

*\*JSON-badged figures are committed in `benchmarks/riemann-lean4/results/rtx4090_hg4_4models_benchmark.json` and `benchmarks/public-runs-20261126/*.json`; `CLI*` figures are uncommitted interactive host CLI measurements recorded during bring-up (not in standalone JSON files in this repository).*

---

## Six Broken AI Myths — Verified on Real Hardware (`2026-11-26`)

Every engineering claim in Clover v6.0 is backed by raw JSON ledgers ([`benchmarks/public-runs-20261126/`](benchmarks/public-runs-20261126/README.md)) and a standalone `<5s` zero-GPU/zero-API-key reproducibility suite (`python -W error site/hg-format/reproduce_hg_myths.py` or paste [`site/hg-format/REPRODUCE_AGENTS.md`](site/hg-format/REPRODUCE_AGENTS.md) into any coding agent):

1. **Myth 1 (*"AI is slow because chips aren't fast enough"*):** Decoding is bound by warehouse weight-truck bandwidth (`RAM` slower trucks vs `VRAM` faster trucks), not production-line compute — without dedicated `VRAM`, laptop CPU and GPU run identically (`12.4` vs `12.8 tok/s`, noise / zero real gain), while Walsh-Hadamard rotation (`HG4-16R`) shrinks weight cargo `4×` (`87/87` exact tokens on `LFM3-2B`, `86/86` on `Qwen3.6-35B`, `7.57×` on `Boltz-2`).
2. **Myth 2 (*"Adding more RAM or VRAM makes AI faster"*):** Memory capacity (`GB`) is only admission (`172.2 -> 171.8 tok/s` across `4× vs 8× RTX 4090` in `rtx4090_hg4_4models_benchmark.json`; `48.7 -> 48.7 tok/s` across `2 -> 128 GB` RAM and `33.4` vs `12.4 tok/s` S23 vs DDR4 laptop in host logs).
3. **Myth 3 (*"One giant server beats many small machines"*):** Multiple workers splitting one shared memory bus contend linearly, whereas independent right-sized GPUs multiply memory bandwidth (`4 × 936 = 3,744 GB/s` across `4× RTX 3090`; `4 × 168.4 = 673.4 tok/s` across `4× RTX 4090`, `4 × 58.5 = 234.1 tok/s` `3090` CLI*).
4. **Myth 4 (*"Connecting multiple machines is too slow over the network"*):** Binary token sockets (`tsock://`, `0xC107`) stream `4 Bytes/token` (`49.4×` smaller wire payload than HTTP JSON SSE) with `T=0` priority request clubbing.
5. **Myth 5 (*"One universal giant model beats domain specialists & computed tools"*):** `0B` computed tools (`10/10` exact BigInt math in `<1 ms` vs `7/10` `120B` solo; `0B` Periodogram+Conformal `MASE = 0.3084` in `<110 ms` CPU vs `2B/35B/120B` `0/3` valid; `180/180` Defeasible Resolver) and 3-cycle `32B Goedel-HG4 + Lean 4` compiler repair (`5/12` Turn-1 -> `8/12 = 66.7%` MiniF2F vs `5/12 = 41.7%` 1-turn `120B` solo at `3.7×` smaller size) beat monoliths cleanly.
6. **Myth 6 (*"AI agents solve problems all on their own"*):** On 6 underspecified domain-convention tasks, unsteered AI-alone scored `0/6` (`0.0%`, `STOP_AND_ASK = 0`, `595.0` mean tokens) while 1 sentence of Human Steering + Clover scored `5/6` (`83.33%`) in `1.17` mean cycles using `53%` fewer tokens (`277.5`).

---

## Prior Research Lineage (`v4–v5`, Archived in Repo)

Earlier neuro-symbolic research milestones — including the single-workstation `Kimi K3` CPU engine ([`clover-kimi-k3-cpu-context.md`](docs/runtime-context/engine-and-architecture/clover-kimi-k3-cpu-context.md)), the 11 Foundational Papers (`1943–2017`) & Direction B hardware data-movement study ([`research/clover-ai-system-paper/README.md`](research/clover-ai-system-paper/README.md), `33/70 → 70/70` across 3 backbones), Collective Problem-Solving Learning ([`research/collective-learning/`](research/collective-learning/README.md)), and SWE-Bench Pro V2 verification ([`benchmarks/swebench-pro-v2/`](benchmarks/swebench-pro-v2/README.md)) — remain preserved with full scripts and raw logs under `research/`, `benchmarks/`, and [`docs/reference/release-v4.md`](docs/reference/release-v4.md).

## Contributing

Clover is an open, evidence-driven project. Corrections, disagreements, independent reproductions, and practical improvements are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
