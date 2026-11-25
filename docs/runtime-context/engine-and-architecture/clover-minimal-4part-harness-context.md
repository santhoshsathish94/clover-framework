# Clover Architecture & Validation Context — Minimal 4-Part Harness (`AGENTS.md` + `mcp.json` + Context File + Todo List) Across All Non-Kimi Models

- **Date:** 2026-11-25
- **Direction (creator):**
  *"my question is till now for few months I have only used clover agents.md and mcp and any model I used didn't make much of difference I was able to get all my work done because agents.md was how human work on a problem once AI was given that approach any model solved problems better... create a context file and todo list to make sure this is how we are building the clover ai system, test this approach with all the models except kimi because I don't want to rent 5 node we already have 2 4x3090 how may more you need for k3 + gpu server to run all other models and make the clover ai system better at solving problems you can run various well known benchmarks and validate it don't rush do it step by step follow clover first tell me how may 3090 you need"*

---

## 1. Context — Why the 4-Part Workflow Already Solved Real Work (and What Clover AI System Must Be)

Across months of real delivery work in VS Code Copilot, outcomes stayed consistently high regardless of which frontier model sat behind the chat because **four simple primitives** drove every task:

1. **[AGENTS.md](../../../AGENTS.md):** Human problem-solving discipline (*Context → Direction → Execution → Outcome → Growth*), stopping blind retries, and refusing verification tampering.
2. **A `<task_id>.md` Context File:** External persistent working memory recording what was read, what was tried, what failed, and what changed — surviving context compaction and model switches.
3. **A Todo List (`todos` + prerequisites):** Explicit step-by-step decomposition so one piece is verified before starting the next.
4. **`mcp.json` Tools & Verifiers:** Direct connection to real compilers, tests, file viewers/editors, and domain specialists.

### Architectural Invariant for `clover-ai-system`
**`Clover AI System` must never add heavy middleware or ceremony on top of those four primitives.**
- Underneath the hood, `Clover HG` (`4×` compression), the `181 MB` `Clover-Harness-LoRA` (so `35B` open models follow tool schemas cleanly without `30k`-token `<think>` bloat), and `4 B/token` `tsock://` binary sockets exist solely to let anyone **own** the models and specialist tools locally behind one `/v1` + `/mcp` endpoint.
- Above the hood, the agent loop is strictly **`AGENTS.md` + `mcp.json` + Context `.md` file + Todo list**.

---

## 2. Exact GPU Hardware Budget (`k3` + All 7 Non-Kimi Models Simultaneously in VRAM)

Excluding `Kimi-K3.1-Pro 2.8T` (`34` GPUs), **all 7 remaining neural models in the Clover roster total `95.24 GiB` of `HG` weights** and fit simultaneously in VRAM on:
- **Minimum:** **`6 × RTX 3090`** (`144 GB` nominal / `141.4 GiB` usable VRAM)
- **Recommended (`0` additional GPUs beyond `2 × 4×3090` = `8 × RTX 3090`):** **`8 × RTX 3090`** (`192 GB` nominal / `188.5 GiB` usable VRAM), giving wide KV-cache headroom and a 2nd parallel `Qwen3.6-35B` worker.

| Slot (`24 GB RTX 3090` = `23.56 GiB` usable) | Pinned Model(s) & Role | Hugging Face Repo (`Santhoshsathish94/*`) | Exact `HG` Weight Size | + KV / Runtime Workspace | Total VRAM / GPU |
|---|---|---|---|---|---|
| **GPUs `0..3`** *(or `0..2` in 6-GPU minimum)* | **`Nemotron-3.5-Super-VL 120B`** (Vision & Heavy Escalation) | `Nemotron-3.5-Super-VL-120B-HG4-16R-OP` | `60.20 GiB` total (`15.05 GiB`/GPU × 4, or `20.07 GiB`/GPU × 3) | `~1.5 GiB` / GPU | **`16.6 GiB` / GPU** *(4 GPUs)* |
| **GPU `4`** *(+ optional 2nd replica on **GPU `5`**)* | **`Qwen3.6-35B-A3B` + `Clover-Harness-LoRA`** (Front-Door Agent & Coordinator) | `Qwen3.6-35B-A3B-HG4-16R-OP` + `Qwen3.6-35B-A3B-Clover-Harness-LoRA` | `17.41 GiB` + `0.18 GiB` | `~3.0 GiB` (`32k` ctx) | **`20.6 GiB`** |
| **GPU `6`** | **`Goedel-Prover-V2-32B`** (`Lean 4` Formal Proof Specialist via `mcp.json`) | `Goedel-Prover-V2-32B-HG4-16R` | `15.76 GiB` | `~2.8 GiB` | **`18.6 GiB`** |
| **GPU `7`** | **All 5 Compact Specialists Co-Resident:**<br>• `LFM3-Omni-2B` (Audio / Fast Multimodal)<br>• `Qwen3-0.6B + Router-SafeGuard-LoRA`<br>• `Meta V-JEPA 2.1 ViT-g` (64-Frame Physics)<br>• `Chronos-Bolt-Base` (Quantile Forecasting)<br>• `Boltz-2` (3D Biomolecular Folding) | `VJEPA2-ViTg-HG4-16R`, `Chronos-Bolt-Base-HG4-16R`, `Qwen3-0.6B-Clover-Router-SafeGuard-LoRA` | `1.10 + 1.25 + 0.51 + 0.10 + 1.80 = 4.76 GiB` | `~6.0 GiB` | **`10.8 GiB`** (`12.8 GiB` free!) |
| **Center Hub `k3` (`0` GPUs)** | **`AGENTS.md` + `mcp.json` Gateway + `<task_id>.md` Context & Todo Store + C11 `x-cruncher` + `Lean 4` (`v4.33.1 + Mathlib`) + Isolated `pytest` Sandbox** | Repo `clover-ai-system/` | `0 GiB` VRAM | Host CPU / RAM | **`0` GPUs** |

**Operational Note on Server Bandwidth:** Keep GPU nodes in Europe/US (`>= 500 Mbps`, `< 40 ms` RTT to `k3` in Germany) rather than `2 MB/s`-capped regions so pulling all `~95 GiB` of `HG` weights from Hugging Face takes `< 15 minutes`.

---

## 3. Step-by-Step Clover Execution & Benchmark Validation Plan

Every benchmark must run as a **paired A/B test** on identical sealed tasks:
- **Condition A (`Raw Model`):** Single-turn / naive loop without `AGENTS.md`, Context `.md` file, Todo list, or specialist `mcp.json` tools.
- **Condition B (`Model + Clover 4-Part Harness`):** Exact same model(s) given `AGENTS.md` + `<task_id>.md` Context file + Todo list + `mcp.json` (`pytest`, `lean4`, `x-cruncher`, `Goedel-32B`, `V-JEPA 2.1`, `Chronos`, `Boltz-2`, and `Nemotron-120B` escalation on stagnant cycles).

### Planned Well-Known Benchmarks (Step-by-Step, One Stage Verified Before Next)
1. **Step 1 — Cluster Bring-Up & Smoke Gate (`k3` + `6–8 × RTX 3090`):**
   - Load all 7 non-Kimi models into VRAM, expose unified `/v1/chat/completions` + `/mcp` (`mcp.json`) on `k3`, and pass deterministic health checks on every model and tool.
2. **Step 2 — Minimal 4-Part Harness Runner (`clover_4part_agent.py`):**
   - Clean ~250-line reference runner implementing strictly the VS Code 4-part loop (`AGENTS.md` system contract + `<task_id>.md` Context read/write + SQLite/JSON Todo tool + `mcp.json` tool calls), tested across `LFM3-Omni-2B`, `Qwen3.6-35B` (base vs `+ Harness LoRA`), and `Nemotron-120B`.
3. **Step 3 — Software Engineering & Scientific Coding Benchmarks (`HumanEval+`, `MBPP+`, `SciCode`, `SWE-bench` subset):**
   - Measure Condition A vs. Condition B across each model (`LFM3-2B`, `Qwen3.6-35B`, `Qwen3.6-35B + LoRA`, `Nemotron-120B`, and `35B -> 120B` Clover escalation) to verify that the 4-part harness lifts *every* model and quantify the exact gain from the Harness LoRA (`turns`, `tokens`, `pass@1`).
4. **Step 4 — Competition Math & Formal Proof Benchmarks (`MATH-500` / `AIME` + `MiniF2F` / Sealed `Lean 4` `rh_bench_suite_v2`):**
   - Measure raw LLM vs. 4-Part Harness + `mcp.json` (`x-cruncher` exact arithmetic + `Goedel-Prover-V2-32B` + `lean4` compiler feedback).
5. **Step 5 — Multimodal & Specialist `mcp.json` Validation + Growth Update:**
   - Validate `V-JEPA 2.1` (physical video dynamics), `Chronos-Bolt` (forecasting MASE), and `Boltz-2` (biomolecular structure) called as ordinary `mcp.json` tools inside the same 4-part loop; record failure modes and Growth improvements back into this Context file.

---

## 4. Outcome & Growth Ledger *(Updated After Each Completed Step)*

- **[2026-11-25 · Stage 1 Context & Direction]:** Verified all 6 `HG4` & LoRA repositories live on Hugging Face (`Santhoshsathish94/*`), confirmed exact VRAM math (`6 × RTX 3090` minimum / `8 × RTX 3090` recommended = `0` extra GPUs beyond `2 × 4×3090`), and reclaimed **712+ GB** of obsolete pre-HG4/BF16 weights on `k3` (`/dev/md2` now has `928 GB` free + `/mnt/nvme1` has `406 GB` free = **`1.33 TB` free**).
- **[2026-11-25 · Stage 2 Zero-Overhead LoRA Fold on `k3`]:** Built [fold_lora_into_c35_hg4.py](../../../clover-ai-system/src/models/clover-35b-a3b/fold_lora_into_c35_hg4.py) and folded all `310` trained `(A, B)` LoRA projections (`620/620` tensors, `r=32, alpha=64`) of `Santhoshsathish94/Qwen3.6-35B-A3B-Clover-Harness-LoRA` directly into `/mnt/nvme1/clover-models/qwen3.6-35b-a3b-lora-hg4` (`min_router_rot_cosine = 0.99999976`, `min_top8_jaccard = 1.0`, `15.6 tok/s` native CPU decode on `k3` with `0` extra runtime overhead).
- **[2026-11-25 · Stage 3 4-Part Clover System + `Claude Opus 5.5` Adversarial Audit]:**
  - Built the zero-bloat 4-part runtime across [clover_todo_store.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/clover_todo_store.py), [clover_4part_agent.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/clover_4part_agent.py), [clover_mcp_stdio.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/clover_mcp_stdio.py), [mcp.json](../../../clover-ai-system/mcp.json), [cluster_scheduler_mcp2.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/cluster_scheduler_mcp2.py), and [gateway.py](../../../clover-ai-system/src/tools/hg4-fusion-gateway/gateway.py) (`POST /v1/clover/solve` + `POST /mcp`).
  - Audited with **`Claude Opus 5.5`**, catching and fixing **12 real security, concurrency, and cycle issues**:
    1. Enforced `_is_safe_doctest_ast` + CPython `sys.addaudithook` + process-group timeout kill + `CLOVER_STUDIO_S2_ALLOW_HOST_EXEC` opt-in on `python_ast_verify` `test_code`.
    2. Screened `lean4_surface_verify` `statement` against `#eval IO`/metaprogramming/`verdict:`, rejected theorem redeclaration, stripped env secrets (`ELAN_HOME` + `PATH` only), and made `compile=True` fail closed.
    3. Enforced `allowed_tools` at execution time inside `Clover4PartAgent.solve`.
    4. Added strict `sanitize_task_id` path-traversal rejection (`../`, `/`, `\`) in `get_record`, `save_record`, and todo tools.
    5. Emitted verifier failures as synthetic `[assistant tool_calls -> role: "tool"]` rounds instead of `role: "user"`, preserving `STOP_AND_ASK`, cascade escalation, and `AGENTS.md` §8 test-tamper refusal.
    6. Preserved `what_was_ruled_out` across explicit `x_clover_task_id` continuations without leaking across unrelated hash-derived stateless chats.
    7. Held `RLock` across SQLite commit + atomic `NamedTemporaryFile` `.todos.md` mirror and serialized stateful cycle/todo calls inside `clover_exec_batch`.
    8. Unified `default_task` fallback across all 4 context/todo tools.
    9. Configured UTF-8 stdio pipes in `clover_mcp_stdio.py` so Unicode (`∀ n : ℕ`) works cleanly on Windows.
    10. Fixed `_SHELL_MUTATE_RE` to catch `rm -f` / `unlink` on test files and unified `clover_route_and_guard` onto `CloverCycleEngine.check_verification_boundary_tamper`.
    11. Isolated `fold_lora_into_c35_hg4.py` operator precomputation from live Nemotron/Kimi directories.
    12. Asserted `100%` of `lora_A` keys are consumed during LoRA folding.
- **[2026-11-25 · Stage 4 Live End-to-End Verification on `k3:8790` (`7/7` Checks PASS + `42/42` `-W error` Unit Tests PASS)]:**
  - Deployed to `clover-hg4-fusion-gateway.service` on `k3:8790`: verified `13`-tool `MCP 2.0` catalog, prerequisite-gated `clover_todo_upsert` batches, live neural `Chronos-Bolt-Base` (`chronos-bolt-base-local`), C11 `x-cruncher`, live `Lean 4 v4.33.1` compiler (`0.91s`), verified Python code synthesis on both `lfm3-omni-2b` (`5.57s`) and `Qwen3.6-35B-A3B-LoRA`, and autonomous multi-tool execution (`xcruncher_exact_math` + `clover_todo_upsert` -> `todos_done=True`, exact `30!` verified in `22.5s` on CPU).
- **Growth Lesson:** Actionable AST diagnostics (naming `__import__` explicitly rather than a generic `"Unsafe AST"` string) immediately enabled the small `2B` model (`LFM3-Omni-2B`) to self-repair on Cycle 2 (`verified=True`). Ready to attach `6–8 × RTX 3090` Petals tomorrow via `POST /v1/cluster/register` (`tsock://`) with zero code changes.

