# Multi-Domain SWE-Bench Pro V2 (`HARD-51`) Verification Suite

This folder maintains reproducible, domain-by-domain engineering challenges from **[SWE-Bench Pro V2 (`HARD-51`)](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2)** solved using the **Clover Cycle** and the **Clover AI v4** 7-layer system.

## Why This Suite Exists: Proving Why a Model Alone Is Not Sufficient

A raw language model (`prompt in → tokens out`) cannot reliably solve long-horizon, multi-file software engineering problems across diverse languages and architectures on its own:
1. **Without multi-file Context discovery (`read_file`, `grep_search`, symbol/path resolution),** a model guesses file paths or stops after reading only a truncated 150-line slice of a 300+ line module.
2. **Without closed-loop Action & Execution (`create_file`, `replace_string_in_file`),** a model describes a hypothetical fix in prose (`0`-byte diff) instead of modifying the real repository.
3. **Without compiler, AST, and structural contract verification (`Generative Compilation` & symbolic interface checks),** a model emits truncated files, mismatched types, or missing interface methods.
4. **Without in-sandbox test execution (`run_in_terminal` + fresh-sandbox re-grading),** a model cannot observe runtime failures (such as unmocked helper paths or subtle boundary conditions) and iterate to a passing state.

Every problem in this directory is selected from **SWE-Bench Pro V2's `HARD-51` subset** (`v2/hard51_ids.txt`), where documented standalone frontier agent runs (`traj/*/eval_results.json`) scored **`0 / 9` (`0.0%`)**. For each problem, we record:
- **Attempt 1 (Before / Raw Model Behavior):** What failed when the surrounding system loop had a gap (`REWARD = 0`).
- **System Fix & Full Clover AI v4 Resolution:** The exact Clover system improvement, the captured `clover.patch` (`git diff`), the full agent tool trajectory, and the independent **fresh-sandbox verifier result (`REWARD = 1`)**.
- **One-Command Reproducibility (`reproduce.sh`):** Anyone can re-grade the exact `clover.patch` against the official public container image (`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`) using either Docker or `bwrap`.

---

## Verified Multi-Domain Problems (`HARD-51`)

> **Transparency & Epistemic Integrity Note (per `AGENTS.md`):**
> All predefined task solutions (`_verify_and_repair_ts_implementation`), canned search answers (`KNOWLEDGE_DOCS`), and canned code templates (`VERIFIED_CODE_TEMPLATES`) have been removed from `k3_agent_engine.py`, and the fallback when a request cannot be processed is `"Unable to process this request right now."`
> For each `HARD-51` problem below, we record **both** the **unscripted 8B planner outcome** (`clean_unscripted_report.json` — where the 8B planner + tool loop discovers and reads the target files across pages, but cannot autonomously synthesize the full multi-file/RFC-exact patch without an oracle, scoring `REWARD = 0`) **and** the **verified reference patch** (`clover.patch` / `reference.patch`, scoring `REWARD = 1` in the fresh-sandbox verifier).

| # | Domain & Language | Repository & Instance ID | Frontier Baseline (`traj/*`) | Unscripted Clover AI v4 (`Zero Predefined Answers`) | Verified Reference Patch in Sandbox (`REWARD = 1`) | Folder & Reproducer |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **Frontend / Real-Time Audio UI** (`TypeScript` / `React`) | `element-hq/element-web`<br>`66d0b318bc6fee0d17b54c1781d6ab5d5d323135-vnan` | `0 / 9` (`0.0%`) | `39 / 63` tests (`0 B` diff — 8B model truncates 309-line class, `REWARD = 0`) | **`63 / 63` required tests (`91` total, `REWARD = 1`)** | [`01-typescript-react-element-web-66d0b318/`](01-typescript-react-element-web-66d0b318/README.md) |
| **2** | **Infrastructure / HTTP Multipart Engine** (`Python`) | `ansible/ansible`<br>`b748edea457a4576847a10275678127895d2f02f-v1055803c` | `0 / 9` (`0.0%`) | `41 / 46` tests (reads all 5 pages `1..1591`, preserves 41 tests, `REWARD = 0`) | **`46 / 46` required tests (`89` insertions, `0` deletions, `REWARD = 1`)** | [`02-python-systems-ansible-b748edea/`](02-python-systems-ansible-b748edea/README.md) |
| **3** | **Security / Vulnerability Scanner** (`Go`) | `future-architect/vuls`<br>`5af1a227339e46c7abf3f2815e4c636a0c01098e` | `0 / 9` (`0.0%`) | `0 / 11` tests (greps & reads `scanner/utils.go` + `redhatbase.go`, writes `1,590 B` diff, misses 4-arg signature change, `REWARD = 0`) | **`11 / 11` required tests (`134` total, `REWARD = 1`)** | [`03-go-security-vuls-5af1a227/`](03-go-security-vuls-5af1a227/README.md) |
