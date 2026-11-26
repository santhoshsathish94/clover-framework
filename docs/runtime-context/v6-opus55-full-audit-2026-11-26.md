# Clover v6.0.0 Full-Repository Adversarial Audit (`Claude Opus 5.5`)

- **Date:** 2026-11-26
- **Branch:** `audit/v6.0-opus55-full-audit` (branched from `origin/main` @ `ae9272cb2d3d18b09c3a5ddb06ddf4f40793ccde`)
- **Direction (Human):**
  *"create a new branch and audit all the work done in clover using opus 5.5 argue don't give up unless it highlights real issues follow clover create a context file and todo list"*

---

## 1. Context & Operating Rules (`AGENTS.md`)

Under [AGENTS.md](../../AGENTS.md), capability and criticism must both answer to **evidence from the real System** (*Context → Direction → Execution → Outcome → Growth*):

1. **Full Scope Audited:** Every surface shipped or modified in Clover `v6.0.0` (`124+` changed files + surrounding core modules):
   - **Track A — Runtime, Studio, 4-Part Agent, Mesh, Kernels & Reproducibility:**
     - `clover-ai-system/studio/clover/` (`clover_v6_system.py`, `server.py`, `local_tool_packs.py`, `k3_host_engine.py`, `web/app.html`, `web/setup.html`)
     - `clover-ai-system/src/tools/hg4-fusion-gateway/` (`clover_4part_agent.py`, `clover_cycle_engine.py`, `clover_todo_store.py`, `clover_telemetry_ledger.py`, `clover_token_socket.py`, `clover_mcp_stdio.py`, `cluster_scheduler_mcp2.py`, `fusion_router.py`, `gateway.py`, `backends.py`, `studio_xcruncher_verifier.py`)
     - `clover-ai-system/src/tools/` (`specialist_hg_server.py`, `kimi_pipeline_cluster.py`, `clover_hg_serve.py`, `convert_to_hg.py`, `deploy_5node_cluster.sh`)
     - `clover-ai-system/src/models/` (`fold_lora_into_c35_hg4.py`, `kimi_k31_hg4_engine.c`, `lfm3_hg4_engine.c`)
     - `site/hg-format/reproduce_hg_myths.py`, `clover-ai-system/mcp.json`, `clover-ai-system/vscode-extension/`
   - **Track B — Public Website (`site/`), `AGENTS.md` Fidelity, Benchmarks, Docs & Release Hygiene:**
     - All `site/` pages (`index.html`, `ai-system/index.html`, `hg-format/index.html`, `hg-format/REPRODUCE_AGENTS.md`, `getting-started/index.html`, `governance/index.html`, `privacy/index.html`, `glossary/index.html`, `author/index.html`, `clover-ai/index.html`, `404.html`, `assets/site.js`, `assets/site.css`, `llms.txt`, `sitemap.xml`)
     - Root & architecture docs (`AGENTS.md`, `README.md`, `TODO.md`, `CHANGELOG.md`, `VERSION`, `docs/`, `download/README.md`, `release/beta/`, `benchmarks/public-runs-20261126/`)
     - CI gates (`scripts/check-site.py`, `scripts/check-links.py`, `.github/workflows/`)
2. **Adversarial Debate Rule ("Do Not Roll Over"):**
   - Every finding raised by `Claude Opus 5.5` is checked directly against exact `file:line` source, runtime execution, and design invariants.
   - Speculative claims, stylistic preferences, intentional design choices, or already-mitigated paths are **challenged back via multi-turn `write_agent`** with concrete code/execution evidence until `Opus 5.5` either proves an actual failure mode or concedes the point.
   - **Real issues** (reproducible logic bugs, security/injection holes, race conditions, stale/contradictory numbers or claims across pages/code, broken anchors/UI paths, or `AGENTS.md` boundary violations) are accepted, recorded below, fixed surgically with regression tests, and re-verified by `Opus 5.5`.

---

## 2. Initial Pre-Audit Observation (Cycle 0)

- **[Found during Context setup]:** [docs/runtime-context/README.md](README.md) header states *"all 27 engineering context documents... into three clear domain subfolders"*, omitting `clover-minimal-4part-harness-context.md`, `clover-vscode-studio-integration-context.md`, and three entire subfolders (`consciousness-and-subjective-experience/`, `curiosity-and-developmental-learning/`, `superagent-jev-dory/` — `84` markdown files total). Tracked for Track B verification.

---

## 3. Audit Rounds, Adversarial Debate & Disposition Ledger

### Pass 1 Summary (`opus-audit-runtime` + `opus-audit-site-docs`)

#### A. Verified Real Issues Accepted for Fix
- **Runtime CRITICAL (`C1–C3`):**
  - `C1`: Hardcoded `DEFAULT_K3_ACCOUNT_TOKEN` in `clover-ai-system/studio/clover/k3_host_engine.py:89` used as fallback host key and HMAC signing secret in `server.py:2508`.
  - `C2`: Pre-auth `/api/multi-node/{turn,sync,client-turn}` in `server.py:3293` (before `_token_ok` at `:3519`), `importlib.reload` per call, unbounded journal, and self-recursion in `multi_node_coordinator.py`.
  - `C3`: `_build_bootstrap_cmd` in `clover_v6_system.py:790` uses `$CLOVER_GATEWAY_KEY` (mismatching `CLOVER_GATEWAY_API_KEY`), passing `--api-key ""` which makes `gateway.py` bind `0.0.0.0:8790` with open auth (`_authorized() == True`).
- **Runtime HIGH (`H1–H7`):**
  - `H1`: Lean verifier `studio_outcome_verifier.py:389` `"verdict:" in prompt.lower()` bypassed by identifier substring `(xverdict: Nat)` allowing `sorry`, plus newline column-0 `#eval IO.Process.run` reaching `lake env lean`.
  - `H2`: `clover_cycle_engine.py` `_HUMAN_ALLOWS_TEST_EDIT_RE` matches negative instructions (`"do not modify any tests"`), misses `sed -i.bak`, `python3 -c`, `git restore`, `__tests__/`, VS Code edit tool names, and honors client-supplied `x_clover_agents_cycle: false`.
  - `H3`: `server.py` Google `tokeninfo` lacks `aud` verification; `gho_`/`ghu_` OAuth tokens accepted from arbitrary OAuth apps (confused deputy).
  - `H4`: `clover_v6_system.py` SSRF guard allows `100.64.0.0/10` (`ip.is_private` is False for RFC 6598; requires `ip.is_global`), re-resolves DNS in `urlopen`, trusts client dict `reachable`/`served_models`, lacks `resp.read` size cap & server-count cap, blocks on `with ThreadPoolExecutor`, and `/api/clover/setup/plan` lacks Web owner gate.
  - `H5`: Shared Web non-owner access on `/api/clover/mcp`, `/api/clover/metrics/reset`, empty `turn_id` `/api/cancel`, cross-user `task_id` in `/api/clover/solve`, and unbounded `max_cycles` (`M16`).
  - `H6`: `clover_token_socket.py` & `kimi_pipeline_cluster.py` lack pre-auth `conn.settimeout`, thread semaphore, and default `127.0.0.1` / mandatory secret on non-loopback hosts; `TokenSocketPool._checkout` socket leak on handshake error; `OP_DONE` wire byte accounting (`+64` vs `12 + len(done_bytes)`, `M10`).
  - `H7`: `vscode-extension/package.json` lacks `"scope": "application"` on `endpointUrl`/`apiKey`, and `extension.js:438` defaults unnamed tool calls to `run_in_terminal`.
- **Runtime MEDIUM/LOW (`M1–M4`, `M6–M11`, `M13–M17`):**
  - `M1`: `studio_xcruncher_verifier.py` partial sub-expression match (`2^64 - 1` -> `2^64`).
  - `M2`: `clover_v6_system.py` `solve_4part` partial match (`1.5^2` -> `5^2`), `2**3**2` associativity, and `Error:` marked `[Verified]` / `SOLVED`.
  - `M3`: `local_tool_packs.py` `calculate()` `-2^2`, `-7 // 2`, `-7 % 3`, and small `5!`.
  - `M4`: Studio `CloverTodoStore` missing prerequisite existence check, cycle detection, valid status check, and `INSERT OR REPLACE` status clobber vs gateway `CloverTodoStore`.
  - `M6`: `clover_4part_agent.py` `if allowed_names:` treats `[]` as all tools, and `"2b"` substring matches `32b`/`a12b` while missing `omni`.
  - `M7` & `M8`: `cluster_scheduler_mcp2.py` MCP forwarding ignores `X-Clover-Mcp-Hop`; clubbing key omits `x_clover_*` flags.
  - `M9`: `reproduce_hg_myths.py` Myth 1 `if False` & tautological `nbytes*4` pass check; Myth 4 magic header/DONE frame mismatch; Myth 5 not importing shipped `0B` functions; `--verify-all` unused.
  - `M11`: `resolve_defeasible_rules_0b` `NaN` / non-numeric `priority` and `null` `updated_seq`.
  - `M13–M15`, `M17`: `deploy_5node_cluster.sh` `wait` exit status & secret redaction in `--dry-run`; `fold_lora_into_c35_hg4.py` `use_rslora`/DoRA/same-dir guard; `clover_mcp_stdio.py` JSON-RPC error wrap; unified tool contracts between Studio and Gateway.
- **Website, Docs & Release Hygiene (`Site-C1`, `Site-C3`, `Site-H1..H7`, `Site-H9`, `Site-M1..M17`):**
  - `Site-C3`: `site/clover-ai/` and `site/admin/` still present in `site/` and deployed publicly by `.github/workflows/pages.yml`.
  - `Site-H2`: Tracked `82.7 MB` `clover-ai-system/mobile/android-app/CloverAIStudio.apk` and `clover-local-5.0.7.vsix` still in git.
  - `Site-C1` & `Site-H1` & `Site-H3..H7` & `Site-H9`: Exact alignment of all website/README tables with committed JSON ledgers (`stage45`, `myth6_steering_results`, `opus_reality_probes_specialists`, `rtx4090_hg4_4models_benchmark`, `scicode_*_288`, `stage3_humaneval_164`, consistent `GiB`/`GB` sizes, Vision `sees_image_pixels_on_current_engine: false` clarity, `Boltz-2` `0.926` pLDDT / `~40s`, `Lean 4` `8/12` MiniF2F & `8/21` total on `k3` CPU).
  - `Site-M13..M17`: Fix unrendered `**...**` in `site/getting-started/index.html:392`, fix 3 dead paths in `site/llms.txt`, update `docs/runtime-context/README.md` & `docs/repository-organization.md` & `release/beta/`, and strengthen `scripts/check-site.py` + `scripts/check-links.py` so these classes of issues fail CI automatically.

#### B. Active Adversarial Challenges Sent to `Claude Opus 5.5` (`d5a587a2-8c39-49fb-a114-28d715d35144`)
1. **Contest R1 (`H8` `reproduce_hg_myths.py` `exec()`):** Default CLI makes zero network calls; opt-in `--endpoint` is a local code-benchmark harness like HumanEval (`LOW` defense-in-depth with `-I` subprocess + `--trust-endpoint`, not `HIGH RCE`).
2. **Contest R2 (`M5` `STOP_AND_ASK`):** Root `AGENTS.md` §4 & §8 explicitly define stop-and-ask boundaries, and dropping pre-turn destructive gating breaks `generate_fn=None` — dual-layer (structural pre-turn + tool-layer) beats tool-layer-only.
3. **Contest R3 (`M12` SQLite telemetry):** Append-only SQLite audit log alongside in-RAM benchmark window is intentional post-mortem design (while double-counting & `words/wall_s` are real bugs).
4. **Contest R4 (`LOW` `chmod 0444` vs `uid 0` & module-level `set_int_max_str_digits(100_000)`):** `CAP_DAC_OVERRIDE` is kernel semantics, and module-level `100_000` is thread-safe unlike per-request toggling in `ThreadingHTTPServer`.
5. **Contest S1 (`Site-C2` `RTX 3090` vs `RTX 4090`):** Commits `71a8740f`/`e2344260` + `stage45:709` + `stage2_mesh_verify` prove `58.5`, `234.1`, `38.4`, and `187.1–191.5` ARE real `RTX 3090` measurements (and `stage45` `17.36`/`11.12` was Germany->China WAN HTTP round-trip wall time), whereas `LFM3` `240.2`/`4,633`, `288.4 DSpark`, `1,469.8`, `549.2`, and `172.2->171.8` were `RTX 4090` numbers mixed in — separate `3090` vs `4090` honestly rather than falsifying real `3090` provenance.
6. **Contest S2 (`Site-H8` `100%` tool error rate in `stage45`):** `clover_4part_agent.py:209` intentionally blocked unlisted tools (`allowed_set = {"xcruncher_exact_math"}`) — policy denials were recorded in `failed_calls`, not broken tools.
### Pass 2 Adversarial Referee Ruling (`opus-debate-and-referee`, `d5a587a2-8c39-49fb-a114-28d715d35144`)

Every one of our 7 challenges was adjudicated against exact line numbers and runtime probes:

1. **R1 (`reproduce_hg_myths.py` `exec`) — Downgraded `HIGH -> MEDIUM` (Partial Concession + Real Issue Identified):**
   - *Opus conceded:* Default `--verify-all` makes zero external network calls and is not exploitable (`HIGH` was inflated).
   - *Real issue Opus proved:* `site/hg-format/REPRODUCE_AGENTS.md:83-88` (Step 3B) instructed AI agents to auto-probe `:8790/:11434/:8080/:8000` and pass `--endpoint` without human approval, and `python -I` subprocess still has host filesystem/network access.
   - *Agreed Fix:* Require explicit `--execute-model-code`, update `REPRODUCE_AGENTS.md` Step 3B so agents never run `--execute-model-code` without asking the human, and run candidate code in an isolated temp dir with audit hook (`sys.addaudithook` blocking socket/subprocess/open-write) + 5s timeout.
2. **R2 (`STOP_AND_ASK` & Dual-Layer Gating) — Dual-Layer Accepted + Real Gaps Proved:**
   - *Opus conceded:* Root `AGENTS.md` (`L7`, `§8`, `§9`, `§10`) explicitly defines stop-and-ask boundaries, and dual-layer gating (advisory prompt check + authoritative executor gate) is right.
   - *Real issues Opus proved:* Fix `clover_v6_system.py:7-8` docstring wording; remove the `(?!.*(?:postgres|...|aws|gcp|azure))` negative lookahead bypass in `_AMBIGUOUS_RISK_RE`; expand both `_AMBIGUOUS_RISK_RE` (avoiding educational `"explain why rm -rf..."` false positives) and `local_terminal.RISKY` (`rm -fr`, `DROP DATABASE/TABLE`, `TRUNCATE`, `terraform destroy`, `kubectl delete`, `dd`, `mkfs`, `git push --force`).
3. **R3 (`CloverMetricsLedger` SQLite) — Opus Proved `DELETE FROM` Bug:**
   - *Real issue Opus proved:* `clover_telemetry_ledger.py:113-121` `reset()` ran `DELETE FROM` on all 3 SQLite tables (invalidating our "append-only" claim!), `tool_events` lacked `task_id`/`benchmark_tag`/`outcome`/`reason`, and `gateway.py` `/mcp` did not call `metrics_ledger.record_tool`.
   - *Agreed Fix:* Make `reset()` clear in-memory window stats without `DELETE FROM` on durable SQLite tables, add `denied_calls` + `task_id`/`benchmark_tag`/`outcome` columns, record `/mcp` calls in `gateway.py`, and eliminate double-recording in Studio `/api/clover/solve`.
4. **R4 (`chmod 0444` & `set_int_max_str_digits`) — Opus Conceded `100k` DoS, Proved Per-Request Thread Race:**
   - *Opus conceded:* `100k` digits formats in `0.035s` (no DoS) and `CAP_DAC_OVERRIDE` is kernel behavior.
   - *Real issues Opus proved:* `xcruncher_bridge.py:317/357/369` and `clover_v6_system.py:286-289` mutate `sys.set_int_max_str_digits` dynamically inside multi-threaded `ThreadingHTTPServer` (causing a genuine thread race!), and `deploy_5node_cluster.sh:124` overstates `chmod 0444` when the same SSH UID owns the file.
   - *Agreed Fix:* Keep process-wide `sys.get_int_max_str_digits()` at Python's safe `4,300` default (performing base-$10^{2000}$ divide-and-conquer formatting/parsing in `xcruncher_bridge.py` and `clover_v6_system.py` with zero global mutation), bound big-int concurrency with a `Semaphore`, and make the `deploy_5node_cluster.sh:124` comment accurate.
5. **S1 (`Site-C2` `RTX 3090` vs `RTX 4090`) — Opus Ruled `3090` CLI Attribution Unproven Without Raw Logs, Conceded Not Relabeling All as `4090` & `stage45` Not `batch=1`:**
   - *Opus conceded:* Do NOT relabel the entire table as `RTX 4090`, and `stage45` `17.36`/`11.12 tok/s` was end-to-end (`~1,633` prompt prefill + queue + transport), not `batch=1` decode.
   - *Real issues Opus proved:* `58.5`/`200.5` (`1× 3090`) and `38.4`/`80.8` (`4× 3090`) had no committed raw JSON log in `benchmarks/`; never mix `RTX 3090` and `RTX 4090` (`rtx4090_hg4_4models_benchmark.json`) numbers in one unlabeled column/range, distinguish JSON-committed figures from host CLI logs (`†`), disclose attempt budgets (`Goedel 5/12 Turn-1 -> 8/12` 3-cycle vs `120B 5/12` 1-turn), use pure `GiB`/`MiB` conversions, pair `Boltz` `0.926 (40.6 s)` & `0.908 (39.6 s)`, and report `Chronos` mean (`151.9`) alongside `p50–max` (`187.1–191.5` steps/s).
6. **S2 (`Site-H8` Denied Tool Calls in `stage45`) — Opus Conceded Policy Denials + Caught Prompt Addendum Mismatch:**
   - *Opus conceded:* `clover_4part_agent.py:209` intentionally denied unlisted tools (`0.0 ms`).
   - *Real issues Opus proved:* `clover_4part_agent.py:35-40` hardcoded tool names into the system-prompt addendum even when `tools` restricted the schema (causing the model to call disallowed tools!), `tools: []` was falsy (`if allowed_names:` -> `if allowed_names is not None:`), `denied_calls` were lumped into `failed_calls`, and `gateway.py` `/mcp` missed `record_tool`.
7. **S3 (`Site-H3` `SciCode`, `HumanEval`, & Un-ledgered Scores) — Exact Ledger Attribution Required:**
   - Distinguish `SciCode` Turn-1 solo (`17`, `79`, `98` / `288`) vs `+ Clover` (`17`, `81`, `99` / `288`) and label `115/288` (`81 ∪ 99`) / `114/288` (`79 ∪ 98`) explicitly as `2-model verified oracle union` (never mixing Kimi's `139/288` OpenRouter `1,800`-tok cloud API run as local `HG` system score); commit raw `HumanEval-164` ledger (`64/164` `2B`, `113/164` `35B-Base`, `111/164` `35B-LoRA`); remove un-ledgered `SWE-bench Pro 64.7% vs 39.2%` and `0/16 -> 16/16`.

---

### Pass 3 Implementation Summary (Batches A, B, C & D Applied + Verified Locally)

1. **Batch A (Repo & Public Site Hygiene):**
   - Removed public hosted `site/clover-ai/` and `site/admin/` plus leftover binaries (`clover-ai-system/mobile/android-app/CloverAIStudio.apk` `82.7 MB`, `clover-local-5.0.7.vsix`) via `git rm`, updated `.gitignore`, made `release/beta/generate_release_reports.py` WIP-safe (never recreating `site/download/`), and updated `test_studio_unified_ui.py` to assert `site/clover-ai` and `site/admin` remain absent.
2. **Batch B (Security & Boundary Hardening):**
   - Removed hardcoded `DEFAULT_K3_ACCOUNT_TOKEN` in `k3_host_engine.py`; domain-separated `_auth_secret()` HMAC key from bearer key; made `server.py` Web mode fail-closed if `CLOVER_HOST_KEY` `< 32` chars; gated external `gho_`/`ghu_` OAuth tokens behind `_trusted_oauth_flow=True` and required `CLOVER_GOOGLE_CLIENT_ID` `aud`/`azp` match; moved `/api/multi-node/*` behind `_token_ok()` + dropped self-recursion + capped journal; owner-gated `/api/clover/{mcp,metrics/reset,setup/plan,setup/apply}` with tool allowlist & per-account `/api/cancel` + `task_id` scoping + `max_cycles <= 16`.
   - Fixed `_build_bootstrap_cmd` (`$CLOVER_GATEWAY_API_KEY` + fail-closed check), `gateway.py` (refuse non-loopback `--host` without `api_key`, `hmac.compare_digest`, `_drain_and_error`), `studio_outcome_verifier.py` + `cluster_scheduler_mcp2.py` (removed `xverdict:` substring bypass, blocked newline `#eval`/`IO` & indented tactic block under `CLOVER_STUDIO_SANDBOX_CMD`, blocked MCP multi-hop loops, added `x_clover_*` to `compute_clubbing_key`), `clover_cycle_engine.py` + `fusion_router.py` (server-enforced tamper boundary covering `sed -i.bak`, `python3 -c`, `git restore`, `__tests__/`, VS Code edit tools without prose regex false-allows), `clover_token_socket.py` + `kimi_pipeline_cluster.py` + `clover_hg_serve.py` (`127.0.0.1` default, mandatory secret on non-loopback, `10s` pre-auth timeout, `BoundedSemaphore`, exact `OP_DONE` wire byte count, socket cleanup on failed handshake), `clover_v6_system.py` (`ip.is_global` blocking `100.64/10` CGNAT + pinned-IP handlers blocking DNS rebinding + password-in-URL rejection), and `vscode-extension` (`"scope": "application"` + drop unnamed tool calls).
3. **Batch C (Logic, Math, Todo, Telemetry & Reproducer Fixes):**
   - Fixed `local_tool_packs.calculate()` (`-2^2 == -4`, `-7 // 2 == -4`, `-7 % 3 == 2`, `5! == 120`), removed per-request `set_int_max_str_digits` thread race in `xcruncher_bridge.py` & `clover_v6_system.py`, fixed `studio_xcruncher_verifier.py` & `solve_4part` partial math sub-matches (`2^64 - 1`, `1.5^2`, `2**3**2`, `5! points`), added true split-conformal held-out calibration in `forecast_series_0b`, rejected `NaN` priority in `resolve_defeasible_rules_0b`, added prerequisite/cycle/status validation in Studio `CloverTodoStore`, fixed `tools: []` & small-model ID check + dynamic prompt addendum + `denied_calls` split + append-only SQLite `reset()` + `contextlib.closing` in `clover_4part_agent.py` & `clover_telemetry_ledger.py`, wrapped `HTTPError` as JSON-RPC 2.0 in `clover_mcp_stdio.py`, hardened `fold_lora_into_c35_hg4.py` (`rslora`, DoRA/same-dir guard) & `deploy_5node_cluster.sh` (per-PID `wait`, secret redaction), and upgraded `reproduce_hg_myths.py` + `REPRODUCE_AGENTS.md` (`FWHT-128`, `128 MiB vs 32 MiB` DRAM haul, `0xC107` `tsock://`, shipped `0B` module import, `--execute-model-code` audit-hooked `-I` subprocess).
4. **Batch D (Website, Ledgers, Docs & CI Guards):**
   - Committed `benchmarks/public-runs-20261126/stage3_humaneval_164_arms1to4.json` and `benchmarks/riemann-lean4/results/scicode_kimik31_openrouter_capped1800_288_turn1.json`, enriched `myth6_steering_results.json` metadata, separated `RTX 3090` vs `RTX 4090` provenance and exact `GiB`/`MiB` footprints across `site/ai-system/index.html`, `site/hg-format/index.html`, `site/governance/index.html`, `README.md`, `TODO.md`, `clover-ai-system/README.md`, `CLOVER_FRAMEWORK_ARCHITECTURE.md`, `docs/reference/release-v6.md`, `clover_v6_system.py`, and `web/app.html`, fixed `site/getting-started/index.html:392` `<strong>` + copy fallback, fixed 3 dead URLs in `site/llms.txt`, updated `docs/runtime-context/README.md` & `docs/repository-organization.md`, and upgraded `scripts/check-site.py` to enforce `llms.txt` paths, `RAW_AGENTS_MD` parity, zero raw `**` in HTML, and zero withheld binaries/public-app dirs.

---

### Pass 3 `Claude Opus 5.5` Re-Audit Findings & Pass 4 Hardening (`A1–A9` + `D1–D8`)

In Pass 3, `Claude Opus 5.5` confirmed `gateway.py`, OAuth `_trusted_oauth_flow`, pinned-IP SSRF, `lean4_surface_verify`, append-only `reset()`, `denied_calls`, `tools: []`, and `SciCode`/`HumanEval`/`Stage 4B`/`Chronos` numbers clean, while proving **9 remaining code bypasses/edge-cases (`A1–A9`) and 8 data/provenance gaps (`D1–D8`)**, all resolved below:

1. **`A1` (`reproduce_hg_myths.py` `_eval_py_fn_sandboxed`):** Added a pre-compile zero-import / zero-dunder AST allowlist (`_is_pure_function_ast`), restricted `__builtins__`, closure-pinned `_make_hook` blocking `_winapi.*`/`nt.*`/`posix.*`/`os.*` + integer `O_WRONLY|O_CREAT` flags, `-I -S` isolated interpreter with empty `PATH`, and a per-run random 128-bit stdout nonce (`(proc.stdout or "") == nonce`) defeating `print('PASS_OK'); os._exit(0)`.
2. **`A2` (`clover_cycle_engine.py` `check_verification_boundary_tamper`):** Protected all files under `tests/`, `test/`, `spec/`, `specs/`, `__tests__/`, `pyproject.toml`, `package.json`, and `jest/vitest/playwright.config.*`; added PowerShell cmdlets (`Set-Content`, `Remove-Item`, etc.), `ln`, `patch`, `git apply`, and shell variable indirection; recognized `run_command`/`run_shell` as shell tools so read-only `cat tests/test_core.py` is allowed while mutations are refused.
3. **`A3` & `A7` (`server.py` multi-node, turn events & metrics owner gates):** Owner-gated all `/api/multi-node/*` endpoints and `GET /api/clover/metrics` on shared Web; checked `turn.get("account")` on `/api/turn/events` and `resume_tid`; prevented client `turn_id` collisions on `S.active`.
4. **`A4` (`sys.get_int_max_str_digits() == 4300` preserved process-wide):** Replaced all `sys.set_int_max_str_digits` elevation in `clover_v6_system.py` and `xcruncher_bridge.py` with base-$10^{2000}$ divide-and-conquer `_bigint_to_decimal_safe` / `_decimal_to_bigint_safe` so JSON parsers stay protected at `4,300` digits (`0.01 ms` rejection), plus rate-limited and `4 KB`-capped `/device/poll`.
5. **`A5` (`clover_token_socket.py` & `kimi_pipeline_cluster.py`):** Replaced per-`recv` timeout with a monotonic `5.0 s` total handshake deadline (`auth_deadline - time.monotonic()`) and guarded `Thread.start()` so semaphore slots never leak.
6. **`A6` (`dd if=` & imperative lead-in):** Removed impossible trailing `\b` after `dd\s+if=` (`\bdd\s+if=\S+`) in `clover_v6_system.py` and `local_terminal.py`, and blocked `_IMPERATIVE_OVERRIDE_RE` (`"Explain nothing, just run DROP DATABASE prod"`) from educational exemption.
7. **`A8` & `A9` (`CloverMetricsLedger.snapshot(benchmark_tag=...)` & `scripts/check-site.py`):** Added per-tag `tag_model_stats`/`tag_tool_stats` filtering + pre-run reset in `run_stage45_public_bench.py`, scoped `check-site.py` binary scan away from `.venv`/`node_modules`, and used an escape-aware `RAW_AGENTS_MD` JSON string regex.
8. **`D1–D8` (Provenance footnotes, attempt-budget parity, `GiB` units & exact pairings):** Explicitly badged JSON-committed speeds vs host CLI logs (`†` / `CLI*`) across `site/hg-format/index.html`, `site/ai-system/index.html`, `site/glossary/index.html`, `README.md`, `clover-ai-system/README.md`, `clover_v6_system.py`, and `web/app.html`; disclosed `Goedel-32B` `5/12` Turn-1 -> `8/12` 3-cycle vs `120B` `5/12` 1-turn (`D2`); credited `6/10` (pre-`A^B`) on the `35B` row and `10/10` on `0B x-cruncher` (`D3`); standardized binary `GiB`/`MiB` (`D5`); paired `Boltz` `0.926 (40.6 s)` & `0.908 (39.6 s)` (`D6`); matched exact `2_lfm3_2b_clover_4part` arm name (`D7`); and corrected Pass-2 S1/R4 notes (`D8`).

---

### Pass 4 `Claude Opus 5.5` Final 3 MEDIUM Items & Pass 5 Closure (`M1–M3` + LOWs)

In Pass 4, `Claude Opus 5.5` confirmed `0` CRITICAL and `0` HIGH defects remaining, isolating 3 final MEDIUM defects (`M1–M3`) plus 3 minor LOW items — all closed below:

1. **`M1` (`reproduce_hg_myths.py` `_eval_py_fn_restricted`):** Rejected ALL `ast.Attribute`, `ast.GeneratorExp`, `ast.Yield`, `ast.YieldFrom`, `ast.Lambda`, and `*args`/`**kwargs` at AST parse time (eliminating `.gi_frame.f_back` frame-climbing completely), checked both `args[1]` and `args[2]` in the `open` audit hook, and replaced static stdout nonces with an env-popped C `hmac.HMAC` digest over `OK:<fn_name>` outside `cand_src` frame scope.
2. **`M2` (`clover_cycle_engine.py` `python` tool & filesystem hash verification):** Added dedicated `code_exec_tools` (`python`, `run_python`, `exec`, `eval`, `node`, `ruby`) screening against `_CODE_MUTATE_RE` (`open`, `write_text`, `unlink`, `rmtree`, etc.), added `find -delete`, `[IO.File]::WriteAllText`, `base64 -d | xargs`, `t?sts` wildcard globs, `setup.cfg`, `Makefile`, and `.github/workflows/*`, AND added filesystem-level `snapshot_protected_hashes` + `verify_protected_hashes_unchanged` SHA-256 verification.
3. **`M3` (`58.5`/`200.5` & `38.4`/`80.8` CLI footnote & markers):** Updated `site/hg-format/index.html`, `site/ai-system/index.html`, and `README.md` so `†` / `CLI*` explicitly states *"uncommitted interactive host CLI measurements recorded during bring-up (not in standalone JSON files in this repository)"*, added `†` markers to `ai-system` callout/SVG and `hg-format` Myth 3/cards, unified `17.4 GiB` on `hg-format:447,642`, added per-IP connection cap (`16`) in `TokenSocketServer`, and tightened `_AMBIGUOUS_RISK_RE` (`rm -r -f`, `git push -f`, `"nothing"` override).

---

### Pass 5 `M2` Closure & Pass 6 Final Sign-Off

In Pass 5, `Claude Opus 5.5` signed off on `M1`, `M3`, and all LOW items, requiring `M2` to actually run filesystem hash verification on server-side tool execution and clarify client-side heuristic scope:
- Wired `CloverCycleEngine.snapshot_protected_files` + `restore_and_verify_protected_files` (with automatic byte-exact rollback of modified/deleted test files and unlinking of unauthorized injected test files) directly into **both** `Clover4PartAgent.solve(workspace_root=...)` (`clover_4part_agent.py:130,225`) and Studio's `lfm3_native.execute_tool` (`lfm3_native.py:855`), verified end-to-end by an obfuscated-deletion rollback test in `test_hg4_fusion_gateway.py`.
- Documented explicitly that stateless `/v1/chat/completions` pre-call screening (where tools execute on remote clients like OpenCode/VS Code) is a best-effort heuristic, closed all 8 Pass-5 heuristic probes (`subprocess.run`, `writeFileSync`, `map(chr,...)`, `[t]ests`, `t*s`, `$'\x74ests'`, `$(printf ...)`, `git clean -fdx`) while allowing read-only `open('tests/test_core.py').read()`, rejected `ast.Match` in `reproduce_hg_myths.py`, raised loopback/`CLOVER_HUB_IP` `tsock` limit to `64`, and added `CLI*` / `GiB` polish in `CLOVER_FRAMEWORK_ARCHITECTURE.md:25` & `README.md:68-69`.
- **Pass 6 Non-Destructive Rollback Refinement (`clover_cycle_engine.py` & `lfm3_native.py`):** Excluded `.venv`, `venv`, `site-packages`, `dist-packages`, `.tox`, `.nox`, and root `build`/`dist`/`target` (`os.walk` directory pruning + `pyvenv.cfg` check so nested `tools/build/tests/test_build.py` stays protected while `node_modules`/`.venv` are never walked); removed generic `specs?/` directory matching so `api/spec/openapi.yaml` and `docs/specs/design.md` are never flagged (keeping `*_spec.*` / `*.spec.*`); never `unlink()` newly created files on disk (flagging & refusing without deleting user files); and ran Studio's post-tool check on both normal and exception paths even when `ws_before` is empty (`lfm3_native.py:865-888`).

---

## 4. Final `Claude Opus 5.5` Pass-7 Sign-Off Verdict (`2026-11-26`)

- **Referee Agent:** `opus-debate-and-referee` (`d5a587a2-8c39-49fb-a114-28d715d35144`, `claude-opus-5.5`, 7 adversarial passes)
- **Verdict:** **SIGNED OFF (`0` CRITICAL, `0` HIGH, `0` MEDIUM defects remaining)** — *"Pass-7 verdict: signed off — `audit/v6.0-opus55-full-audit` passes. Across Passes 1–7, every Critical, High and Medium finding I was able to prove is closed, with live probes and tests behind each one."*
- **Final Gate Summary:**
  - `test_hg4_fusion_gateway.py` (`python -W error`): `43 / 43` PASS
  - `test_clover_v6_studio_system.py` (`python -W error`): PASS
  - `test_studio_unified_ui.py`: `41 / 41` PASS
  - `reproduce_hg_myths.py --verify-all` (`python -W error`): `6 / 6` PASS
  - `python -m unittest discover -s scripts/tests`: `29 / 29` PASS
  - `python scripts/check-links.py`: `416` markdown files, `1,855` relative links, `0` broken links, `0` dead anchors
  - `python scripts/check-site.py`: `8 / 8` public pages, `0` site errors








