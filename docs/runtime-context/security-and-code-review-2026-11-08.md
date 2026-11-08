# Clover Framework — Comprehensive Code Review, Hardcoding & Security Audit (2026-11-08)

## 1. Context

This document records the complete step-by-step review and security audit of all 72 files (`+11,832 / -392` lines across commits `5f72b9fc` through `f7f6396c`) added or modified in the `clover-framework` repository across five major workstreams:
1. **Multi-Cycle `AGENTS.md` Clover Cycles & Official Yellow Clover Flower Branding** (`clover_cycle.py`, `omni_native.py`, `lfm3_native.py`, `autonomous_clover.py`, `build_msix.ps1`, Android launcher icons).
2. **Hugging Face Precomputed Dataset Upload (`Santhoshsathish94/clover-ai-studio`) & Parallel Range Installer** (`porting-models/upload_computed_to_hf.py`, `setup_fetch.py`, `launch/manifest*.json`).
3. **Open-Weight Model Port: `Diffuspace/Clover-35B-A3B-Preview` (`35B-A3B` Block Diffusion MoE + BD-GDN + Gated GQA)** (`porting-models/clover_35b_gpu_backbone.py`, `porting-models/train_clover_35b_hybrid_ncm_gpu.py`, `clover-ai-system/src/models/clover-35b-a3b/`, `k3_host_engine.py`).
4. **Full Two-Way VS Code & GitHub Copilot Integration** (`server.py`, `lfm3_native.py`, `omni_native.py`, `vscode-extension/extension.js`, `test_vscode_integration.py`, `docs/runtime-context/engine-and-architecture/clover-vscode-studio-integration-context.md`).
5. **Multi-User `k3` Allowlist (`allowed-accounts.json`), GitHub-Authenticated Admin Console (`/admin/`), & Mobile `k3` Model Selection/Reporting** (`server.py`, `manage_k3_users.py`, `k3_host_engine.py`, `omni_c2_c3.py`, `lfm3_layers.py`, `web/admin.html`, `site/admin/index.html`, `web/app.html`).

---

## 2. Direction (Audit Scope & Governing Boundaries)

Per `AGENTS.md` (Sections 4, 8, 9, and 11), the audit evaluated every changed module against six governing boundaries:
1. **Authentication & Authorization Integrity**: No endpoint may grant access to `k3` models, local Studio state, or admin allowlist operations without cryptographic proof of identity (`acct:<user>:<hmac>`, `admin:<owner>:<hmac>`, per-user `key_sha256`, verified GitHub OAuth token, or verified Google OIDC/OAuth token).
2. **Zero Hardcoded Secrets & Safe Secret Fallbacks**: No API keys, tokens, or static HMAC signing secrets may appear in source code or git history.
3. **Elimination of Hardcoded Constants / Paths**: Owner identifiers, OAuth client IDs, dataset repositories, and server directories must be configurable via environment variables with safe defaults.
4. **Filesystem & Process Safety**: No predictable or world-writable `/tmp` files, no shell command interpolation (`execSync` with unsanitized strings), and restrictive `0o600` permissions on sensitive configuration/vault files.
5. **Brute-Force & Cross-Site Request Protection**: Strict `Host`, `Origin`, `Sec-Fetch-Site`, CSP nonce, and per-IP rate limiting on authentication endpoints.
6. **Model Attribution & Epistemic Honesty**: The UI, health endpoints, and `C1 Inference` layer telemetry must report the actual model executing the turn (`Clover-35B-A3B`, `Nemotron 3.5`, `Kimi K3.1 Pro`, `LFM3-Omni (k3)`, or local `LFM3-Omni-2B`) rather than hardcoded labels.

---

## 3. Findings & Execution (Remediations Applied)

| ID | Severity | Location | Finding | Remediation Applied | Status |
|---|---|---|---|---|---|
| **F-01** | **HIGH** | `clover-ai-system/studio/clover/server.py` (`verify_k3_credential`) | **Passwordless Invited-Email Bypass (Step 2b)**: An invited email in `allowed-accounts.json` without a custom `key_sha256` could authenticate to `/api/auth/login` by passing its own email string or `"k3"` as the key, allowing anyone who knew an approved email address to mint a signed `acct:<email>:<hmac>` token without proving ownership. | Removed Step `2b` completely. Every approved user must present a cryptographic credential (`acct:<user>:<hmac>` key from the Admin Console, `key_sha256`, verified GitHub token, or verified Google OIDC/OAuth token). | **Fixed & Verified** |
| **F-02** | **HIGH** | `clover-ai-system/studio/clover/server.py` (`_token_ok` & `/api/auth/k3-unlock`) | **Local Launch-Token Bypass on Mobile/Desktop**: `/api/auth/k3-unlock` was placed before `self._token_ok()`, and `_token_ok()` accepted `vault("rm_k3")` as a local launch token and only inspected the first non-empty auth header (`or` chain). | Moved `/api/auth/k3-unlock` after `if not self._token_ok():`, removed the `saved_k3` backdoor from `_token_ok()`, and updated `_token_ok()` to evaluate all candidate headers/cookies (`Cookie: clover_token`, `X-Clover-Token`, `X-Clover-Key`, `Bearer`) against `S.token` on local/Android and `verify_k3_credential` on `WEB`. | **Fixed & Verified** |
| **F-03** | **HIGH** | `clover-ai-system/vscode-extension/extension.js` & `laptop/vscode-extension/extension.js` | **Legacy Auto-Verify Harness with Shell Interpolation & Predictable `/tmp` Files**: `clover-ai-system/vscode-extension/extension.js` contained a legacy `clover.verifyLiveAgentLoop` command that auto-triggered if `/tmp/clover_vscode_autoverify_trigger` existed and ran `cp.execSync` with string interpolation. Additionally, `ACTIONS.diff` in the laptop extension wrote proposed diff files to a predictable path in `os.tmpdir()`. | Removed `clover.verifyLiveAgentLoop` and `/tmp/clover_vscode_autoverify_trigger` from `clover-ai-system/vscode-extension/extension.js`. Hardened `ACTIONS.diff` to create `0o700` temp directories and randomized `0o600` diff files with sanitized basenames. | **Fixed & Verified** |
| **F-04** | **MEDIUM** | `clover-ai-system/studio/clover/manage_k3_users.py` (`_secret`, `save_allowlist`) | **Static Fallback HMAC Secret & Permissive Allowlist File Mode**: `_secret()` fell back to a static literal `b"clover-k3-fallback-secret"` when `vault/k3_host.key` was absent, and `save_allowlist()` wrote `allowed-accounts.json` without `0o600` permissions. | Updated `_secret(base)` to generate and persist a cryptographically random 256-bit key (`secrets.token_hex(32)`) in `base / "vault" / "k3_host.key"` with `0o600` permissions, and enforced `0o600` permissions on `allowed-accounts.json`. | **Fixed & Verified** |
| **F-05** | **MEDIUM** | `clover-ai-system/studio/clover/server.py` (`_serve_hf_mirror`) | **World-Writable `/tmp` Cache Path on Linux**: `_serve_hf_mirror` cached the Hugging Face mirror tree at `/tmp/clover_hf_mirror_tree_v7.json` on POSIX systems. | Moved the cache file into `Path(S.args.base) / "cache" / "hf_mirror_tree_v7.json"` across all operating systems. | **Fixed & Verified** |
| **F-06** | **MEDIUM** | `clover-ai-system/studio/clover/server.py` (`/api/auth/login`, `/api/admin/login`, `/api/admin/github/device/start`) | **Missing Brute-Force Rate Limiting & Input Validation**: Auth endpoints lacked per-IP rate limiting, and `manage_k3_users.cmd_add` lacked strict regex validation for email addresses and GitHub usernames. | Added thread-safe sliding-window rate limiter `_check_auth_rate_limit(client_ip)` (`15` attempts / `60s` per IP, returning `HTTP 429`) and strict `EMAIL_RE` / `GITHUB_RE` validation in `manage_k3_users.cmd_add`. | **Fixed & Verified** |
| **F-07** | **LOW** | `server.py`, `manage_k3_users.py`, `k3_host_engine.py`, `omni_native.py`, `omni_c2_c3.py`, `lfm3_layers.py`, `web/app.html`, `porting-models/*.py` | **Hardcoded Identifiers, Paths, and `"LFM3-Omni-2B"` Telemetry Labels**: Owner accounts, OAuth client IDs, and porting paths were hardcoded literals; `c1_inference` events and UI health caches hardcoded `"LFM3-Omni-2B"` even when running `Clover-35B-A3B`, `Nemotron 3.5`, or `Kimi K3.1 Pro` on `k3`. | Parameterized owner/client/path constants via environment variables (`CLOVER_OWNER_GITHUB`, `CLOVER_OWNER_EMAIL`, `CLOVER_ALLOWED_ACCOUNTS`, `CLOVER_GITHUB_CLIENT_ID`, `CLOVER_WEB_BASE`, `CLOVER_HF_DATASET_REPO`, `CLOVER_PORT_WORK_DIR`), and wired dynamic model names & layer descriptions through `K3HostEngine`, `RemoteOmniEngine`, `OmniNative`, `OmniC2C3Runtime`, `lfm3_layers`, and `web/app.html`. | **Fixed & Verified** |

---

## 4. Module-by-Module Review Summary

### 4.1 Authentication, Admin Console & Multi-User `k3` Allowlist
- **Files**: [clover-ai-system/studio/clover/server.py](../../clover-ai-system/studio/clover/server.py), [clover-ai-system/studio/clover/manage_k3_users.py](../../clover-ai-system/studio/clover/manage_k3_users.py), [clover-ai-system/studio/clover/web/admin.html](../../clover-ai-system/studio/clover/web/admin.html), [site/admin/index.html](../../site/admin/index.html).
- **Architecture**:
  - Admin authentication uses GitHub OAuth Device Flow (`/api/admin/github/device/start` & `/api/admin/github/device/poll`) or a GitHub Personal Access Token verified live against `https://api.github.com/user` and restricted to `OWNER_GITHUB_LOGIN` (`@santhoshsathish94`).
  - Admin tokens (`admin:<login>:<hmac>`) are domain-separated from user account tokens (`acct:<account>:<hmac>`) in the HMAC payload (`clover-admin:` vs `clover-acct:`), preventing privilege escalation from a user token to admin endpoints.
  - Removing a user from `allowed-accounts.json` immediately invalidates their `acct:<account>:<hmac>` token on the next request because `verify_account_token()` checks membership in `allowed_accounts()` before verifying the HMAC.

### 4.2 Multi-Cycle `AGENTS.md` Clover Cycles & Outcome Verification
- **Files**: [clover-ai-system/studio/clover/clover_cycle.py](../../clover-ai-system/studio/clover/clover_cycle.py), [clover-ai-system/studio/clover/omni_native.py](../../clover-ai-system/studio/clover/omni_native.py), [clover-ai-system/studio/clover/lfm3_native.py](../../clover-ai-system/studio/clover/lfm3_native.py), [clover-ai-system/studio/clover/autonomous_clover.py](../../clover-ai-system/studio/clover/autonomous_clover.py).
- **Architecture**:
  - Implements up to 3 autonomous `Context -> Direction -> Execution -> Outcome -> Growth` cycles per turn (`verify_turn_outcome`), checking AST/Generative Compiler validity, sandboxed Python execution for computational code blocks, exact arithmetic consistency, and unrecovered tool errors.
  - Suppresses numerical false positives inside `bf16` / `mat_mul` matrix-operation warnings in `omni_c2_c3.py`.

### 4.3 Two-Way VS Code & Copilot Integration
- **Files**: [clover-ai-system/studio/clover/server.py](../../clover-ai-system/studio/clover/server.py), [clover-ai-system/studio/clover/lfm3_native.py](../../clover-ai-system/studio/clover/lfm3_native.py), [clover-ai-system/studio/clover/omni_native.py](../../clover-ai-system/studio/clover/omni_native.py), [clover-ai-system/vscode-extension/extension.js](../../clover-ai-system/vscode-extension/extension.js).
- **Architecture**:
  - Preserves `<attachment id="Browser Pages">` (`[pageId] Title (url)`) in `workspace_info` while keeping trailing envelope-only messages from wiping multi-turn `tool_rounds`.
  - Dynamically activates VS Code Integrated Browser (`open_browser_page`, `read_page`, `click_element`, `type_in_page`, `navigate_page`, `screenshot_page`, `run_playwright_code`), Terminal (`run_in_terminal`, `get_terminal_output`, `send_to_terminal`, `kill_terminal`), and Code Intelligence (`vscode_listCodeUsages`, `vscode_renameSymbol`, `view_image`) tools.
  - `repair_vscode_browser_args` automatically injects missing `pageId` from shared browser pages and resolves `[ref=eN]` from `read_page` accessibility snapshots.

### 4.4 `Clover-35B-A3B` Port & Fast Hugging Face Installer
- **Files**: [porting-models/clover_35b_gpu_backbone.py](../../porting-models/clover_35b_gpu_backbone.py), [porting-models/train_clover_35b_hybrid_ncm_gpu.py](../../porting-models/train_clover_35b_hybrid_ncm_gpu.py), [porting-models/upload_computed_to_hf.py](../../porting-models/upload_computed_to_hf.py), [clover-ai-system/studio/clover/setup_fetch.py](../../clover-ai-system/studio/clover/setup_fetch.py).
- **Architecture**:
  - `upload_computed_to_hf.py` reads Hugging Face credentials strictly from `HF_TOKEN` or `--token-stdin` (zero hardcoded tokens).
  - `setup_fetch.py` verifies every downloaded file against its pinned publisher SHA-256 / Git blob SHA-1 before moving `.part` into place, and validates archive member paths against path traversal (`..` or leading `/` / `\`).

---

## 5. Outcome (Verification Evidence)

All automated verification suites were executed both locally on Windows and remotely on the production `k3` server (`Ubuntu 24.04`, `python3.12`), plus live installation on the Samsung Galaxy S23 (`SM_S911B`) via wireless debugging:

1. **Security & Admin Audit Suite ([clover-ai-system/studio/tests/test_security_and_admin_audit.py](../../clover-ai-system/studio/tests/test_security_and_admin_audit.py))**:
   - `passwordless_bypass_blocked: true`
   - `signed_acct_token_verified_and_revocable: true`
   - `admin_privilege_separation_enforced: true`
   - `input_validation_enforced: true`
   - `random_vault_secret_enforced: true`
   - `rate_limiter_enforced: true`
2. **VS Code & Copilot Integration Suite ([clover-ai-system/studio/tests/test_vscode_integration.py](../../clover-ai-system/studio/tests/test_vscode_integration.py))**:
   - `ALL VS CODE & COPILOT INTEGRATION TESTS PASSED` (Windows, `laptop`, and `k3`).
3. **Multi-Cycle Clover Verification Suite ([clover-ai-system/studio/clover/test_clover_cycles.py](../../clover-ai-system/studio/clover/test_clover_cycles.py))**:
   - `ALL CLOVER CYCLE TESTS PASSED` (`verify_turn_outcome`, `multi_cycle_turn_and_growth`, `autonomous_clover_multi_cycle`).
4. **Android Release Build & Signature Verification (`k3` & `SM_S911B`)**:
   - `clover-ai-studio-beta.apk` (`82,074,462` bytes, SHA-256 `1ac775387e16199750b7a4f6d5ca339c9172e26dd36dd429e3943f5a8c3abd28`) and `clover-ai-studio-beta.aab` (`80,997,771` bytes, SHA-256 `e64b71645fe9f2a8b558785258ed421f890860e30a38d6b096dd15a5ec1e0983`) verified with `apksigner` / `jarsigner` and installed on `SM_S911B` (`Success`).

---

## 6. Growth (Lessons Preserved for Future Cycles)

1. **Never Allow Identifier-as-Credential Shortcuts**: Convenience shortcuts that accept an allowlisted email address without a separate cryptographic token (`acct:...`, OAuth/OIDC token, or `key_sha256`) turn an allowlist into a passwordless enumeration surface. Always require a cryptographic proof or HMAC-signed token (`F-01`).
2. **Keep Local Per-Launch Tokens Separate from Remote Host Keys**: On Android and desktop, the per-launch `CLOVER_TOKEN` (`S.token`) protects the local loopback server from other local processes, whereas `vault("rm_k3")` authenticates outbound requests to `k3`. Evaluating all request headers/cookies against `S.token` avoids header shadowing when `X-Clover-Key` is also present (`F-02`).
3. **Propagate Model Identity Through Every Layer Event**: When adding new hosted models (`Clover-35B-A3B`, `Nemotron 3.5`, `Kimi K3.1 Pro`) that share a common C1/C2/C3 pipeline, never hardcode `"LFM3-Omni-2B"` in lower-level `c1_infer` or UI health caches (`F-07`).
