# Clover v5.0.0 — Five-Goal Operating Plan & Cycle Record (2026-11-08)

## Overview

This context document establishes the five goals requested by the human and tracks each goal through the 5-stage Clover operating cycle (`Context → Direction → Execution → Outcome → Growth`), per `AGENTS.md`:

1. **Goal 1 — Stabilize `v5.0.0`**: Resolve version inconsistencies across manifests, extensions, Android/Windows packaging, and store checklists; regenerate deterministic release artifacts; and complete the security review of the latest Context file & Task Todo List changes.
2. **Goal 2 — End-to-End Cycle Validation Across All Four Models**: Execute identical multi-step tasks across all four Clover models (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron 3.5 Super VL`, and `Kimi K3.1 Pro`) and verify all five stages (`1. Context`, `2. Direction`, `3. Execution`, `4. Outcome`, `5. Growth`), including multi-cycle Outcome retries, Task Todo List progression, and Stage 5 Growth persistence.
3. **Goal 3 — Meaning Engine Bridge (`Memory → Ontology → HybridAgent/JEPA → Action`)**: Connect the Memory Graph (`S2`), Symbolic Ontology Graph (`S2`), HybridAgent (`C2`), and JEPA (`C2`) so that concepts, entities, and learned corrections recalled from Memory automatically query linked Ontology rules/constraints, synthesize a 64-d semantic context cue (`semantic_cue_64`) and 8-d action capability prior (`bridge_action_prior_8`), and drive C2/JEPA tool selection and action execution (`S3`).
4. **Goal 4 — Collective Learning (`CPSL`) Installation-to-Installation Transfer Experiment**: Test whether knowledge and problem-solving strategies acquired by Installation A (and published through the OEWN closed-vocabulary privacy gate + Ed25519-signed release bundle) measurably improve Installation B on held-out tasks without leaking private entities or overriding personal preferences (`Personal > Collective > General`).
5. **Goal 5 — Independent Validation & Reproducible Ablation Report**: Run and publish a reproducible evaluation suite with full before/after metrics, component ablations (`Full System`, `-Memory->Ontology Bridge`, `-JEPA Action Prior`, `-Outcome Verification Retry`, `-Collective Transfer`), and honest reporting of failure modes and boundaries.

---

## Goal 1: Stabilize `v5.0.0` (Versions, Artifacts & Security Review)

### 1. Context
Inspection across the repository (`c:\personal\oss\clover-framework`) and the laptop build tree (`c:\personal\oss\scratch-agents\lfm3-hybridagent\laptop`) identified the following version drift and artifact state:
- `VERSION` and `server.py` (`STUDIO_VERSION`) are `"5.0.0"`, and `launch/manifest-android.json` is `"5.0.0-android"`.
- **Inconsistencies found**:
  1. `clover-ai-system/studio/clover/launch/manifest.json` has `"version": "0.4.2"`.
  2. `clover-ai-system/vscode-extension/package.json` (`"4.1.2"`), `extension.js` (`"4.1.2"` / `"4.2.0"`), and `README.md` (`"4.1.2"`, missing `clover-35b-a3b`).
  3. `laptop/vscode-extension/package.json` and `extension.js` have `"0.4.0"`.
  4. `clover-ai-system/studio/browser-extension/manifest.json` has `"version": "0.1.0"`.
  5. `clover-ai-system/mobile/android-app/build.gradle` has legacy `versionCode 6`, `versionName "2.1.0"`, `targetSdk 34` instead of `versionCode 14`, `versionName "5.0.0"`, `targetSdk 36`.
  6. `clover-ai-system/studio/studio/build_msix.ps1` has `AppxManifest.xml` `Version="0.5.0.0"` while `Setup.cs`, `Launcher.cs`, and `RELEASE-MANIFEST.json` declare `5.0.0.0`.
  7. `release/beta/store/google-play/{release-notes.md,submission-checklist.md}`, `release/beta/store/microsoft/{release-notes.md,submission-checklist.md}`, and `release/RELEASE-READINESS-v5.0.0.md` reference `0.5.0-beta.1` / `versionCode 9` / `versionCode 12`.
- **Latest changes requiring security review**:
  - Commit `c54df640` added `Workspaces.record_outcome_note`, `local_context_for_paths`, `Repo.save_todos`/`load_todos`, `OperatingCyclePlan.todos`, and the `manage_todo_list` tool across `workspaces.py`, `clover_repo.py`, `clover_cycle.py`, `lfm3_native.py`, `omni_native.py`, `k3_host_engine.py`, `fused_engine.py`, `server.py`, and `web/app.html`.

### 2. Direction
- Unify all `v5.0.0` version identifiers across Studio manifests, VS Code extensions (`clover-vscode-provider` and `clover-local`), browser extension, Android `build.gradle`, Windows `AppxManifest.xml` (`5.0.0.0`), and store checklists.
- Audit the `c54df640` Context file & Todo list changes against path traversal, prompt injection, vault read/write isolation, and XSS in `renderTodoListBox`.
- Rebuild the Windows `.vsix`, `CloverAIStudio-Setup.exe`, and `.msix`/`.msixupload` packages, sync the latest Android `5.0.0` (`versionCode 14`) `.apk` and `.aab` from `k3`, and run `release/beta/generate_release_reports.py` to regenerate all manifests, build reports, and `SHA256SUMS.txt`.

### 3. Execution, 4. Outcome & 5. Growth
- **Execution**:
  - Updated `clover-ai-system/studio/clover/launch/manifest.json` (`0.4.2` $\to$ `5.0.0`), `clover-ai-system/vscode-extension/{package.json,extension.js,README.md}` (`4.1.2`/`4.2.0` $\to$ `5.0.0` + added `clover-35b-a3b`), `laptop/vscode-extension/{package.json,extension.js}` (`0.4.0` $\to$ `5.0.0`), `clover-ai-system/studio/browser-extension/manifest.json` (`0.1.0` $\to$ `5.0.0`), `clover-ai-system/mobile/android-app/build.gradle` (`versionCode 14`, `versionName "5.0.0"`, `targetSdk 36`), and `clover-ai-system/studio/studio/build_msix.ps1` (`AppxManifest.xml` `Version="5.0.0.0"`, `SanthoshNarayanan.CloverAIStudio_5.0.0.0_x64.msix`).
  - Updated `build_assets.py` so `build_icons()` reads from `assets/clover-yellow-flower-icon-pack` and never overwrites the official 5-petal yellow Clover flower icon with legacy assets.
  - Hardened `workspaces.py` (`F-08` in [docs/runtime-context/security-and-code-review-2026-11-08.md](security-and-code-review-2026-11-08.md)): blocked `local_context_for_paths` on `paths.WEB`, blocked `vault/` path segments, and wrapped all `CONTEXT.md`, `TODO.md`, and `*-context.md` reads in `semantics.flag_injection(...)`.
  - Rebuilt `.vsix` (`clover-local-5.0.0.vsix`), `CloverAIStudio-Setup.exe`, `SanthoshNarayanan.CloverAIStudio_5.0.0.0_x64.msix`, `clover-ai-studio-beta.msixupload`, and Android `5.0.0` (`versionCode 14`) `.apk` & `.aab`, and ran `release/beta/generate_release_reports.py`.
- **Outcome**:
  - `python clover-ai-system/studio/tests/test_security_and_admin_audit.py`: **10/10 security checks passed** (`context_and_todo_security_enforced: true`).
  - Deterministic artifact hashes verified across `release/beta/RELEASE-MANIFEST.json`, `download/SHA256SUMS.txt`, and store submission checklists:
    - Windows Setup (`download/laptop/CloverAIStudio-Setup.exe`): `9,678,296` bytes (`SHA-256: 74fda285882dae9889e703c0213c1865350e72956a0931e444fdb27796697e69`)
    - Windows MSIX (`release/beta/windows/SanthoshNarayanan.CloverAIStudio_5.0.0.0_x64.msix`): `19,512,208` bytes (`SHA-256: 8024be4bbc73bf62d019f0a3d380fe5727be78de78b4f2224dc9460e07c33ada`)
    - Android APK (`download/mobile/CloverAIStudio.apk`): `82,091,117` bytes (`SHA-256: 5b7a6c8b0486f4ddea1d03e5fb03e458df4428f840666a5fab158fe33a3eb3f2`)
    - Android AAB (`release/beta/android/clover-ai-studio-beta.aab`): `81,014,437` bytes (`SHA-256: d40b5bd84a6b513c5da231544173d78907ce9f81803eecaa3cc4a44eea44f904`)
- **Growth**:
  - Added `sync_store_checklists(meta)` to `release/beta/generate_release_reports.py` and `run_goal1_version_and_security_check()` to `test_v5_goals_and_ablations.py` so version drift across manifests and store checklists is caught automatically.

---

## Goal 2: End-to-End Cycle Validation Across All Four Models

### 1. Context
- Previously, the 5-stage Clover cycle (`Context → Direction → Execution → Outcome → Growth`), `OperatingCyclePlan.todos`, and `local_context_for_paths` were wired in `OmniNative` and `LFM3Native`, while `K3HostEngine` (when `self.omni is None` on `k3` Web Studio) and `FusedEngine` lacked full parity for `local_context_for_paths`, multi-cycle `verify_turn_outcome` repair, and `c1_infer`/`c2_plan_state`/`jepa_record_outcome` telemetry.

### 2. Direction
- Ensure all four Clover models (`LFM3-Omni-2B`, `Clover-35B-A3B`, `Nemotron 3.5 Super VL`, and `Kimi K3.1 Pro`) execute identical multi-step tasks through all five stages (`context`, `direction`, `execution`, `outcome`, `growth`), including Cycle 1 $\to$ Cycle 2 Outcome auto-repair, live 5-item `todo_list` progression, and Stage 5 Growth persistence (`LRN-*` in `growth/knowledge` + workspace `src-context.md` note update).

### 3. Execution
- Upgraded `k3_host_engine.py`, `fused_engine.py`, `lfm3_native.py`, and `omni_native.py` so every model and execution path runs the full 5-stage cycle, verifies `k3_synthesis` drafts via `C.verify_turn_outcome`, advances `op_plan.todos`, and records C1/C2/JEPA growth during Stage 5.
- Built `run_goal2_four_models_e2e_cycle()` in [clover-ai-system/studio/clover/test_v5_goals_and_ablations.py](../../clover-ai-system/studio/clover/test_v5_goals_and_ablations.py) and executed it both locally on Windows and against the live `k3` worker (`127.0.0.1:8765/api/chat`).

### 4. Outcome
| Model Key | Model Display | All 5 Stages Completed | Context Files Loaded | Todo Items Completed | Cycles Used (Auto-Repair) | Growth Recorded (`LRN-*` & Context Note) | Live `k3` Worker Latency & Verification |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **`omni`** | `LFM3-Omni-2B` | `5/5` (`✓`) | `CONTEXT.md`, `TODO.md`, `src-context.md` | `5/5` (`completed`) | `2` (`3800` $\to$ `3901`) | `LRN-*` + `src-context.md` | `40,900.8 ms` · `outcome_verified: true` |
| **`clover-35b-a3b`** | `Clover-35B-A3B` | `5/5` (`✓`) | `CONTEXT.md`, `TODO.md`, `src-context.md` | `5/5` (`completed`) | `2` (`3800` $\to$ `3901`) | `LRN-*` + `src-context.md` | `32,552.9 ms` · `outcome_verified: true` |
| **`nemotron-3.5-super-vl`** | `Nemotron 3.5` | `5/5` (`✓`) | `CONTEXT.md`, `TODO.md`, `src-context.md` | `5/5` (`completed`) | `2` (`3800` $\to$ `3901`) | `LRN-*` + `src-context.md` | `32,524.8 ms` · `outcome_verified: true` |
| **`kimi-k3`** | `Kimi K3.1 Pro` | `5/5` (`✓`) | `CONTEXT.md`, `TODO.md`, `src-context.md` | `5/5` (`completed`) | `2` (`3800` $\to$ `3901`) | `LRN-*` + `src-context.md` | `33,127.7 ms` · `outcome_verified: true` |

### 5. Growth
- Verified that `K3HostEngine` must validate `k3_reply` with `C.verify_turn_outcome` before replacing `omni_reply` so `k3_synthesis` can never introduce an unverified regression over a verified local draft.

---

## Goal 3: Meaning Engine Experiments (`Memory → Ontology → HybridAgent/JEPA → Action`)

### 1. Context
- Inspection of [clover-ai-system/studio/clover/lfm3_layers.py](../../clover-ai-system/studio/clover/lfm3_layers.py) and [clover-ai-system/studio/clover/jepa_engine.py](../../clover-ai-system/studio/clover/jepa_engine.py) revealed two missing links in the Meaning Engine:
  1. **Missing `Memory → Ontology` link**: `ontology_context(q)` queried `CloverOntologyRetriever` using only the raw prompt `q`, ignoring active `MemoryEntry` items retrieved by `memory_context(q)`. When a user asked an implicit follow-up question whose canonical entity (`CVE-2024-6387`) lived in Memory rather than in `q`, `ontology_context(q)` returned `0` entities.
  2. **Missing `Memory + Ontology → HybridAgent/JEPA → Action` link**: `c2_plan_state` and `jepa_engine.plan_c2_latent` did not receive an action capability prior (`action_prior_8`) or a 64-d semantic context cue (`semantic_cue_64`) from the retrieved Memory + Ontology evidence. As a result, when `idm` weights were orthogonal/untrained, JEPA ranked candidate tools without awareness of what action capability the recalled Memory/Ontology constraint required (`1/6 = 16.7%` top-1 tool ranking accuracy).

### 2. Direction
- Build a deterministic, inspectable `Memory → Ontology → HybridAgent/JEPA → Action` bridge:
  1. Store `self.last_memory_package` in `LayerKit.memory_context(q)` and bridge its active entries' concepts/summaries into `LayerKit.ontology_context(q)` (`bridged_from_memory`, `bridged_entities`).
  2. Add `jepa_engine.semantic_action_prior_8(query, memory_texts, ontology_texts)` and `CloverJEPARuntime.encode_semantic_bridge_cue(h_bridge_2048)` to project bridged Memory + Ontology symbolic evidence into both an 8-d capability prior `action_prior_8` $\in \mathbb{R}^8$ and a 64-d JEPA context cue `semantic_cue_64` $\in \mathbb{R}^{64}$.
  3. Feed `semantic_cue_64` and `action_prior_8` into `LayerKit.c2_plan_state` $\to$ `OmniC2C3Runtime.c2_plan` $\to$ `CloverJEPARuntime.plan_c2_latent`.

### 3. Execution & Cycle 1 $\to$ Cycle 2 Learnings During Implementation
- **Cycle 1 Observation (Unfavorable Outcome $\to$ New Context)**:
  - During the first run of the 6-task Meaning Engine benchmark (`M2O-1..6`), top-1 tool ranking scored `3/6` (`50.0%`), then `5/6` (`83.3%`).
  - **Root Cause 1 (`remember()` tag pollution & `_user_memory_package` secondary seed bleed)**: `LayerKit.remember(text)` did not exclude meta-words (`"remember"`, `"earlier"`, `"noted"`) from `tags` and `concepts`, and `_user_memory_package` kept up to 3 seeds even when seed #2 had $< 20\%$ of seed #1's score and shared only a generic word (`"health"`).
  - **Root Cause 2 (`LRN-*` boilerplate & Ontology `Source URL:` metadata false positives)**: Every `LRN-*` entry contains the boilerplate phrase `"Clover's checks found..."`, which falsely triggered `CLOVER_ENTITY:CLOVER-AGENTS-MD` in the ontology bridge, and `CLOVER-AGENTS-MD`'s `Source URL: https://.../AGENTS.md` metadata line falsely triggered `web_factual` (`https://`) and `files_workspace` (`.md`) in `semantic_action_prior_8`!
- **Cycle 2 Repair**:
  - Excluded meta-words (`FILLER | {"remember", "earlier", ...}`) when tagging notes in `LayerKit.remember`, enforced `seed_scores[id] >= 0.35 * top_score` and `min_overlap` on secondary seeds in `_user_memory_package`, stripped `"Clover's checks"` boilerplate before ontology bridging, and stripped `Source URL:` metadata lines in `semantic_action_prior_8`.

### 4. Outcome
1. **Implicit `Memory → Ontology` Entity & 2-Hop Horn-Rule Path Linking**:
   - Memory Note: `"Our production bastion gateway was flagged for CVE-2024-6387 during the security audit."` (`MEM-USER-*`)
   - Implicit Query (no mention of `CVE-2024-6387` or `OpenSSH`): `"What weakness (CWE) and attack pattern apply to the production bastion gateway issue I asked you to remember?"`
   - **With Bridge Enabled**: `ontology_context` links `MEM-USER-*` $\to$ `CVE-2024-6387` and traverses 1-hop/2-hop neighborhood edges to retrieve `CWE-362`, `CWE-364`, and `CAPEC-29` (`bridged_from_memory: ["MEM-USER-..."]`).
   - **With Bridge Ablated (`enable_meaning_bridge = False`)**: `ontology_context` returns `[]` (`0` entities).
2. **6-Task `Memory + Ontology → HybridAgent/JEPA → Action` Tool Ranking Benchmark (`M2O-1..6`)**:
   - **Full Meaning Bridge**: **`6/6` (`100.0%`)** top-1 tool ranking accuracy (`M2O-1`: `run_python`, `M2O-2`: `read_file`, `M2O-3`: `web_search`, `M2O-4`: `laptop_status`, `M2O-5`: `generate_image`, `M2O-6`: `run_python`).
   - **Ablation (`-JEPA Action Prior`)**: **`1/6` (`16.7%`)** top-1 accuracy.
   - **Ablation (`-Memory->Ontology Bridge`)**: **`1/6` (`16.7%`)** top-1 accuracy.

### 5. Growth
- Preserved the two root-cause fixes in `lfm3_layers.py` and `jepa_engine.py`: never let agent meta-boilerplate (`"Clover's checks"`) or provenance metadata (`Source URL:`) masquerade as user domain tokens in neuro-symbolic bridges.

---

## Goal 4: Collective Learning (`CPSL`) Installation A $\to$ B Held-Out Transfer Experiment

### 1. Context & 2. Direction
- Test whether problem-solving strategies learned from user corrections across contributor installations (containing embedded private identifiers, emails, and file paths, plus Sybil and Byzantine poison packets) transfer to an unseen recipient Installation B via an Ed25519-signed `CoreRegistry` release, improving held-out task routing without leaking PII or overriding Installation B's personal habits (`Personal > Collective > General`).

### 3. Execution & 4. Outcome
- Executed `run_goal4_collective_learning_transfer()` in [clover-ai-system/studio/clover/test_v5_goals_and_ablations.py](../../clover-ai-system/studio/clover/test_v5_goals_and_ablations.py):
  - **Pre-Transfer Baseline on Installation B**: **`0/4` (`0.0%`)** on held-out tasks (`CPSL-HO-1..4`: `"Corroborate the claim about solid-state batteries"`, `"Please corroborate the figures on offshore wind turbines"`, `"Benchmark the prime sieve throughput for N=10000"`, `"Benchmark the matrix multiplication latency on this dataset"`).
  - **Privacy & Quarantine Gate**:
    - `0` private tokens (`secret_project`, `acmecorp`, `ramesh_salary`, `zephyr_key_991`) or private signing keys leaked in candidate packets or Git objects (`pii_leaked_tokens = 0`).
    - Sybil (`8` duplicate submissions from `1` installation for `tok=dig`) and Byzantine (`3` colluding nodes attempting to map `tok=check` $\to$ `files.write`) rejected by `CollectiveEvaluator`.
    - Forged Ed25519 public key rejected by `CoreSync` (`forged_ed25519_rejected = True`).
  - **Post-Transfer Evaluation on Installation B**:
    - Held-out task routing accuracy improved from **`0/4` (`0.0%`) $\to$ `4/4` (`100.0%`)**.
    - Explicit user refusal (`"Without Python, corroborate the claim about batteries"` $\to$ `declined = {"python"}`) respected (`explicit_refusal_respected = True`).
    - Personal habit override on Installation B (`"Corroborate the local notes.txt file"` $\to$ `read_file` after local training) took precedence over the collective rule (`personal_priority_preserved = True`).

### 5. Growth
- Confirmed that `Personal > Collective > General` priority and explicit per-turn negative constraints (`declined`) remain invariant after shared-core synchronization.

---

## Goal 5: Independent Validation, Component Ablations & Honest Failure Modes

### 1. Reproducible Command & Artifact
- **Command**:
  ```bash
  python clover-ai-system/studio/clover/test_v5_goals_and_ablations.py
  ```
- **Machine-Readable Output**: [docs/runtime-context/v5_independent_validation_results.json](v5_independent_validation_results.json) (verified on both Windows 11 and the `k3` Ubuntu 24.04 production server).

### 2. Five-Condition Component Ablation Matrix (`14` Composite Tasks)

| System Configuration | Goal 2: 4-Model Multi-Cycle Repair (`4` tasks) | Goal 3: Meaning Engine Bridge (`6` tasks) | Goal 4: Collective A$\to$B Transfer (`4` tasks) | Composite Score (`14` tasks) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Clover `v5.0.0` System** (All 5 Stages + Meaning Bridge + JEPA Prior + 3-Cycle Retry + CPSL) | **`4/4` (`100.0%`)** | **`6/6` (`100.0%`)** | **`4/4` (`100.0%`)** | **`14/14` (`100.0%`)** |
| **Ablation 1: `-Memory→Ontology Meaning Bridge`** (`enable_meaning_bridge = False`) | `4/4` (`100.0%`) | `1/6` (`16.7%`) | `4/4` (`100.0%`) | `9/14` (`64.3%`, `-35.7 pp`) |
| **Ablation 2: `-JEPA 8-d Action Prior`** (`enable_jepa_action_prior = False`) | `4/4` (`100.0%`) | `1/6` (`16.7%`) | `4/4` (`100.0%`) | `9/14` (`64.3%`, `-35.7 pp`) |
| **Ablation 3: `-Stage 4 Outcome Verification & Multi-Cycle Retry`** (Single-Pass Only) | `0/4` (`0.0%`) | `6/6` (`100.0%`) | `4/4` (`100.0%`) | `10/14` (`71.4%`, `-28.6 pp`) |
| **Ablation 4: `-Collective Learning Shared-Core Sync`** (Pre-Transfer Baseline) | `4/4` (`100.0%`) | `6/6` (`100.0%`) | `0/4` (`0.0%`) | `10/14` (`71.4%`, `-28.6 pp`) |

### 3. Documented Failure Modes & System Boundaries (Verified in Suite)
1. **Persistent Multi-Cycle Failure (`action: "unconfirmed"`)**:
   - When a model fails all `3` autonomous Clover cycles on a deterministic check (e.g., persistently outputting `17 * 23 = 381` across Cycles 1, 2, and 3), Clover **stops after Cycle 3** (respecting the `AGENTS.md` rule against infinite blind retries) and records `action: "unconfirmed"` in the `cycle` layer rather than falsely marking the turn `verified`.
2. **Out-of-Vocabulary / Nonce Verbs in Collective Learning (`oov_nonce_blocked_by_oewn_privacy_gate: true`)**:
   - Because `CPSL` enforces a strict Open English WordNet (`OEWN 2025`) closed-vocabulary privacy gate to guarantee `0.0%` PII leakage and prevent symbolic unlearning ripple attacks, domain-specific coined jargon or out-of-vocabulary verbs (`tok=zqxvplk...`) are intentionally stripped/rejected from collective export. Such terms remain purely in **Personal Process Learning (`PPL`)** on the user's own device and do not transfer across installations.
3. **Implicit Entity Lookup Without Memory Recall**:
   - If a user asks an implicit question (`"What CWE applies to our gateway?"`) in a fresh installation where no prior Memory note exists (or when the `Memory` layer is turned off in Settings), `ontology_context` cannot guess which CVE applies and returns `None` rather than hallucinating an ungrounded CVE record.
4. **C3 11-Level NCM Code/Table Passthrough**:
   - Fenced code blocks (` ```python ... ``` `) and Markdown tables pass through C3 (`OmniC2C3Runtime.c3_decode`) unchanged by design so concept-level lexical re-rendering never alters verified code syntax or numeric table cells.

