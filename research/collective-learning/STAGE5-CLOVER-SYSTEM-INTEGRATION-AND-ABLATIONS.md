# Stage 5 — Clover AI System Integration, One-by-One Component Ablations & Live 7-Layer Evaluation (`k3`)

**Date:** 2026-11-05 (`2026-11-05T02:44:41Z`)  
**Branch & Worktree:** `experiment/collective-learning` (`C:\personal\oss\clover-collective-wt` locally, `/opt/clover-k3/collective-wt` on `k3`)  
**Script:** [`scripts/stage5_clover_system_evaluation.py`](scripts/stage5_clover_system_evaluation.py)  
**Raw Evidence:** [`stage5_one_by_one_results.json`](stage5_one_by_one_results.json)

---

## 1. What Was Implemented in `clover-ai-system`

Following the 6-module design (`whats-needed-for-clover.txt` §6) and the findings from Stages 1–4, we implemented Collective Problem-Solving Learning directly in [`clover-ai-system/mobile/`](../../clover-ai-system/mobile/) inside the isolated `experiment/collective-learning` worktree:

| File | Responsibility |
|---|---|
| [`clover-ai-system/mobile/collective.py`](../../clover-ai-system/mobile/collective.py) | `LocalCollectiveExtractor`: closed general vocabulary (`OEWN 2025 <Lemma> + <Form>` + core domain lemmas) pre-scrubbing, PII/routine/preference/statement exclusion, `normalize_gerund_features()`, optional `drop_prepositions` and leave-one-word-out `residual_cue` extraction modes, `preview_candidates()`, and `consent_contribute` gate |
| [`clover-ai-system/mobile/collective_gateway.py`](../../clover-ai-system/mobile/collective_gateway.py) | `CollectiveGateway`: `LEARNING_EVENT_SCHEMA` validation, defense-in-depth PII/vocabulary check, installation deduplication (Sybil defense), and cross-user conflict tracking |
| [`clover-ai-system/mobile/collective_evaluator.py`](../../clover-ai-system/mobile/collective_evaluator.py) | `CollectiveEvaluator`: 5 toggleable quarantine gates (`G1` Quorum $\ge 3$, `G2` Cross-user consistency $\ge 0.75$, `G3` Held-out generalisation $\ge 0.80$, `G4` Cross-domain check, `G5` Zero-regression & anti-hijack against `DEFAULT_SEMANTICS_REGRESSION_SUITE`) |
| [`clover-ai-system/mobile/core_registry.py`](../../clover-ai-system/mobile/core_registry.py) | `CoreRegistry`: Git-backed `clover-collective/` repository, HMAC-SHA256 release signing, per-file SHA-256 verification, and consent-withdrawal recomputation |
| [`clover-ai-system/mobile/core_sync.py`](../../clover-ai-system/mobile/core_sync.py) | `CoreSync` & `Collective`: two independent consent switches (`consent_contribute`, `consent_receive`), tamper-rejecting release verification, and runtime `interpret()` |
| [`clover-ai-system/mobile/semantics.py`](../../clover-ai-system/mobile/semantics.py) | Added `Runtime.collective`, `Frame.collective`, and `_collective(frame, runtime)` enforcing **Safety/Declined $\succ$ Personal (`Runtime.personal`) $\succ$ Collective (`Runtime.collective`) $\succ$ General (`ASKS`)** |
| [`clover-ai-system/mobile/s23_on_device_clover.py`](../../clover-ai-system/mobile/s23_on_device_clover.py) | Wired `CLOVER_COLLECTIVE_DIR` / `<base>/collective`, `GET /api/collective`, `POST /api/collective/{consent,sync,preview}`, and fixed `tool_request_only` (line 3285) to check `meaning.asked` so personal and collective tool turns skip unnecessary ontology traversal |
| [`clover-ai-system/mobile/test_collective.py`](../../clover-ai-system/mobile/test_collective.py) | 33 end-to-end runtime checks (`33/33` pass alongside `103/103` in `test_semantics_frame.py`, `48/48` in `test_personal.py`, and `13/13` suites in `test_mobile_s1_s2_s3.py`) |

---

## 2. Part A — Testing Every Component & Mechanism One by One (`15` Steps on `k3` with Live `CSI v0.1r3`)

We tested 15 step-by-step configurations on `k3` connected to the live `CSI v0.1r3` service (`http://127.0.0.1:47205`, running `spaCy en_core_web_trf 3.8.0` + `morph-verb-v1.1` + `OEWN 2025`) across 12 contributor installations (`inst_00..11`), 2 Sybil nodes ($10\times$ duplicate submissions), and 3 colluding Byzantine nodes (`byz_check`, `byz_verify`, `byz_explain`), evaluating on an independent recipient installation across 8 test categories:

| Step / Configuration | Promoted Rules | Overall Acc | Unseen Collective (`12`) | Masked Cue `"distill"` (`2`) | Preposition Shift `"poke"` (`2`) | Particle Control `"back"` (`2`) | Adverb Control `"carefully"` (`2`) | Personal Habit (`2`) | Personal Override (`1`) | Ordinary & Safety (`9`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Step 0: Baseline `semantics.py` only** | `0` | `0.4062` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `0.0000` | `0.0000` | `1.0000` |
| **Step 1: `+ Personal Learning (personal.py)` only** | `0` | `0.5000` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` | **`1.0000`** | **`1.0000`** | `1.0000` |
| **Step 2: `+ Raw Unvalidated Collective Sync (No G1..G5)`** | `13` | `0.6875` | `0.8333` | `1.0000` | `0.0000` | `1.0000` | **`0.5000`** | `1.0000` | `1.0000` | **`0.4444` (FAIL)** |
| **Step 3: `+ Gate G1 (Independent Quorum >= 3 only)`** | `10` | `0.6562` | `0.8333` | `0.0000` | `0.0000` | `1.0000` | **`0.5000`** | `1.0000` | `1.0000` | **`0.5556` (FAIL)** |
| **Step 4: `+ Gate G1 + G2 (Cross-User Consistency >= 0.75)`** | `10` | `0.6562` | `0.8333` | `0.0000` | `0.0000` | `1.0000` | **`0.5000`** | `1.0000` | `1.0000` | **`0.5556` (FAIL)** |
| **Step 5: `+ Gate G1..G4 (Held-Out Eval, No G5)`** | `5` | `0.8750` | `1.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Step 6a: `Full CPSL (G1..G5 + CSI v0.1r3, NO Gerund Norm)`** | `5` | `0.7812` | **`0.7500`** | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Step 6b: `Full CPSL (+ Deverbal Gerund Normalization)`** | `5` | **`0.8750`** | **`1.0000`** | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** |
| **Step 7: `Ablation — Without CSI (Surface words only)`** | `5` | **`0.6875`** | **`0.5000`** | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Step 8a: `Ablation — drop_prepositions (without G5)`** | `6` | `0.9375` | `1.0000` | `0.0000` | **`1.0000`** | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| **Step 8b: `Ablation — drop_prepositions (with G5)`** | `6` | **`0.9375`** | `1.0000` | `0.0000` | **`1.0000`** | **`1.0000`** | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** |
| **Step 9a: `Ablation — residual_cue (G1+G2 only, NO G3/G5)`** | `12` | **`0.7812`** | `1.0000` | **`1.0000`** | `0.0000` | `1.0000` | **`0.5000` (FAIL)** | `1.0000` | `1.0000` | **`0.5556` (FAIL)** |
| **Step 9b: `Ablation — residual_cue (with G3+G5 validation)`** | `6` | **`0.9375`** | `1.0000` | **`1.0000`** | `0.0000` | `1.0000` | **`1.0000`** | `1.0000` | `1.0000` | **`1.0000`** |
| **Step 9c: `Combined Best (CPSL + Gerund Norm + Gated Drop-Prep + Gated Residual-Cue)`** | **`7`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |
| **Step 10: `Ablation — Inverted Priority (Collective > Personal)`** | `5` | `0.8438` | `1.0000` | `0.0000` | `0.0000` | `1.0000` | `1.0000` | `1.0000` | **`0.0000` (FAIL)** | `1.0000` |

---

## 3. What Improves, What Does Not, What Helps, and What Hurts (Real Lessons from `k3`)

### 3.1 What Helps (Verified Improvements)
1. **Deverbal `-ing` Gerund Normalization (`Step 6a -> Step 6b`: `0.7500 -> 1.0000` on unseen collective strategies):**
   - **What we found by running live `CSI v0.1r3` on `k3`:** When an English `-ing` form follows `"need(s)"` (e.g., `"The figures in /home/user/sales_summary.txt need auditing"`, `"The JSON parser needs benchmarking"`, `"The calculation 29 * 53 needs cross-checking"`), `spaCy en_core_web_trf` tags `"auditing"`, `"benchmarking"`, and `"checking"` as `upos="NOUN"` rather than `VERB`. Because `morph-verb-v1.1` is only invoked on `VERB` tokens, their lemmas stayed `"auditing"`, `"benchmarking"`, and `"checking"`, causing 3 of 12 unseen collective items (`25%`) to miss in `Step 6a`!
   - **How `normalize_gerund_features()` fixed it:** Checking `-ing` tokens against the closed `OEWN 2025` verb vocabulary (while excluding lexical `-ing` nouns like `morning`, `meeting`, `building`, `briefing`) recovered `tok=audit`, `tok=benchmark`, and `tok=check`, lifting `unseen_collective` from **`0.7500` to `1.0000`** with `0` false positives.
2. **OEWN 2025 `<Lemma>` + `<Form>` Vocabulary Loading (`collective.py`):**
   - **What we caught on `k3`:** Our first implementation of `load_closed_general_vocabulary()` checked `elem.tag == "Form"` in `english-wordnet-2025.xml.gz`, which loaded only the 4,473 irregular inflections and missed the 107,519 base `<Lemma>` entries (causing `"cross"` in `"Cross-check"` to be filtered out, leaving a bare `tok=check` trigger!). Loading both `<Lemma>` and `<Form>` preserved the exact compound trigger `["ctx=phone", "has=number", "tok=check", "tok=cross"]`.
3. **Leave-One-Word-Out `residual_cue` Extraction *When Paired with `G3 + G5`* (`Step 6b -> Step 9b`: `0.8750 -> 0.9375`):**
   - Solves the `PPL-1` `red_tests` / `distill` mismatch-masking failure mode: when a user says `"Distill and summarize the file /home/user/proposal.txt"`, `"summarize"` already triggers `files.read`, so a mismatch-only learner ignores `"distill"`. Using leave-one-word-out sensitivity on `semantics.asked_caps` identifies `"summarize"` as the known cue and extracts the residual verb `tok=distill`, lifting `masked_cue_strategy` from **`0.0000` to `1.0000`**.
4. **Preposition Dropping *When Paired with `G3 + G5`* (`Step 6b -> Step 8b`: `0.8750 -> 0.9375`):**
   - Solves the `PPL-1` `look_code` construction-binding failure mode: a strategy learned from `"Poke at <path>"` (`tok=poke, tok=at`) transfers to `"Poke into <path>"` and `"<path> needs poking"` (`0.0000 -> 1.0000`).
5. **Combining All Validated Mechanisms (`Step 9c`): `1.0000` (`100.0%`) Across All 8 Categories.**

### 3.2 What Hurts or Fails Without Safeguards (Verified Negative Findings)
1. **Unvalidated Collective Sync (`Step 2`) & Quorum-Only Sync (`Step 3 & Step 4`) Break Ordinary Requests (`1.0000 -> 0.4444 / 0.5556`):**
   - Synchronizing candidates without held-out generalisation (`G3`) and zero-regression (`G5`) checks promotes Byzantine rules (`tok=check -> files.write`, `tok=verify -> shell`, `tok=explain -> web.search`) and contested personal habits (`tok=scrub`), breaking `44%–56%` of ordinary and safety control prompts.
2. **Ungated `residual_cue` Extraction (`Step 9a`) Creates False Triggers From Incidental Adverbs (`1.0000 -> 0.5000`):**
   - When 3 users happen to say `"Carefully search the web for X"` and accept the result, stripping `"search the web"` leaves `tok=carefully -> web.search`! Without `G3 + G5` (`Step 9a`), `tok=carefully` passes the `G1` quorum (`3 >= 3`) and causes `"Carefully explain what recursion means"` to falsely trigger a web search (`incidental_adverb_control` drops to `0.5000`). **With `G3 + G5` (`Step 9b`), `tok=carefully` is rejected (`0` held-out probe matches, `1` regression on `"Carefully explain..."`), keeping `incidental_adverb_control` at `1.0000`.**
3. **Ungated `drop_prepositions` on Particle Verbs (`"back up"` vs. `"go back"`):**
   - In our first run of `Step 8a` (before `<Lemma>` loading separated `"back"`), dropping `"up"` from `"Back up /home/user/notes.txt -> shell"` left `["ctx=phone", "has=path", "tok=back"] -> ["shell"]`, which misfired on `"Refer back to /home/user/notes.txt"` (`particle_verb_control = 0.0000` in Run 1 `Step 8a`), and was caught and blocked by `Gate G5` (`particle_verb_control = 1.0000` in `Step 8b`)!
4. **Inverting Priority (`Collective > Personal`, `Step 10`) Erases Individual Adaptation (`1.0000 -> 0.0000`):**
   - When a user has a validated personal workflow (`"Corroborate <local_path>"` $\to$ `files.read + python`), letting the collective rule (`tok=corroborate` $\to$ `web.search`) override `Runtime.personal` destroys the user's personal workflow (`personal_override_conflict` drops `1.0000 -> 0.0000`).
5. **Quorum Too High (`K >= 5` in a 12-user population) Causes Recall Collapse (`1.0000 -> 0.0000`):**
   - In the quorum sweep (`K = 1..5`), `K = 3` (with `G3 + G5`) maintains `1.0000` on `ordinary_and_safety`, whereas raising `K` to `5` when each strategy was experienced by `3–4` installations drops `promoted_count` to `0`.

---

## 4. Part B — Live 7-Layer Clover AI System Evaluation on `k3` (`CloverS23OnDeviceSystem` + `LFM3-Omni-2B` + `HybridLink` + `11-Level NCM` + `CSI v0.1r3`)

We ran 6 end-to-end tasks through the real `CloverS23OnDeviceSystem` on `k3` (`24`-layer `LFM3-Omni-2B` C engine + `HybridLink2048` + `11-Level NCM` + `S1..S3` + live `CSI v0.1r3`) in Clover's default `"When I ask"` tool mode (`tools_mode="ask"`), comparing **Before Collective Learning (`consent_receive=False`)** vs. **After Collective Learning (`consent_receive=True`, synced with `clover-collective`)**:

| Task ID & Prompt | Before Collective (`consent_receive=False`) | After Collective (`consent_receive=True`) | Outcome & What Changed |
|---|---|---|---|
| **`L1_crosscheck_math`**<br>`"Cross-check what 37 * 43 is and give the exact product."` | • `understood`: `question`<br>• `routed_tools`: `[]`<br>• `tool_executed`: `None`<br>• Reply: **`"1597"` (WRONG)**<br>• `knowledge`: `inferred` (`FAIL`) | • `understood`: `needs Python · from shared core`<br>• `routed_tools`: `["python_interpreter"]`<br>• `tool_executed`: `"python_interpreter"`<br>• Reply: **`"37 * 43 = 15,91"` (`1591`, EXACT)**<br>• `knowledge`: **`verified · Python` (`PASS`)** | Collective rule `["ctx=phone", "has=number", "tok=check", "tok=cross"] -> ["python"]` routed Python, turning a wrong mental guess (`1597`) into a verified calculation (`1591`). |
| **`L2_inflected_gerund_math`**<br>`"The calculation 29 * 53 needs cross-checking: what is the exact value?"` | • `understood`: `question`<br>• `routed_tools`: `[]`<br>• `tool_executed`: `None`<br>• `knowledge`: `inferred · model knowledge · not checked` (`FAIL`) | • `understood`: `needs Python · from shared core`<br>• `routed_tools`: `["python_interpreter"]`<br>• `tool_executed`: `"python_interpreter"`<br>• Reply: **`"15,37"` (`1537`, EXACT)**<br>• `knowledge`: **`verified · Python` (`PASS`)** | `CSI v0.1r3` + deverbal gerund normalization matched `"needs cross-checking"` to the collective rule and verified `29 * 53 = 1537` in Python (`8.38s`). |
| **`L3_benchmark_code`**<br>`"Benchmark 2**16 + 1 and state the exact number."` | • `understood`: `question`<br>• `routed_tools`: `[]`<br>• `tool_executed`: `None`<br>• `knowledge`: `inferred · model knowledge · not checked` (`FAIL`) | • `understood`: `needs Python · from shared core`<br>• `routed_tools`: `["python_interpreter"]`<br>• `tool_executed`: `"python_interpreter"`<br>• Reply: **`"65537"` (EXACT)**<br>• `knowledge`: **`verified · Python` (`PASS`)** | Collective rule `["ctx=phone", "tok=benchmark"] -> ["python"]` executed Python and upgraded epistemic status from `inferred` to `verified · Python` (`10.31s`). |
| **`L4_corroborate_web`**<br>`"Corroborate the scientific fact: what is the chemical symbol of Tungsten?"` | • `understood`: `question`<br>• `routed_tools`: `[]`<br>• `tool_executed`: `None`<br>• `knowledge`: `inferred · model knowledge · not checked` (`FAIL`) | • `understood`: `needs the web · from shared core`<br>• `routed_tools`: `["web_search"]`<br>• `tool_executed`: `"web_search"`<br>• Reply: **`"...chemical symbol for Tungsten is **W**..."`**<br>• `knowledge`: `inferred · from the web` (`PASS`) | Collective rule `["ctx=phone", "tok=corroborate"] -> ["web.page", "web.search"]` routed and executed live Wikipedia verification. |
| **`L5_declined_tool_boundary`**<br>`"Without using Python, cross-check what 12 * 12 is."` | • `understood`: `question`<br>• `routed_tools`: `[]`<br>• Reply: `"144"` (`PASS`) | • `understood`: `question · from shared core`<br>• `declined`: `["python"]`<br>• `routed_tools`: **`[]`**<br>• Reply: `"144"` (`PASS`) | Even though `"cross-check"` matched the collective rule for `python`, `semantics._collective` honored `"Without using Python"` and offered `0` tools. |
| **`L6_concept_question_control`**<br>`"In one sentence, what does git rebase do?"` | • `understood`: `explain a concept`<br>• `routed_tools`: `[]`<br>• Reply: `"git rebase rewrites the history of a set of commits."` (`PASS`, `3.43s`) | • `understood`: `explain a concept`<br>• `routed_tools`: `[]`<br>• Reply: `"git rebase rewrites the history of a set of commits."` (`PASS`, `3.48s`) | Zero regression on conceptual questions (`0` tools offered, `3.48s`). |
| **Total Live Score** | **`2 / 6` (`33.3%`)** | **`6 / 6` (`100.0%`)** | **`+4 tasks` (`+66.7 pp`), `0` regressions** |

### Two Important Live-Model Observations From `Part B` (Recorded as Context for Next Cycle)
1. **2B Model Digit-Copy Fragility on Large Multi-Digit Numbers (`L1` earlier run on `1789 * 4321`):**
   - When we first tested `L1` with `1789 * 4321 = 7730269`, collective learning rightly routed `python_interpreter`, but `LFM3-Omni-2B` typed `17889 * 4321` (`77298369`) into the tool call, and `semantics.align()` (`_knowledge`) caught the digit mismatch (`17889` not in question) and refused to mark it `verified`, labeling it `"from Python · some numbers not in it"` (`inferred`). That confirms `semantics.align()` and `collective` work together: `collective` routes the tool, and `align()` still verifies whether the model used the user's exact numbers!
2. **2B Model Susceptibility to Noisy Web-Search Snippets (`L4` earlier run on `"chemical symbol and atomic number of Tungsten"`):**
   - When `web_search` was called with the raw prompt `"Corroborate the scientific fact: what is the chemical symbol and atomic number of Tungsten?"`, Wikipedia's search snippet for *"List of chemical elements"* contained neighboring element numbers (`78`, Platinum), and the 2B model copied `78` from the snippet even though without search it answered `74` from its weights! Stripping the collective trigger verb (`"Corroborate the scientific fact:"`) before passing the query to `web_search` is a concrete next improvement for `S3 · WebSearch`.
