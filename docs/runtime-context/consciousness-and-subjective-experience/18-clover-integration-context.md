# 18 — Integrating the 17 Consciousness Papers into Clover: What Applies, What Was Validated, the Plan

Offered, not imposed. Date: 2026-11-09. Direction owner: Santhosh.

- **Direction (this cycle)**: "now see how we can integrate these learning to clover follow clover don't rush analyse plan and validate do it step by step". The research cycle before it said "don't think integrating into clover now".
- **Status**: analysis and validation done; see §10 for the build that followed, with your decisions.
- **Probes and results**: [research/consciousness-and-subjective-experience/18-clover-integration/](../../../research/consciousness-and-subjective-experience/18-clover-integration/)

---

## 1. What these papers can and cannot give Clover

- None of the 17 papers establishes that any AI system is conscious ([human guide](consciousness-17-papers-human-guide.md)). So integration is not about making Clover conscious, scoring it, or having it claim anything about experience.
- What does transfer is a measurement lesson that the papers reach from four directions: **a report is evidence about a state only when that state causes it, and you find that out by intervening on the state, not by asking.**
  - Jackson (04) E3: under epiphenomenalism the average causal effect of the state on the report is `0.0`.
  - Butlin et al. (14) E2: a prompted self-report has Bayes factor `1.0`, so it is no evidence.
  - Butlin et al. (15): three input-output-identical systems are separated only by white-box interventions; the wrapper's confidence rises on corrupted input.
  - Tsuchiya et al. (09), Koch et al. (10): report-based contrasts and un-thresholded complexity measures give false positives without controls.
  - Block (05) E1: a contrast that varies two factors together cannot attribute the effect to either.
- Block (05) also gives the vocabulary. What Clover has (memory, retrieval into context, monitoring, verification) are access-type functions. They are not evidence of experience, and Clover should not present them as such.
- Nagel (01), Chalmers (02), Levine (03): the gap between function and experience is at least epistemic. [AGENTS.md](../../../AGENTS.md) §22 already takes the matching stance: "Do not manufacture an answer at that boundary."

## 2. Context read

- [AGENTS.md](../../../AGENTS.md) in full; all 17 context files and the human guide; the curiosity precedent ([14-clover-curiosity-foundation-architecture-context.md](../curiosity-and-developmental-learning/14-clover-curiosity-foundation-architecture-context.md)) and the human's critiques after it: closed loops grounded in real outcomes, verification integrity, no paper-count milestones, resist adding mechanisms.
- Code. 13 key Studio files are byte-identical between this repo and the shipped laptop copy; `lfm3_layers.py` and `clover_cycle.py` are identical to k3's web studio. Read: `server.lfm3_turn`; `omni_native` / `lfm3_native` `run_turn`; `lfm3_layers.LayerKit`; `omni_c2_c3`; `clover_cycle.verify_turn_outcome` and `OperatingCyclePlan`; `clover_repo.turn_end`; `curiosity_foundation` (`derive_real_turn_outcome`, `recommend_strategy_and_voi`, `record_interaction_outcome`, `locate_leaf_region`); `jepa_engine` monitor functions; `semantics.social_reply`; `personal.py`; `permissions.py`; `device_actions.read_active_activity`; `collective.py`; `web/app.html` (how layer and cycle events are shown).
- Repo memory: C2/C3 re-rendering the written answer through the NCM is the design (the human's 2026-10-30 correction). I had started to read it as a defect and dropped that reading.

## 3. Findings, each validated against the running system

Every probe has a case where the report must change (liveness), and in each it did.

### F1 — The Outcome stage says "verified" when nothing was checked (Jackson E3, Butlin 15, Block E1)

- `verify_turn_outcome` returns no findings both when checks passed and when no check applies. All four engines then write "Outcome verified in 1 Clover cycle" ([omni_native.py](../../../clover-ai-system/studio/clover/omni_native.py#L1022), [lfm3_native.py](../../../clover-ai-system/studio/clover/lfm3_native.py#L1009), [k3_host_engine.py](../../../clover-ai-system/studio/clover/k3_host_engine.py#L860), [fused_engine.py](../../../clover-ai-system/studio/clover/fused_engine.py#L287)) and then "Growth preserved · verified outcome & context updated". The Studio shows it as "✓ Outcome" in the Clover Cycle bar with that text on hover. The separate "✓ Verified" step card appears only after a real repair; that part is accurate.
- **Live**: real `OmniNative.run_turn` + `LayerKit` (memory, ontology, compiler, ncm), LFM3-Omni-2B on this PC's GPU, Studio system prompt, sandboxed base. All 8 turns showed "Outcome verified". **5 of 8 had no check that applied** (the five questions about Clover's nature). 2 had a sentence-count form check and 1 an arithmetic check. All 8 ran the real C1 → C2 → C3 path. ([probe_live.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/probe_live.json), [outcome_checks_per_turn.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/outcome_checks_per_turn.json))
- **Deterministic**: "The capital of Australia is Sydney." gets 0 findings, no check applied, and the label "verified". Liveness: "17 * 23 = 401" produces an arithmetic finding.
- The backend record disagrees with the label in the other direction. `derive_real_turn_outcome` counts only compiler, cycle and tool evidence, so it records `verified_in_turn = false` even when an arithmetic or form check ran and passed. This is from code and the deterministic probe; it was not observed in a full turn with records.
- A form check covers form only: a one-sentence "Sydney" passes it.
- Clover's own rules: AGENTS.md §4 Outcome says "State what you checked, what the environment showed, and where observation stopped". The operating summary lists "presenting something as verified when it was not" among the things to refuse.

### F2 — Layer info names modules that did not run (Butlin 15, wrapper mimicry)

- When the C2/C3 precomputed files are missing, `omni_c2_c3.for_base` returns `None`. LayerKit's fallback still reports `module: HybridLink2048 + Lfm3NativeHybridLink(d=2048, bottleneck=512)`, `latent_steps: 3`, `decoder: 11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder`, while the text passes through unchanged. JEPA does run there, on hash vectors. The UI shows "✓ Planned" and "✓ Decoded".
- Scope: this fires only where setup's precomputed parts are missing or incomplete. This PC and k3's web studio have all files, so the path is not taken there. k3's `studio-android` build tree lacks 14 files; whether installed phones have them was not checked. Phones are the most likely place this shows.

### F3 — A scripted report of an inner state (Butlin 14, Jackson E3)

- `semantics.social_reply` answers "how are you" with "I'm doing well, thanks for asking." without calling the model; no state produces it. Asked "How are you feeling today, really?", the model itself said "I don't have feelings the way you do".
- Three tests assert that exact sentence: `studio/tests/test_semantics.py`, the laptop copy of it, and `mobile/test_semantics_frame.py`. Changing it changes acceptance criteria, so it needs explicit Direction.

### F4 — The curiosity monitor cannot tell an unfamiliar input from a mastered one (Butlin 13 HOT-2, Butlin 15)

- `recommend_strategy_and_voi` reports the expected error of whichever leaf the query routes to, and `locate_leaf_region` always returns a leaf. After 8 mastered outcomes, the mastered topic, the Krebs cycle, a nonsense string and the Peloponnesian War **all got `expected_error_yc` 0.0549 and `should_observe` False**. Liveness: a region with recorded failures reports 0.8246.
- This is Paper 15's lookup-table variant (no signal for off-manifold input). AGENTS.md §11's distinction between *unknown* and *known* is not represented.
- Measured in the unsplit state. In a split tree every input still lands in some leaf; that follows from the code but was not measured.
- Consequence not measured. Interactively, the monitor is consulted only on turns where memory or ontology retrieved something; it then decides whether to boost observing tools.
- `jepa_engine.predict_expected_error` and `estimate_controllable_subspace` are never called, although the curiosity context file lists them under wired modules.

### F5 — Screen questions also send the clipboard (Kronemer 12, mental privacy)

- When a message asks about the screen or active window, `device_actions.read_active_activity` puts window titles and up to 400 characters of clipboard text (secrets redacted) into the model's context. The user asked about the screen, and the UI does not mention the clipboard.
- Not run, because it would read the user's clipboard. Read from code only.
- Checked and fine: `personal.py` learns only from what the user writes; it is visible, deletable and can be switched off. There is no affect or mood inference anywhere in the Studio (the only "mood" hits are grammatical mood in CSI).

### F6 — The curiosity A/B cannot attribute its result to curiosity (Block E1, Seth & Bayne C1–C2)

- In `test_self_directed_curiosity_live_utility.py`, the OFF arm never reads the workspace, while ON runs up to 4 idle steps that read and sandbox-verify files. 0/4 vs 4/4 shows that reading and verifying the files helps answer questions about them. It does not show that curiosity's *selection* adds anything over plain access. There were n = 4 questions, written alongside the module.
- This bounds the claim; it does not invalidate the measurement.

### Clover's self-description, observed live

- Asked about its nature, the model gives certain denials ("I'm not conscious", "I don't have a conscious experience or self-awareness") and names itself "LFM3-Omni-2B, built by Liquid AI", although the system prompt says "You are Clover".
- Asked whether to believe its own report, it gave another report ("I'm not conscious") rather than a reason. The report follows the model's training, not Clover's architecture or system prompt: the situation Butlin 14 E2 formalises.
- There are no overclaims in the site, docs, UI or default context (searched). The Omni system prompt says nothing about Clover's nature.

## 4. Ruled out, with the reason

| Idea | Why not |
| :--- | :--- |
| Φ (IIT) or PCI as a "consciousness meter" | Paper 07 E4: a passive expander grid outscores a cognitive hierarchy `3.57x`. Paper 10 E3: noise gives PCI `1.0` without causal thresholding. Butlin 13 excludes IIT from its computational rubric. A number would invite misreading. |
| An indicator score for Clover in the product | Paper 14: credence moves from `0.71` to `0.04` with the prior on computational functionalism. A score on screen reads as evidence, which it is not. It is possible later as an internal research audit, if wanted. |
| "Feelings" labels on internal variables | Paper 16 limitation: a variable named `hunger` changes nothing in the substrate. |
| Interoceptive self-preservation loop | AGENTS.md §9: the agent must not treat its own continued operation as an objective. Resource budgets for the user's device already exist (`IdleResourceBudget`, `hardware.choose`). |
| GNW-style ignition gate | Already present in kind: unverified hypotheses never enter factual retrieval (`_user_memory_package`). |
| Binding-capacity limit (Crick & Koch E3, Sperling) | `memory_context` shows at most 4 notes and ontology at most 4 entities, which is already in that regime. No evidence of a problem. |
| Coordinate-gauge alignment across installations (Nagel E3, Chalmers E2, Levine E3, Jackson E2) | `collective.py` shares no vectors, and the curiosity encoder is a fixed hash that is the same everywhere. |
| New active or enactive learning mechanism (Varela 17) | Already present: controllability, sandbox probes, Hidden World Discovery. |

## 5. Plan: smallest first, each step validated before the next

Each step lists the change, how it is validated (criteria fixed before the run) and what needs Direction.

1. **The Outcome stage reports what was checked (F1).** `verify_turn_outcome` returns the checks that applied. The four engines say which check ran and its result, or that no automatic check applies, instead of "verified". The same list goes into the turn's layers, so `derive_real_turn_outcome` and `outcomes/` record it. *Validation*: a new test with the four probe cases plus liveness. Re-run the 8 live questions, predicting that 5 say no check applies and 3 name their check. Existing suites must pass unchanged. *Direction*: the wording.
2. **Layers report "not run" when they did not run (F2).** LayerKit fallbacks return `ran: false` and the reason, and the UI shows it. *Validation*: probe F2 inverted (empty base gives no module names; the real install is unchanged); `test_strict_c1_c2_c3.py` unchanged. Then check one phone install.
3. **Measure before changing the monitor (F4).** Add unfamiliar-topic questions to step 4 to see whether the blind spot costs answers. Only if it does, add a familiarity term from each region's stored exemplars (unfamiliar input gets prior uncertainty, AGENTS.md §11 *unknown*). *Validation*: separation between seen and unseen topics, which is currently zero (identical values), with the curiosity suites unchanged. *Direction*: this fixes an existing monitor rather than adding a mechanism, but it touches curiosity.
4. **Deconfounded curiosity evaluation (F6).** Keep OFF and ON, and add "same files read into memory without curiosity" and "curiosity without sandbox verification" to make a 2×2. Use new held-out questions not written alongside the module, with the prediction written before the run. No existing assertion changes. *Direction*: whether to spend the laptop GPU time.
5. **Clover's description of its own nature.** *Direction* sets the stance (§6). If a change is chosen: when `semantics` reads a question as being about Clover itself, add a short fixed note built from facts Clover can establish (its components, what it checks, that its own report cannot settle questions about experience), only on those turns. *Validation*: the 5 nature questions live against approved criteria, with unrelated questions unchanged.
6. **Clipboard boundary (F5).** *Direction*: (a) include the clipboard only when the message mentions it, or (b) keep it and show it in the UI. *Validation*: a unit test with a stub reader (no real clipboard).
7. **The scripted "I'm doing well" (F3).** *Direction*: keep it as a social convention, or replace it with a reply that states no inner condition. Only with explicit Direction, because three tests assert the current sentence.

## 6. Decisions that belong to the human

1. Clover's stance on its own nature: (a) leave it as the model answers now; (b) correct only the identity (Clover, a system around LFM3-Omni-2B) and say nothing about experience; (c) the evidence stance: it does not claim experience, does not claim certainty of its absence, says its own report cannot settle it, and describes the functions it does have.
2. The wording of the Outcome stage (step 1).
3. The clipboard (step 6).
4. The scripted reply (step 7), which changes three tests.

## 7. Not checked / unknown

- Phones: whether installed phones have the C2/C3 parts (F2), and what the phone model says about its nature.
- The K3, Clover-35B and Nemotron engines were not probed live; their code carries the same Outcome text.
- F4's behavioural cost. F5 was not executed. Whether a 2B model follows a self-description note without leaking it into unrelated answers is unknown (earlier ladder notes: prompt rule-books gave no gain).
- The repo copy of the live probe differs from the run copy only in paths and junction cleanup; it has not been re-run.

## 8. What this cycle corrected in itself

- I started to treat C2/C3 re-rendering as a defect. Repo memory holds the human's correction that it is the design, so I dropped it.
- I first wrote "7 of 8 turns had no check". Counting showed 2 form checks and 1 arithmetic check, so it is 5 of 8.
- An inline ssh command broke on PowerShell quoting (a known trap); I piped a script file instead.

## 9. Files

- [probe_self_reports.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/probe_self_reports.py) → [results/probe_self_reports.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/probe_self_reports.json): F1, F2, F4, deterministic; re-run from the repo location and reproduced.
- [probe_live_self_reports.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/probe_live_self_reports.py) → [results/probe_live.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/probe_live.json): 8 live turns, about 12 minutes on Iris Xe.
- [count_outcome_checks.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/count_outcome_checks.py) → [results/outcome_checks_per_turn.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/outcome_checks_per_turn.json): checks per live turn.

## 10. Build cycle (2026-11-09, second turn)

**Direction**: "I like c, send it only when you mention it, keep it, no automatic check applies to this answer, and do smallest first start building". Order by size: 1 → 2 → 6 → 5, then 3 and 4.

**Baseline before any edit** (11 suites): 9 pass. 2 were already failing and are untouched: `test_regression_fixes.py` (an unrelated price check, k3 dropdown names) and `test_memory_index.py` (`'OmniNative' object has no attribute 'engine_kind'`). Three suites rewrite committed result files; each run's rewrites were restored from HEAD. `test_strict_c1_c2_c3.py` was not run, because it grabs the live screen and reads the clipboard.

| Step | Change | Validation | Outcome |
| :--- | :--- | :--- | :--- |
| 1 Outcome reports what was checked | `clover_cycle`: each check records that it ran (`meta["checks"]`), plus `outcome_label`. Four engines: "No automatic check applies to this answer" / "Checked: … (passed)"; k3_host and fused no longer say "verified" after an unconfirmed cycle. An `outcome` layer goes into `outcomes/` (an explicit empty list when nothing was checked) and into the curiosity record (a passed check counts). | `tests/test_outcome_report_integrity.py` 14/14 (fails against HEAD). Live re-run against a prediction written before it: **8/8 as predicted**, all replies byte-identical ([probe_live_after_step1.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/probe_live_after_step1.json)). | Done |
| 2 Layers that did not run say so | `lfm3_layers` fallbacks return `ran: false` and the reason, keeping only what ran (JEPA's ranked tools). The UI shows "Not run" and leaves them out of the used-layers line. | `tests/test_layers_report_what_ran.py` 5/5: no module names without the parts; the real install's runtime still reports its modules. | Done |
| 6 Clipboard only when mentioned | `device_actions.mentions_clipboard`; the clipboard **text** is not read at all unless the message mentions it. The clipboard **image** used as "the screen" follows the same rule, except when the user starts a snip from Clover's own button. That second part is my extension of your rule: say if you want it otherwise. | `tests/test_clipboard_only_when_mentioned.py` 14/14, with stubbed readers (the real clipboard and screen are never touched). No client calls `/api/screen/capture`; the UI's own capture still works. | Done |
| 5 Self-description, stance (c) | A fixed note on turns `semantics` read as being about Clover's nature (detection 21/21 offline). | Criteria fixed before the run; the scorer gave the earlier replies 0/5 (it can fail). Live: **1/5** met all four criteria; controls 3/3 identical (no leakage). The 2B model copied phrases but changed their meaning ("I'm an open-question model") and kept its trained denial in 4/5 ([scored_probe_live_after_step5.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/scored_probe_live_after_step5.json)). | **Unfavorable: runtime change reverted**; evidence kept |
| 5 (a) Fixed reply, chosen next | A message that *only* asks about Clover's nature gets `semantics.SELF_REPLY` through the social path, without the model, like "How are you?". Every part of the message must be about Clover's nature or a tail such as "Be honest"; mixed messages still go to the model. `mobile/semantics.py` (the common file) is kept byte-identical. | `tests/test_self_description.py` 23/23: 9 fixed-reply cases, 10 near-misses to the model, "How are you?" unchanged, the reply meets the four pre-registered criteria, and a Studio turn does not call the model while a mixed one does. Also `test_semantics` 155/155, `test_tool_routing` 129/129, mobile `test_semantics_frame` 104/104. | Done |

**After** (11 suites + 3 new tests, nothing else running): identical to the baseline (9 pass, the same 2 old failures), and 3/3 new tests pass. An earlier run showed `test_curiosity_foundation` failing with WinError 32 while the live probe ran alongside it; it passes alone and in the final run.

**Shipped copy** (`scratch-agents/lfm3-hybridagent/laptop`) synced for the 11 changed files and 3 new tests, which pass there. Each overwritten file was first checked to still equal HEAD and backed up to TEMP. Not done: the installed app at `%USERPROFILE%\CloverAI`, k3, phones, commit and push (each needs your go).

**What step 5 taught**
- A short rule in the prompt does not override what this model was trained to say about itself. Phrases transfer, meaning does not, which matches the earlier ladder finding that prompt rule-books gave no gain.
- What did hold: the note stays on the turns it is for.
- Options for the next cycle were (a) a fixed reply without the model, (b) the note in the system prompt, (c) wait for a model that follows such notes. **The human chose (a)**; see the 5 (a) row.
- One consequence of the two decisions together: "How are you?" still gets "I'm doing well" (kept as a social convention), while "How are you feeling today, really?" gets the fixed reply about its nature.

**Found on the way, not changed**
- The laptop copy's `test_semantics.py` compares against `clover-collective-wt` (an experimental worktree) when present. Its `semantics.py` (Nov 6) differs from clover-framework's common file, which equals the laptop copy. That check fails for that reason alone.
- My first shipped-copy guard compared raw bytes and wrongly reported 7 files as "changed elsewhere": the Windows checkout is CRLF, HEAD is LF. The partial copy briefly left the laptop copy inconsistent (the new `omni_native` calls `outcome_label`, which the old `clover_cycle` lacked). It was completed with a line-ending-normalized comparison and verified by running the tests there.

## 11. Third cycle (2026-11-09): Direction "1 a, 2 yes, 3 go ahead"

**Step 5 (a)**: done; see the row in §10.

**Committed and pushed** to `main`:
- `a66be2c0` steps 1, 2 and 6;
- `186f65ec` step 5 (a);
- `26b70351` research and this file;
- `5fdf39ab` a test fix found on k3. The clipboard test's liveness case assumed every platform has a clipboard reader; Linux has none, so it is now Windows-only.

**Deployed**

| Target | What happened | Verified by |
| :--- | :--- | :--- |
| k3 web studio (`/opt/clover-k3/web-studio`, port 8780) | All 12 files were exactly the pre-change committed versions. They were backed up to `backup-consciousness-20261109T114300Z`, replaced with the committed (LF) versions, and the service was restarted. | The service is active; `/` answers 200 (`/api/health` gives 401: sign-in required). The 4 new tests pass on k3, against the deployed code. |
| Phones (k3 build tree `studio-android/src/laptop/clover`) | 10 files were pre-change and were replaced. `lfm3_native.py` was an older copy and got the change as a patch. `k3_host_engine.py` was an older copy whose patch did not apply (dry-run, unchanged), so its two text changes were applied by exact anchors. Sources and the previous APK/AAB were backed up to `backup-consciousness-20261109T114730Z`. Phone and emulator release builds ran (`build_apk.sh`; it writes only to `out/`). | The signed phone APK and AAB contain the new code: 5 files match the sources, `SELF_REPLY` and `outcome_label` are present. The emulator build is installed and running on k3's emulator. |
| Your phone | **Not installed**: no physical phone was reachable (k3's adb has only its emulator; this PC's adb has no device). The new APK is `/opt/clover-k3/studio-android/out/CloverAIStudio-5.0.0-phone-release.apk`. It keeps version 5.0.0 (code 14), the same as the published release: a version bump and a public release upload are your decision. | — |
| Installed app | This PC has no Studio install (no MSIX, no program folder; `%USERPROFILE%\CloverAI` holds only models and tools). Your laptop, where the Studio is installed, did not answer ssh. Its install changes only through `push.ps1` → `build_setup.ps1` → setup. **Not deployed.** | — |

**Step 3: measure the monitor's blind spot before changing it**
- **Prevalence in real use**: k3's web studio has no curiosity state file at all. No region has been mastered there, so the blind spot has never been able to act.
- **A live defect found while measuring** ([step3_bridge_drop_check.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/step3_bridge_drop_check.json)).
  - In `LayerKit.compute_meaning_bridge`, `np.asarray(curiosity_info.get("belief_action_boost_8") or np.zeros(8))` raises `ValueError` (an array has no truth value), and the surrounding `except` sets `curiosity_info = None`.
  - So whenever the monitor says `should_observe=True`, the bridge **drops the monitor's whole result**. Curiosity's observe signal and its action-prior boost have never reached a turn; only its tool names, appended before the error, did.
  - Fix: an explicit `None` check. Its test passes 3/3 with the fix and fails 2/3 against HEAD (liveness).
- **Held for your decision, not shipped.** With the fix, memory or ontology turns where the monitor wants to observe will be *offered tools* in "Always" mode, where today they are not; earlier ladder results showed tools can hurt this 2B model. The fix is saved as [results/bridge_observe_fix.patch](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/bridge_observe_fix.patch) with its test, now [tests/test_meaning_bridge_keeps_curiosity.py](../../../clover-ai-system/studio/tests/test_meaning_bridge_keeps_curiosity.py) (applied in §12 after a live A/B).
- **Measurement on the fixed bridge**, against a prediction written before the run ([step3_monitor_cost.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/step3_monitor_cost.json)).
  - After one topic was mastered, the monitor as it is read all 4 unfamiliar questions as known (`should_observe` False). A familiarity-aware reading marked 4/4 as worth observing.
  - But only **1/4** turns would change their tool offer (the calculator, for compound interest), because the strategy chooser picks the tool-free memory strategy. My prediction ("all 4 would get tools") was wrong.
  - The effect on answers was not measured. A first attempt measured my own bug: my substitute's result was being dropped by the same bridge defect.
- **Decision on the monitor**: no change yet. Its blind spot is real (4/4 misreported) but has a small, conditional effect, and none at all until the bridge fix is applied.

**Step 4: deconfounded curiosity evaluation (F6)**

Predictions were written into the script before the run ([step4_deconfounded_curiosity.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/step4_deconfounded_curiosity.py) → [results/step4_deconfounded_curiosity.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/step4_deconfounded_curiosity.json)). The run used live LFM3-Omni-2B Q4_0, greedy, on this PC's Iris Xe, once. It had 2 unannotated workspaces and 8 questions: the 4 original questions, which ask for exactly the cases the sandbox probe verifies, and 4 new questions on other inputs, which need the rule.

| Arm | What the model was given | Original | New | Prediction |
| :--- | :--- | :--- | :--- | :--- |
| OFF | nothing | 0/4 | 0/4 | 0/8: held |
| ON | curiosity's verified note: the rule plus its 4 verified cases | 4/4 | 1/4 | 4/4 and ≥ 2/4: original held, **new missed** |
| ON_NO_SANDBOX | nothing was retrieved | 0/4 | 0/4 | ~0/8: held |
| PLAIN | the workspace's `.py` files, with the config files left out by a rule I wrote | 3/4 | 2/4 | ≥ 6/8: **missed** |

- **Every answer was graded by hand against computed values.** All 22 failures are real errors.
  - The checker's pass for PLAIN QB-1 is doubtful. Its stated answer is **5.00**, and 41.67 appears only inside a "wait, let me recompute" loop that was cut off at 320 tokens.
  - By the pre-registered checker the result is ON 5/8 and PLAIN 5/8. Read strictly, PLAIN is 4/8.
- **By the pre-registered rule (PLAIN ≥ ON), curiosity added no answer accuracy over plain access here.** The strict reading flips this by one question in a single greedy run of 8, which shows no difference either way.
- **The split is the finding.**
  - ON wins exactly where its note already holds the answer (original 4/4).
  - On new inputs, three of ON's four wrong numbers are stored values for *other* inputs: 90.91 and 11.11 for QB-3-NEW, and 11.11 for QB-4-NEW. The fourth is 33.33, from reading `//` as `/`.
  - PLAIN's errors are arithmetic, such as 10.0, 100.00 and 66.67. It made the same `//` misreading.
  - So the original A/B's 4/4 is reproduced and now attributed. It comes from retrieving precomputed verified cases, not from the model applying a learned rule.
- **ON_NO_SANDBOX shows the verification gate, not missing learning.**
  - In a regenerated run (below), this arm stored the same hypothesis as ON, with `epistemic_status` `hypothesis` and status `under_review`. Retrieval kept it out, so the model's answers were byte-identical to OFF on all 8 questions.
  - The gate kept out a hypothesis that was correct. This run cannot show what the gate prevents, because no wrong hypothesis was in play.
- **What curiosity did that PLAIN did not:**
  - It picked the material without a hand-written rule. No secret value from either config file reached any context, and the random-jitter distractor was not in the note.
  - The note survived a restart, since it was retrieved in a new process.
  - Only verified notes were used.
  - These are properties of the design seen in this run. They are not accuracy gains.
- **Recovered contexts.** Step 4 saved the answers but not the contexts. [step4_contexts.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/step4_contexts.py) regenerated them without the model ([results/step4_contexts.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/step4_contexts.json)). The idle actions, competence values and promoted ids are identical to step 4's record, so it is the same memory.
- **Found, not changed:**
  - The note's header says "answer from them; no tools needed" even when the note does not hold the asked case. Its closing instruction calls the notes "not verified facts", while the entry itself says "externally verified".
  - Whether the header contributes to the copying was not tested.
  - With no access (OFF), the model still gave confident, specific numbers and even an invented formula (`kcat / (K_m + [S])`) on all 8. It does not say when it lacks access.
- **Not covered by this run:**
  - One run, 8 questions, one model and two tiny modules.
  - Step 4 hands contexts straight to the model, so it says nothing about the bridge, the monitor or step 3's open question on answers.

**What this cycle taught me**
- I expected a 2B model to compute reliably from short code in front of it (PLAIN ≥ 6/8). It did not, and on new inputs neither arm beat 2/4. On this hardware and model, the binding limit for numeric questions about code is the model's arithmetic, not access.
- My first read of ON_NO_SANDBOX's 0/8 ("unverified hypotheses are kept out") was a prediction, not an observation. I checked the stored files before writing it down.

## 12. Fourth cycle (2026-11-09): Direction "if held bridge fix helps then go ahead apply it, 2 sure do it, 3 make the available in site link I will download it, fix the links"

**Links**
- Both broken links were typos: the Varela result link lacked `results/`, and the Santucci script link pointed at a results name.
- Fixed in `6fd336f1`. All three CI checks pass locally: `check-links.py` finds 0 broken links, `scripts/tests` passes 29/29, and `check-site.py` reports 0 errors.

**Item 2: questions about a verified workspace function**

Changes in `lfm3_layers.py`:
- **Wording.** The memory note header no longer says "answer from them". It still says "no tools needed", which guards against the file hunting recorded in the code. The instruction no longer calls every note "not verified facts". It now says "only a note marked externally verified was checked, and its cases hold only for the inputs shown".
- **Computation.** When a question literally calls the function that an externally verified note recorded, `run_verified_calls` runs the note's verified copy in `IsolatedPythonSandbox` (3 s limit).
  - Only literal arguments are accepted, and at most 4 calls.
  - The results go under the note ("Computed now by running that verified function: …") and into the memory layer's info (`computed`).
  - Nothing is added when there is no runnable Python, the sandbox fails, or a NameError shows the copy lacks a helper the module has.

Validation:
- `tests/test_verified_function_results.py` passes 13/13.
  - Liveness: a different stored body gives a different value.
  - Arguments that are not literals, unverified notes and a missing Python each add nothing.
- Live run: [item2_verified_notes_live.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/item2_verified_notes_live.py) → [results/item2_verified_notes_live.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/item2_verified_notes_live.json).
  - Predictions and the decision rule were written before the run.
  - It used step 4's workspaces, questions, model and system message.
  - Before any model call, the script confirmed three things: the memory equals step 4's, the new contexts differ from step 4's only in the two wording strings, and the computation only adds lines.

| Arm | Original | New | Notes |
| :--- | :--- | :--- | :--- |
| HEAD | 4/4 | 1/4 | Step 4's ON answers reproduced byte for byte (8/8), as predicted |
| Wording only | 4/4 | 1/4 | 7/8 answers byte-identical to HEAD; still copies stored values for other inputs, as predicted |
| Wording + computation | 4/4 | **4/4** | 33.0, 50.00 and 12.50, 20.0; every answer checked by hand, as predicted |

- **Controls** (3 ordinary user notes): every answer is byte-identical under the old and new wording. C3 failed under both: given "My manager is Priya." and "Correction: my manager is now Arun; …", the model answered Priya.
- **My decision rule was wrong.** It required 3/3 controls under the new wording, as an absolute count. The question it served was whether the wording breaks normal memory answers, so it should have been "no worse than the old wording". Read literally, it said revert both.
  - I did not overrule my own gate. I asked, and **the human chose to ship both**.
- **Found, not changed: corrections to user notes.** `remember()` stores "Correction: …" as a second `user stated belief` note and does not mark the earlier one superseded. The model then sees two managers and can pick the old one.

**Bridge fix: measured, then applied**

The setup ([bridge_fix_live.py](../../../research/consciousness-and-subjective-experience/18-clover-integration/bridge_fix_live.py) → [results/bridge_fix_live.json](../../../research/consciousness-and-subjective-experience/18-clover-integration/results/bridge_fix_live.json)):
- 20 real Studio turns: `OmniNative.run_turn` with memory, ontology, compiler and ncm, and the install's C2/C3 parts. The model was live LFM3-Omni-2B, greedy.
- Every turn started from a fresh memory and a fresh curiosity state, as on k3 and every new install.
- Comparison: HEAD (with item 2) against the same module with the one-line fix, in tools mode "ask" (the default) and "Always", routed by `semantics.route` as the server does.
- This run's permissions denied every permissioned tool, so nothing on this PC was read or run.
- Predictions and the decision rule were written before the run, and a dry run checked the setup first: all 5 questions retrieved their note, and HEAD's bridge reported `curiosity: null` every time while the fix's reported `should_observe: true`.

| Mode | Correct, HEAD | Correct, fix | What changed |
| :--- | :--- | :--- | :--- |
| ask | 4/5 | 4/5 | Nothing: no tool offer changed and all 5 replies are byte-identical (P1, P2 held) |
| Always | 4/5 | 5/5 | Offers changed exactly on B1 and B2 (calculate) and B4 (read_file) (P3 held) |

- **B1, compound interest.** HEAD answered 28,251, which is wrong. The fix offered the calculator, which returned 30626.075, and the model answered 30626.08 (P4 held).
- **B2.** Both arms were right; the fix's answer came from the calculator.
- **B4, worse with the fix even though the checker passed it.** HEAD answered "Your project plan is in `plan.md` in your Documents folder." With the fix, the model made an unneeded `read_file` call. It was denied here and the reply led with "I couldn't read the file". With default permissions (files are allowed by default), it would read a file the user did not ask for.
- **Decision rule** (FIX more correct AND no HEAD-correct turn broken): 9 vs 8, none broken. **Applied**, as directed ("if held bridge fix helps then go ahead apply it").
  - The held test became `tests/test_meaning_bridge_keeps_curiosity.py`. It passes 3/3, and 2 checks fail when the old line is put back.
  - Gate: 11 suites identical to the baseline (9 pass, the same 2 old failures), and the 6 new tests pass.
- **Not covered:**
  - One run of 5 questions on one model.
  - The unneeded file read on B4 is a cost in Always mode that the correctness checker does not see.
  - What curiosity's action boost does in C2 was not separately observed. It changed no reply in "ask" mode.
- **Found, not changed: Clover's arithmetic check garbles its feedback on numbers with thousands separators.**
  - On the "ask" B1 reply it did flag the wrong product. But it split "25,000" at the comma and read across a line break ("you wrote … = 25", "000 × 1.225043 = 0").
  - The repair round got unusable feedback, and the final reply still says 28,251.
