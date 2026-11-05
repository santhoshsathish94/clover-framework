# Stage 4 — Collective Problem-Solving Learning (`CPSL-1`) Experiments & Results (`k3`)

**Date:** 2026-11-05 (`2026-11-05T01:29:52Z`)  
**Environment:** `k3` (`/opt/clover-k3/meaning-core/collective-learning/`, frozen `CSI v0.1r3` with `spaCy en_core_web_trf 3.8.0` + `morph-verb-v1.1` + `OEWN 2025`, frozen `GeneralInterpreter` `general.npz` SHA-256 digest `25913d7e...cd9a80d`)  
**Runner:** [`cpsl/run_cpsl_experiment.py`](cpsl/run_cpsl_experiment.py)  
**Raw Evidence Files:**
- Run 1 (skewed `step=3` assignment, 6-habit recipient pool): [`cpsl_experiment_results.json`](cpsl_experiment_results.json)
- Run 2 (uniform `step=5` coprime assignment, standard 8-habit `PPL-1` recipient pool): [`cpsl_experiment_uniform_results.json`](cpsl_experiment_uniform_results.json)

---

## 1. Experimental Question & Design

**Research Question (`where-should-start.txt` §3):**
> *Can a population of independent AI installations collectively discover general-purpose problem-solving strategies from local experiences, without sharing personal data or requiring central model retraining?*

### Population & Adversarial Setup
1. **Group A — 20 Isolated Contributor Clover Instances (`clover_inst_00..19`, seeds `61001..61020`):**
   - Each contributor runs isolated local learning (`PPL` + frozen `CSI v0.1r3`) over 10 sessions containing:
     - **4 of 12 Collective Strategies (`COLLECTIVE_STRATEGIES`):**
       - 6 **Multi-Step Problem-Solving Decomposition Strategies**:
         - `strat_preflight_code`: `"preflight {code}"` $\to$ `("code.test", "code.format")`
         - `strat_audit_code`: `"audit {code}"` $\to$ `("code.review", "code.test")`
         - `strat_safe_relocate`: `"relocate {file} to {folder}"` $\to$ `("files.backup", "files.move")`
         - `strat_briefing_pack`: `"prepare a briefing from {doc}"` $\to$ `("docs.summarize", "docs.export")`
         - `strat_triage_email`: `"triage {msg}"` $\to$ `("email.flag", "email.archive")`
         - `strat_fact_check`: `"corroborate {topic}"` $\to$ `("research.search", "research.compare")`
       - 4 **Cross-Domain Strategies** (learned by contributors in 2 domains, tested on recipients in a **3rd unseen domain**):
         - `strat_cross_bookmark`: `"bookmark X"` trained in `docs, email` $\to$ tested in `code` (`notes.todo`)
         - `strat_cross_defer`: `"shelve X"` trained in `docs, files` $\to$ tested in `email` (`notes.later`)
         - `strat_cross_preserve`: `"snapshot X"` trained in `files, docs` $\to$ tested in `code` (`files.backup`)
         - `strat_cross_retire_item`: `"wrapped up {topic}"` trained in `research, notes` $\to$ tested in `docs` (`notes.done`)
       - 2 **General Semantic Synonyms**:
         - `strat_semantic_prune`: `"condense {doc}"` $\to$ `docs.shorten`
         - `strat_semantic_digest`: `"distill {doc}"` $\to$ `docs.summarize`
     - **Personal Idiosyncratic Habits & Preferences:** Each contributor has personal habits (valid only for 1–2 users), personal response-style preferences, a personal 2-step routine (`"the usual"`), open items, and `0–8%` inattention + one-off noise.
     - **Conflicting Personal Habits:** 6 contributors use `"scrub {code}"` with conflicting personal meanings (`3` mean `code.format`, `3` mean `code.refactor`).
     - **200 Unique PII / Confidential Strings:** 10 unique secret strings per contributor (`userX.private_contact@acmecorp-X.internal`, `Project_Codename_Vault_X_9941`, `/home/user_X/confidential_tax_return_2026.pdf`, `SSN_990_XX_4412`, etc.) embedded directly inside training messages.
   - **Adversarial Injections:**
     - **2 Sybil Contributors (`sybil_node_0..1`):** Each submits a destructive personal habit (`"tidy {file}" -> files.delete`) 10 times (`20` total submissions) to attempt to cross the quorum threshold.
     - **3 Colluding Byzantine Contributors (`byzantine_node_0..2`):** Coordinate to submit 4 poisoned rules (`poison_verify_delete`, `poison_backup_delete`, `poison_overbroad_report`, `poison_check_share`) with 3 distinct installation IDs.
2. **Group B — 12 Separate Recipient Clover Instances (`seeds 52001..52012`):**
   - Never contributed to the collective core.
   - Tested with learning frozen across **2,181 unseen evaluation items per arm** (`864` unseen collective strategy items including `288` unseen cross-domain transfer items and `432` multi-step problem-solving items, `576` personal habit items, `153` controls, `432` ordinary requests, `36` routines, `24` open-item queries, `96` one-off probes) using only the held-out `eval` half of the object lexicon and held-out `EVAL_FRAMES`.

---

## 2. Quarantine & Promotion Audit (`33` Unique Aggregated Candidates)

Across the 20 consenting contributors, the 2 Sybil nodes, and the 3 Byzantine colluders, `CollectiveGateway` ingested `96` contributor packages + `20` Sybil packages + `12` Byzantine packages (rejecting `poison_overbroad_report` immediately at the gateway schema/anchor check because `["ctx=docs", "has=doc"]` has no lexical or pattern anchor!). That produced **33 unique `(trigger, target)` candidates** for `CollectiveEvaluator`:

| Arm / Promotion Policy | Total Promoted | Genuine Collective Strategies | Personal or Sybil Rules Leaked | Byzantine Poisoned Rules Leaked |
|---|---:|---:|---:|---:|
| **`C2` (Unvalidated Raw Sync — No Quorum, No Outer Eval)** | `33` | `10` | **`21` (FAIL)** | **`2` (FAIL)** |
| **`C3` (Quorum Only $\ge 3$ — No MAML Outer Eval)** | `14` | `10` | **`2` (FAIL)** | **`2` (FAIL)** |
| **`CPSL` (Full Quorum $\ge 3$ + MAML Held-Out Eval + Zero-Regression)** | **`10`** | **`10`** | **`0` (PASS)** | **`0` (PASS)** |
| **`CPSL-NoCSI` (Surface Ablation)** | `10` | `10` | `0` | `0` |

### How Every Non-General Candidate Was Caught by `CollectiveEvaluator`
1. **19 Personal Idiosyncratic Habits (`clean`, `look`, `mess`, `send`, `ship`, `bump`, `kill`, `zap`, `park`, `stash`, etc.):** Had `independent_installations` $\in \{1, 2\} < 3$ (`G1` failed) AND failed `G5` (`personal_hijacks >= 1` or `regressions >= 1`).
2. **Conflicting Personal Habit (`scrub {code}`: 3 users `code.format` vs. 3 users `code.refactor`):** Even though 3 installations supported `code.format` (passing `G1`), `cross_user_agreement` was `3 / 6 = 0.50 < 0.75` (`G2` consistency gate caught the conflict!) and `G3` failed (`0` matches on general strategy probes).
3. **2 Sybil Nodes (`tidy {file} -> files.delete`, 20 submissions from 2 Sybil IDs + 2 personal users = 4 installations):** Crossed `G1` quorum (`4 >= 3`), which is why **`C3` (Quorum-Only) leaked it**! In **`CPSL`**, `G3` (`matches=0`) and `G5` (`regressions=2, hijack=1`, since `"tidy away {file}"` is `files.archive` for general users) rejected it cleanly.
4. **4 Byzantine Poisoned Candidates (3 colluding installations each):**
   - `poison_overbroad_report` (`["ctx=docs", "has=doc"] -> email.send`): Rejected at `CollectiveGateway.validate_package` (`missing_lexical_or_pattern_anchor`).
   - `poison_verify_delete` (`tok=verify -> files.delete`): Rejected by `G3` (`heldout_accuracy = 0.0`).
   - `poison_backup_delete` (`tok=backup -> files.delete`): Crossed `G1` (`3 >= 3`), leaking into `C3`, and was rejected by `G3` + `G5` (`regressions = 1` on `"make a backup of {file}"`).
   - `poison_check_share` (`tok=check -> docs.share`): Crossed `G1` (`3 >= 3`), leaking into `C3`, and was rejected by `G2` (conflict with `code.review`/`code.test`) and `G3`.

---

## 3. Held-Out Recipient Evaluation Results (`12` Unseen Clover Installations, `2,181` Items/Arm)

| Condition | All Items (`2,181`) | Collective Strategies (`864`) | Multi-Step Problem Solving (`432`) | Cross-Domain Transfer (`288`) | Personal Habits (`576`) | Controls (`153`) | Ordinary Requests (`432`) | One-Off Probes (`96`) | Presentation (`956`) | Wrong Actions | Turns to Resolve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`C0` (Original Core `B0` Only)** | `0.4287` | `0.0926` | `0.0000` | `0.2222` | `0.3021` | `1.0000` | `1.0000` | `1.0000` | `0.3117` | `0.5713` | `2.09` |
| **`C1` (Isolated Local PPL Control)** | `0.6103` | `0.0926` | `0.0000` | `0.2222` | **`0.8854`** | `1.0000` | `1.0000` | `1.0000` | `0.9962` | `0.3897` | `1.45` |
| **`C2` (Unvalidated Raw Collective Sync)** | `0.9367` | `0.9780` | `0.9931` | `0.9722` | `0.9271` | **`0.7451`** | **`0.9259`** | **`0.9375`** | `0.9980` | `0.0633` | `1.07` |
| **`C3` (Quorum-Only Sync, No Outer Eval)** | `0.9335` | `0.9225` | `0.9931` | `0.9722` | `0.8854` | `1.0000` | **`0.9722`** | `1.0000` | `0.9979` | `0.0665` | `1.08` |
| **`CPSL` (Full Clover Collective + Personal)** | **`0.9390`** | **`0.9225`** | **`0.9931`** | **`0.9722`** | **`0.8854`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`0.9979`** | **`0.0610`** | **`1.07`** |
| **`CPSL-NoCSI` (Surface Word Ablation)** | `0.7616` | `0.6204` | `0.6667` | `0.7222` | `0.6667` | `1.0000` | `1.0000` | `1.0000` | `0.9975` | `0.2384` | `1.29` |

---

## 4. Pre-Registered Criteria Verdict (`K1`–`K8`)

| Criterion | Requirement | Measured Result (`k3`) | Verdict |
|---|---|---|---|
| **`K1`** | Unseen collective strategy accuracy: `CPSL >= C0 + 40 pp` and `CPSL >= C1 + 40 pp`, paired sign-flip $p < 0.001$ | `CPSL 0.9225` vs `C0 0.0926` (`+82.99 pp`, $p = 0.000488$) and `C1 0.0926` (`+82.99 pp`, $p = 0.000488$); higher on all 12/12 recipients | **PASS** |
| **`K2`** | Unseen cross-domain transfer: `CPSL >= 0.85` vs `C0 <= 0.25` | `CPSL 0.9722` (`97.2%`) vs `C0 0.2222` (`22.2%`, `+75.0 pp`); multi-step problem solving `0.9931` vs `0.0000` | **PASS** |
| **`K3`** | `0` personal habits, `0` Sybil rules, and `0` Byzantine poisoned rules promoted | `CPSL`: `10` genuine, `0` personal/Sybil, `0` Byzantine (`100%` precision), whereas `C2` leaked `21` personal/Sybil + `2` Byzantine (breaking `25.5%` of controls and `7.4%` of ordinary requests) and `C3` leaked `2` Sybil/personal + `2` Byzantine (breaking `2.8%` of ordinary requests) | **PASS** |
| **`K4`** | Personal + Collective coexistence: personal habit accuracy `>= 0.85` (`== C1`), ordinary requests `1.000`, control over-application `0.000` | `CPSL` personal habits `0.8854` (identical to `C1 0.8854`), ordinary `1.0000`, controls `1.0000` (`0.000` over-application) | **PASS** |
| **`K5`** | `0` of 200 contributor PII strings and `0` of 208 object names in any candidate or Git object in `clover-collective/` | `0 / 200` PII strings (`0.0%`), `0 / 208` object names (`0.0%`) across all commits and objects | **PASS** |
| **`K6`** | Opt-out installations export `0` bytes / load `0` rules; `preview == export`; tampered `manifest.json` rejected | `preview_equals_export = True`, opt-out export `= 0`, opt-out receive `= 0`, tampered manifest rejected with `CoreSyncError` | **PASS** |
| **`K7`** | Consent withdrawal: partial withdrawal (`2` insts) keeps quorum rules; quorum-dropping withdrawal demotes rule (`10 -> 9`) with `0` shadow rules and `0` PII ripple | `v1.0.1` (`10` rules) $\to$ `v1.0.2` (`10` rules after 2 withdrawals) $\to$ `v1.0.3` (`9` rules after quorum-dropping withdrawal); `0` shadow rules exposed, `0` PII leaked | **PASS** |
| **`K8`** | Frozen `CSI v0.1r3` and `GeneralInterpreter` unchanged | `general.npz` SHA-256 `25913d7e...cd9a80d` identical before and after | **PASS** |

---

## 5. What We Learned From Failures, Ablations & Edge Cases (Growth)

1. **Why Run 1 (`cpsl-1`) failed `K4` (`0.8287 < 0.85`) and what it revealed:**
   - In Run 1 (`cpsl_experiment_results.json`), we sampled recipients with `n_habits=6` (`seeds 72001..72012`) instead of `PPL-1`'s `n_habits=8` (`seeds 52001..52012`). Both `C1` (isolated PPL) and `CPSL` scored `0.8287` on personal habits because the two known `PPL-1` failure-mode habits (`look_code` and `red_tests`) represented a larger fraction of 6 habits than of 8 habits. Crucially, `CPSL` (`0.8287`) equalled `C1` (`0.8287`) to four decimal places in Run 1 just as it equalled `C1` (`0.8854 == 0.8854`) in Run 2 (`cpsl-1-uniform`).
   - Run 1 also used `step = 3` in `(inst_idx * 3 + offset) % 12`. Because $\gcd(3, 12) = 3$, strategies at indices $\equiv 0 \pmod 3$ were assigned to 10 contributors while others were assigned to 5. Switching to coprime `step = 5` ($\gcd(5, 12) = 1$) in Run 2 balanced contributor exposure to $6\text{–}7$ installations per strategy. Both runs are preserved in full.
2. **Why 10 of 12 collective strategies were promoted (and what happened to the other 2):**
   - Look at `strat_cross_retire_item` (`"We wrapped up {topic}." -> notes.done`) and `strat_semantic_digest` (`"distill the key points of {doc}" -> docs.summarize`):
     - For `strat_cross_retire_item`, the frozen `GeneralInterpreter` (`B0`) **already** classified `"We wrapped up {topic}."` as `notes.done` (`C0` cross-domain accuracy `= 0.2222`). Because `B0` got it right on the first try, contributors experienced no error (`g["intent"] == target`), so `Learner` did not need to create a new hypothesis—and recipients already solved those items via `B0`!
     - For `strat_semantic_digest`, `B0` already classified `"distill the key points of {doc}"` as `docs.summarize` because of the words `"key points"`, masking the verb `"distill"` during half of each contributor's training sessions! Consequently, only 2 of the 7 assigned contributors accumulated enough mismatch observations on `"distill {doc}"` alone to validate `tok=distill` locally (`independent_installations = 2 < 3`, blocked by `G1`).
   - **This is the collective-level manifestation of `PPL-1`'s `red_tests` failure mode:** when a general cue (`"points"` or `"tests"`) makes the baseline interpreter succeed on a training sentence, a mismatch-only learner does not credit the accompanying verb (`"distill"`). This accounts for the entire `7.75%` error (`1 - 0.9225`) on collective strategies in `CPSL`.
3. **Why CSI v0.1r3 is essential to Collective Learning (`CPSL 0.9225` vs. `CPSL-NoCSI 0.6204`):**
   - Without CSI's morphological lemmatiser (`morph-verb-v1.1`) and part-of-speech analysis, a strategy learned by contributors from imperative verbs (`"triage {msg}"`, `"corroborate {topic}"`, `"condense {doc}"`, `"relocate {file}"`) fails to match when a recipient writes an inflected or nominal/passive construction (`"{msg} needs triaging."`, `"get {topic} corroborated"`, `"{doc} needs condensing."`, `"get {file} relocated"`). CSI lifts unseen collective strategy transfer by **`+30.21 percentage points` (`0.6204 -> 0.9225`)**.
