# Context: Follow-up to the review, items 1-4 (caution-only routing, plain-message benchmark, Dory parity, SuperAgent in Studio)

- **Date:** 2026-11-09 to 2026-11-10
- **Direction (verbatim):** "all 4 items have my approval go ahead do what needs to be done follow clover do them right do them step by step create a todo list and validate all the changes in the actual devices they are ready for you connection, finish the work validate and test them fully in s23 and laptop"
- **Items:** section 6 of [05-review-and-held-out-retest.md](05-review-and-held-out-retest.md).
- **Artifacts:** [05-plain-message-routing/](../../../research/superagent-jev-dory/05-plain-message-routing/PREREGISTRATION.md) (item 2), [06-dory-parity/](../../../research/superagent-jev-dory/06-dory-parity/PREREGISTRATION.md) (item 3).
- **Commits on main:** `301b21a6` (item 4), `269a038c` (item 1), `200585c0` and `41ab8cef` (two defects found on the devices).

---

## 1. Outcome

| Item | Outcome |
|---|---|
| 1. The decision engine never authorizes `act` | Done in code and tested. `superagent_jev_dory.should_inject_authoritative` never injects a confident `act` decision. `caution.apply` can only turn "proceed" into "ask" for an action. The routing reading itself is **not wired** into Studio, because item 2's gate failed. |
| 2. Test routing on plain, naturally ambiguous messages first | Done, pre-registered, 200 messages. **Gate FAIL.** The best eligible arm (LFM3-Omni-2B, ChatML, alpha 0.8, tau 0.9) adds caution on 18 of 75 non-act test messages. The gate needed 45. It never identified a `clarify` message (0/25) or a `verify` message (0/25). |
| 3. Check the Dory engine against an independent implementation | Done, pre-registered. **PASS:** at fp32 the standalone engine matches the llama.cpp Dory PR at K = 1, 2, 4. Both deliberate defects fail, as they should. bf16 is weaker: bf16 rounding alone flips the top-1 token on up to 6 of 24 prompts. |
| 4. Wire the SuperAgent into Studio | Done. Every tool call is a SuperAgent task. A file write completes only after Clover reads the file back. A call cut off by a stop is reported once on the next start. Validated on the laptop (Windows, Iris Xe) and the S23 (Android). Device testing found three defects, all fixed and re-validated (section 5). |

The devices run 5.0.2 **test builds**, built locally and not published. Publishing them needs a decision (section 7).

## 2. Item 1: caution only

- `superagent_jev_dory.should_inject_authoritative` (`269a038c`): a confident `act` returns `{"inject": False, "mode": "none", "why": "caution only: an act decision never authorizes an action"}`. A low-confidence decision still only raises uncertainty. Confident gather, clarify and verify decisions behave as before.
- `clover-ai-system/studio/clover/caution.py` (new) holds the routing reading and its policy:
  - The reading uses exact label log-probabilities from llama-server, in two passes (top-200, then a logit bias on the labels' first tokens). The two passes agreed within 7.8e-6 over 53-60 label pairs per arm.
  - `apply(frame, reading)` changes only one thing: an action that would run without asking now asks first. A question, an action that already asks, and an `act` reading are all left unchanged.
  - The model is not called from the turn path.
- Tests: `studio/tests/test_caution_only.py`, 16 checks, all pass. They also pass on the laptop's installed copy (Windows venv).

## 3. Item 2: plain-message routing benchmark (pre-registered)

**Design** ([PREREGISTRATION.md](../../../research/superagent-jev-dory/05-plain-message-routing/PREREGISTRATION.md)). Frozen before any run:
- `tasks.json` `1ae02981`: 200 plain messages, 50 per label, with no status sentence naming the missing piece and no label words. Hard negatives are marked.
- `split.json` `392d907c`: 25 test and 25 calibration messages per label.
- Pre-registration `4d0372c0`, harness `786f58f7`, `caution.py` `72c96b48`, Omni GGUF `6d2273a3`.

Selection used the calibration half only. The gate used the test half: G1 false caution <= 2/25, G2 caution recall >= 45/75, G3 caution added on at least half of the non-act messages where Studio does not already ask.

**Results** ([results.json](../../../research/superagent-jev-dory/05-plain-message-routing/results.json), [run.log](../../../research/superagent-jev-dory/05-plain-message-routing/run.log)). All 200 messages, tau 0.6 / 0.9.
- FC = false caution on act-gold messages (of 50).
- Recall = caution on non-act messages (of 150).
- 4-way = correct label (of 200).

| Arm | FC 0.6 | FC 0.9 | Recall 0.6 | Recall 0.9 | 4-way | clarify correct | verify correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| O1 Omni Markdown, alpha 0.8 | 15 | 2 | 81 | 13 | 54 | 0/50 | 0/50 |
| O2 Omni ChatML, alpha 0.55 | 10 | 7 | 78 | 63 | 79 | 0/50 | 1/50 |
| O3 Omni Markdown, alpha 0 | 13 | 1 | 98 | 39 | 68 | 0/50 | 0/50 |
| **O4 Omni ChatML, alpha 0.8 (selected, tau 0.9)** | 9 | 3 | 60 | 40 | 76 | 0/50 | 0/50 |
| H2 LFM3-1.7B ChatML, alpha 0.55 (reference, the study's "12/12" setting) | 1 | 1 | 21 | 3 | 89 | 0/50 | 4/50 |

- **Gate on the test half: FAIL.** G1 holds (2/25). G2 fails (18/75). G3 fails (18 of 75).
- **Where the caution comes from.** The Omni arms flag messages by reading them as `gather`, never as `clarify`. On the test half the selected arm predicted act 54, gather 37, clarify 3, verify 6.
- **Studio already asks** on 1 of the 200 messages (c33), in [baseline.json](../../../research/superagent-jev-dory/05-plain-message-routing/baseline.json).
- **Predictions.**
  - "The gate fails, G2 or G3 is the likely failure": held.
  - "Semantics already asks on fewer than 10 of 150": held (1).
  - "Markdown arms give less false caution than ChatML": **wrong**. The Markdown arms read most messages as `gather` (O1: 183 of 200), which costs false caution.

As pre-registered, the reading stays out of the turn path. Clover's own approval rules stay the only gate.

## 4. Item 3: Dory parity against the llama.cpp PR (pre-registered)

**Design** ([PREREGISTRATION.md](../../../research/superagent-jev-dory/06-dory-parity/PREREGISTRATION.md)). Frozen 2026-11-09 22:59:50 +05:30:
- Files: pre-registration `e2052d3c`, prompts `96113a60`, `run_parity.py` `f7f2caec`, `dory_logits.cpp` `17bf9cab`.
- Reference: the llama.cpp Dory PR at `1c3cdd23` ([pr_head.txt](../../../research/superagent-jev-dory/06-dory-parity/pr_head.txt)), with GGUFs converted from the same checkpoint (F32 `eec40092`, BF16 `12902e70`).
- Inputs: 24 fixed prompts, K = 1, 2, 4, cache `per_loop`, final-position logits over the full vocabulary.
- PASS rule: top-1 agreement on all prompts but at most one, median KL <= 0.01, and 90th-percentile KL <= 0.1.

**Results** ([results.json](../../../research/superagent-jev-dory/06-dory-parity/results.json)):

| Comparison | K=1 | K=2 | K=4 |
|---|---|---|---|
| PRIMARY: standalone fp32 vs llama.cpp F32 | PASS: 0 misses, median KL 3.4e-6 | PASS: 1 miss (`w3_easy_http`, near tie, 0.0052 nats), median KL 4.9e-6 | PASS: 0 misses, median KL 2.4e-6 |
| SECONDARY: standalone bf16 vs llama.cpp BF16 | PASS: 0 misses | INCONCLUSIVE: 3 misses | INCONCLUSIVE: 4 misses |
| CONTROL: standalone fp32 vs standalone bf16 | PASS: 0 misses | FAIL: 6 misses | INCONCLUSIVE: 4 misses |
| CONTROL: llama.cpp F32 vs llama.cpp BF16 | PASS: 0 misses | INCONCLUSIVE: 2 misses | PASS: 0 misses |

- **Liveness:** M1 (no input re-injection) FAIL, with 12/24 misses and median KL 0.93. M2 (swapped RoPE theta) FAIL, with 11/24 misses and median KL 0.70. The forward copy used for M1 differs from the engine by exactly 0.0. Verdict: **consistent with the reference**.
- **All three predictions held:** PRIMARY passes, both mutants fail, and bf16 differs more than fp32.

**The study's single-case claims, re-run on both implementations** ([claims.json](../../../research/superagent-jev-dory/06-dory-parity/claims.json)):
- **Verification routing.** `w7_hard_verify_route` and `dory_task_6_hard_agent_verify` are the same prompt in the study, so this is **one case**.
  - At fp32, P(verify) is 0.18 / 0.29 / 0.85 at K = 1 / 2 / 4 on both implementations (they differ by at most 0.0011). The answer changes from `clarify` to `verify` at K = 4. The direction of the study's claim holds.
  - Its numbers are bf16 numbers. The standalone engine in bf16 reproduces the study's 0.3045 at K = 1 exactly, but gives 0.8608 at K = 4, not 0.8317. llama.cpp BF16 gives 0.1732 at K = 1 and 0.8327 at K = 4.
- **Chain-of-thought case: not tested.** Every arm used the whole 200-token budget while still reasoning, so no arm reached an answer. The "final numbers" recorded are numbers from the reasoning text. The greedy texts do match across implementations: fp32 standalone and llama.cpp F32 agree at K = 1, 2, 4, while bf16 standalone and llama.cpp BF16 differ at K = 2 and 4.

## 5. Item 4: tool calls as SuperAgent tasks, and what the devices found

**What changed** (`301b21a6`):
- `clover-ai-system/studio/clover/tool_tasks.py` (new): `ToolTasks` wraps `CloverSuperAgent`.
  - Each tool call is a task, moving running -> verifying -> completed or failed. Each change is written as one JSON line, in `<base>/logs/tool-tasks/<device>-<date>.jsonl`.
  - A `write_file` completes only if the file reads back with the text that was written. The tool's own "written" is not enough.
  - Other tools complete on their own result, and the record says which kind of check it was.
  - Arguments are recorded with secrets redacted.
  - On the next start, a task left running is marked failed with "interrupted: Clover stopped while this step ran". The next answer then says once which step did not finish. Nothing is re-run automatically.
- `lfm3_native.execute_tool` opens and closes the task around every call. If a write does not read back, the model is told it was not done.
- The server adds `GET /api/tasks` and appends the note after the turn.

**Defects the devices found** (all fixed, re-built, re-installed and re-validated):

1. **The model repeated the note** (laptop, first 5.0.2 build). Turn 2 answered "(Note: Clover last stopped with the run_python step, ...)" in its own words, because the previous answer, note included, was in its history. Fix `200585c0`: the note stays in what the person sees and in the saved conversation. It is removed from every copy the model reads (live session, saved conversation, browser seed).
2. **The 5.0.0 and 5.0.1 phone builds carry a C2 mismatch** (found on the S23 with the first 5.0.2 build). The turn ended with "TypeError: OmniC2C3Runtime.c2_plan() got an unexpected keyword argument 'action_prior_8'".
   - **Cause.** k3's Android build tree had five files older than the repository: `omni_c2_c3.py`, `jepa_engine.py`, `workspaces.py`, `autonomous_clover.py` and `memory-knowledge-graph.py`. Each equals an earlier repo commit (7-8 Nov). Meanwhile `lfm3_layers.py` and `omni_native.py` there are current.
   - **Published releases.** The 5.0.0 phone APK (sha `987c13ea`, also the one from the site link) and the published 5.0.1 APK (sha `512dcc03`) carry the same mismatch, read from their bundled Python.
   - **Why it ends the turn.** `OmniNative.run_turn` calls `LayerKit.c2_plan_state` at the end of every turn whenever the C2 runtime is loaded, and that call has no handler of its own. I ran one turn on the mismatched build. I did not run 5.0.0 or 5.0.1 on a phone.
   - **Fix.** The five files were synced into k3's tree, with a guard and a backup. They are the versions the laptop runs. `k3_host_engine.py` and `lfm3_native.py` are deliberate local variants and were left as they are.
3. **"Checked on disk" was invisible on the phone** (S23). The step line is cut off long before its end, even in landscape. Fix `41ab8cef`: a write that reads back now says "✓ Checked on disk" in the step's badge.

**Device validation, final builds.** Setup `d1d6fcdc…`, signature Valid on the build PC. APK `5026958a…`, versionCode 16.

| Check | Laptop (Windows, Iris Xe, API) | S23 (Android, Adreno, UI) |
|---|---|---|
| Write a file | Task completed with evidence "file read back", 18 chars, sha `28e44197`. Step summary ends "· checked on disk". File holds the text. | Step badge "✓ Checked on disk" (screenshot). A separate read-back turn ran `read_file` and showed the text. |
| Stop Clover mid-call | 150 s `run_python` shown as running. Server process killed. On restart `/api/tasks` lists it as interrupted. | 120 s `run_python` showing "Running…". App force-stopped and relaunched. |
| Next answer | "2 + 2 = 4" plus the note, with the tasks layer | "The result of 2 + 2 is 4." plus the note |
| Answer after that | "3 + 3 = 6." (no note, nothing repeated) | "3 + 3 equals 6." (no note) |
| Another restart | Nothing listed as interrupted. The log records the call as failed, "interrupted: Clover stopped...". | "The result of 5 + 5 is 10." (no note) |
| New tests on the installed code | `test_caution_only` 16/16, `test_tool_tasks` 17/17 (before the badge check existed) | not runnable (release build) |

Repository suites: `test_tool_tasks` 18/18 and `test_caution_only` 16/16 pass. The failures already present are unchanged: `test_regression_fixes` (2 checks), `test_memory_index` (`engine_kind`), and `test_browser_first` (1 check). `test_browser_first` reads only `app.html` lines this work did not touch.

## 6. Not verified (named gaps)

- **On the laptop the badge was not seen.** Its installed page is the same file as the one seen on the S23, line endings aside (`0edeb2a8`), but no screenshot of the laptop's UI was taken.
- **On the phone the task log was not read directly.** It sits in the app's private storage (release build). The badge is the evidence, because the page shows it only when the server's task completed with "file read back".
- **The laptop installer's signature** reads "UnknownError" in the laptop's ssh session: the self-signed root is not trusted in that context. The thumbprint matches (`843BC847…`). It reads Valid on the build PC.
- **Device agreement for the routing reading** (pre-registered step 3) was not run, because the gate failed first.
- **The Dory CoT claim** needs a larger generation budget (section 4). The adaptive loop policy has not been re-tested on held-out tasks.
- **The k3 web studio was not updated.** It is a production deployment.

## 7. For Direction

- **Publishing 5.0.2.** The phone fix in section 5 (2) is the strongest reason to publish soon: published 5.0.0 and 5.0.1 phone apps carry the mismatch. Publishing needs approval. Still to do:
  - the GitHub release
  - site download links
  - CHANGELOG
  - store notes
  - MSIX
  - regenerated release reports
- **k3's Android build tree is synced by hand and had drifted.** Five files were stale. A scripted full sync with a list of deliberate variants would prevent a repeat.
- **The routing reading stays off.** Any later attempt needs a model that can identify `clarify` on plain messages, and a fresh pre-registered gate.
- **Dory loops.** The engine can now be trusted at fp32. A loop policy still needs held-out tasks with the probe counted, and bf16 results should not be compared to the third decimal.

## 8. What changed

- **main:** `301b21a6` (tool tasks, `tool_tasks.py`, hooks, `/api/tasks`, tests), `269a038c` (caution only, `caution.py`, tests), `200585c0` (note kept out of model history), `41ab8cef` (badge). Research folders `05-plain-message-routing/` and `06-dory-parity/`, and this record.
- **Local branch `release/5.0.2`**, not pushed: version bump `d89874bb` plus cherry-picks `9cbec6bc` and `f066aad1`. Release notes, CHANGELOG and store notes are not updated.
- **k3 Android build tree** (`/opt/clover-k3/studio-android`). Each step was guarded by hashes and backed up first:
  - items 1-4: 9 files (`backup-items14-20261109T181455Z`)
  - note fix: 2 files (`backup-notefix-20261109T182810Z`)
  - stale-file sync: 5 files (`backup-c2sync-20261109T230009Z`)
  - badge: 1 file (`backup-badge-20261109T232413Z`)
  - `build.gradle` set to versionCode 16 / 5.0.2
- **Laptop:** 5.0.2 test build installed. The test sessions and files were deleted, and the test server was stopped (Studio was not running before). The tool-task logs stay in `logs/tool-tasks`. The Setup is in Downloads.
- **S23:** 5.0.2 test build installed. Five test chats remain in the chat list, and three test files remain in the app's storage. "Stay awake while charging" was on during the test and is back to off.

## 9. Growth

- **Run every stage on the device, not only the stage you changed.** Item 4 worked on the laptop. On the phone the turn died one stage later, in code this work never touched, because the phone's build tree had drifted from the repository.
- **One stale file means survey the whole tree.** The crash named one file. The survey found five.
- **Measured in an API is not seen by a person.** The summary carried "checked on disk" on both devices, but no phone screen ever showed it.
- **Text shown once still sits in the model's context.** A note appended to an answer is read back by the model on the next turn, unless it is removed from the model's copy.
- **A pre-registered prediction can fail usefully.** The Markdown-template prediction was wrong, and the reason (most messages read as `gather`) is the clearest picture of how this model behaves on plain messages.
