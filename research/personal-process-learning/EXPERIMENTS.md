# Personal Process Learning: experiments

## PPL-1 - can Clover learn how one person communicates and works, and help that person better in fresh interactions?

Question (human, 2026-11-03): "Can Clover learn how one individual communicates and works, then demonstrably improve its
assistance for that individual in future, previously unseen interactions?" Tested "with controlled interactions first,
using separate training experiences and genuinely fresh evaluation scenarios", measuring "whether personalization
improves interpretation and assistance without simply memorizing previous conversations".

### Design (fixed before the test; code in `ppl/`, runner `run_ppl.py`)

**World** (`world.py`). 7 areas, 41 intents (each with slot types, the presentation dimensions that apply, and the
general phrasings anyone uses), 6 presentation and working-style dimensions (format, length, ask before acting,
plan first, number of options, sources; first value = general default), object lexicons split in halves (persons see
one half in training and the other in evaluation), request frames split the same way.

**General interpreter** (`general.py`). Logistic regression over CSI features (frozen CSI v0.1r3: content lemmas,
predicate, mood, polarity; object kinds; area), trained once on population data: 3,648 messages - every general
phrasing with objects from both halves and all frames, plus how anyone asks for a different form or states a
preference. It is shared and never updated by a person. Held-out accuracy 1.000 on 566 new population messages (the
same phrasings with new objects and frames: an in-distribution check only). Digest `25913d7e...cd9a80d`.

**Persons** (`personas.py`, `simulator.py`). Each simulated person has 8 habits from one pool - expressions they use
with a personal meaning in some areas (`tidy` for code means *format*; `kill` an email means *archive*; `the proposal
-> PDF` means *export*; `X is way too long` means *shorten*; `park` means *add to my todo list* in any area) - plus
preferences (each dimension differs from the default with probability 0.5), a mix of correction styles (explicit "No,
I meant: ...", terse "No. ...", vague "That's not what I meant.", rephrasing), politeness, a propensity to state a
habit or preference outright, one routine ("the usual" = two steps), and four topics to come back to later.
Noise (added after development run 1, below): inattention - a wrong answer accepted without noticing, rate uniform in
[0, 0.08] per person - and four one-off meanings: an ordinary phrasing that once meant something else and never again.

**Habit pools.** Development: tidy_code, check_code, send_docs, bump_email, ship_code, park, zap, mess_code, see_docs,
clash_cal, arrow_export, arrow_file. Test: tidy_files, clean_code, look_code, kill_email, polish_docs, ship_docs,
stash, ptal, long_docs, red_tests, answer_email, arrow_move, arrow_share, fyi_docs. Both were written before any run;
no interaction with a test-pool person has been run (a unit check only confirms that a sampled test person's habits
come from the test pool).

**Training** (`scenarios.py`), 10 sessions per person: 7 messages per habit (training phrasings, frames, objects),
2 per control (the same expression in an area where the person means what everyone means), 60 ordinary requests,
4 routine requests, 4 "come back to this later" and 2 "that's done", occasional explicit statements, 4 one-offs.

**Evaluation** - fresh scenarios, learning frozen, the first response scored: 6 per habit in held-out frames,
constructions (`give main.go a tidy`, `main.go needs tidying`), objects, and, for area-general habits, new areas;
3 per control; 36 ordinary requests; 3 routine requests; 2 "what's still open?"; 2 probes per one-off (the same
phrasing, meant as everyone means it). No evaluation habit message occurs in training (checked by the tests).

**Interaction.** Clover responds (intent, steps or open items, and a presentation) or asks; the simulated person
accepts, corrects in their style, answers the question, or asks for a different form; up to 6 turns. Clover sees only
text. Ground truth is used only by the simulated person and for scoring.

**Conditions** (`assistant.py`).
- B0: general interpretation only, default presentation.
- B1: B0 + explicit statements remembered and applied (today's "remember that ..."), open items kept on request.
- B2: B0 + conversation memory: the most similar past message (TF-IDF over words and area, threshold tau) decides;
  the last accepted form of each dimension is reused; open items kept. This is the memorising baseline.
- PPL: B0 + Personal Process Learning (`learner.py`; ARCHITECTURE.md section 4).
- Ablations: PPL-surface (personal layer reads surface words, not CSI lemmas), PPL-noval (applies what it saw once,
  never retires), PPL-noctx (hypotheses ignore the area they were learned in).

**Retention** (`store.py`, `evaluate.retention_test`). For every test person, after evaluation: stop learning from all
experiences of the habit with the most validated knowledge, then make a backup clone, then delete all experiences of
the next such habit (files, all git history via git-filter-repo, reflog, `gc --prune=now`; knowledge history
rewritten; recomputed from the remaining eligible experiences).

### Development (6 persons, seeds 1001-1006, development pool)

Selection rules, fixed before development: each condition gets its own best settings by overall first-response
accuracy; PPL subject to over-application at most 10% on controls and on one-off probes; ties to the first in grid
order. Grid: C (ask below this general confidence) 0 / 0.3 / 0.5; K (support to validate) 1 / 2 / 3; P (precision to
validate) 0.7 / 0.9; provisional hypotheses applied or not; tau 0.1-0.7 (0.3-0.7 in run 1).

**Run 1** (no noise; `results/dev_run1_table.json`):

| condition | all | habit | control | control over-applied | generic | routine | open items | presentation |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.667 | 0.413 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.614 |
| B1 | 0.757 | 0.573 | 1.000 | 0.000 | 1.000 | 0.000 | 0.667 | 0.736 |
| B2 (tau 0.3) | 0.772 | 0.583 | 0.984 | 0.000 | 0.949 | 1.000 | 0.667 | 0.981 |
| PPL (K 1, P 0.7, provisional applied) | 0.990 | 0.993 | 1.000 | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 |
| PPL-surface | 0.868 | 0.740 | 1.000 | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 |
| PPL-noval | 0.990 | 0.993 | 1.000 | 0.000 | 1.000 | 1.000 | 0.667 | 1.000 |
| PPL-noctx | 0.950 | 0.993 | 0.667 | 0.333 | 0.986 | 1.000 | 0.667 | 1.000 |

What run 1 showed, and what changed before registration:
1. All 12 PPL settings scored the same: in a world without noise nothing a person says once is misleading, so
   "retain only with sufficient evidence" could not be tested. Inattention and one-offs were added (above).
2. Open items were 0.667 for every condition that keeps them: an object written with a capital at the start of a
   sentence ("Quantum dots is done.") was not matched to the open item. Fixed (objects are compared by canonical name).
3. B2's best threshold was the lowest in the grid, so the grid was extended down to 0.1.

**Run 2** (with noise; final code; `results/dev_run2_table.json`):

| condition | all | habit | control | control over-applied | one-off probe | one-off over-applied | generic | routine | open items | presentation |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.691 | 0.413 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.630 |
| B1 | 0.781 | 0.573 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.747 |
| B2 (tau 0.3) | 0.755 | 0.590 | 0.984 | 0.000 | 0.458 | 0.521 | 0.940 | 1.000 | 1.000 | 0.982 |
| **PPL (K 2, P 0.7, provisional not applied)** | **0.997** | **0.993** | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PPL-surface | 0.879 | 0.729 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| PPL-noval | 0.910 | 0.993 | 0.952 | 0.000 | 0.125 | 0.875 | 0.949 | 1.000 | 1.000 | 1.000 |
| PPL-noctx | 0.960 | 0.965 | 0.810 | 0.190 | 1.000 | 0.000 | 0.995 | 0.833 | 1.000 | 1.000 |

PPL grid at C = 0: applying provisional hypotheses over-applies one-offs (0.708) whatever K; K = 1 validates them
(0.417); P = 0.9 loses habits to inattention (0.924 with K 2). C = 0 was best for every condition (asking never
improves a first response). B2 at tau 0.2 has higher habit accuracy (0.736) but lower overall (0.696). The two
remaining PPL habit errors are "index.html could use some tidying." - CSI gives the noun *tidying* its own lemma.

Chosen (`results/params.json`, sha256 `507b4383...4fd94bb18f`): B0 C 0; B1 C 0; B2 C 0, tau 0.3; PPL C 0, K 2,
P 0.7, provisional not applied (R 2, retire on, preferences after 1 observation at precision 0.6); ablations use
PPL's settings except what they remove.

**Dry run** of the whole test procedure on the development persons (`results/dryrun_dev_result.json`): every step
ran; deletion left 0 of the deleted strings in any store and 47 in the backup clones (reported, as designed);
behaviour on stopped and deleted habits returned to general interpretation (1.000); rebuild from the store equalled
live knowledge for every person; no leakage; CSI and the general interpreter unchanged. (L1's p-value cannot go
below 0.03 with 6 persons; it can with 12.)

### Pre-registered test

- Persons: 12, seeds 52001-52012, test pool only, all conditions above, plus (sensitivity only, not a criterion) B2
  at every tau in the grid.
- Settings: `results/params.json` as chosen. Code: the commit that adds this section; the files run on k3 are checked
  against it by sha256 before the run.
- Integrity values before the run (from the dry run, re-measured by the test itself): general interpreter digest
  `25913d7e...cd9a80d`; CSI code `567caad4...3455b932`, frozen artefacts `61847d6b...3708ecb3`, CSI readings of the
  probe sentences `6555b317...73a27384`.
- Statistics: per-person paired sign-flip test (exact, two-sided) on first-response accuracy.

Criteria (all must pass):

| | criterion |
|---|---|
| L1 | habit items: PPL >= B0 + 20 points and >= B2 + 10 points, both p < 0.01 |
| L2 | all fresh items: PPL >= B0 + 10 and >= B2 + 5 |
| L3 | over-application at most 10% and no more than B2, on controls and on one-off probes |
| L4 | ordinary requests: PPL >= B0 - 2 |
| L5 | presentation adherence (where the intent was right): PPL >= B0 + 25 and >= B1 + 10 |
| L6 | routines >= 80% and open-item F1 >= 0.8 |
| L7 | no difference between persons trained alone and trained interleaved in one process; CSI and the general interpreter unchanged |
| L8 | for at least 90% of persons the stopped and deleted habits had validated knowledge; no knowledge references a stopped or deleted experience; no deleted string remains anywhere in the store's repository; behaviour on those habits returns to general interpretation >= 95%; stopped records all kept; rebuild equals live knowledge |

Decision: all pass - the question is answered yes *for this controlled setting*; next step (with the human's
approval) is Phase 2: runtime integration, off by default, and a consented pilot with real use. Any fail - report
which and why; no integration.

Known limitations, stated before the test:
1. One author wrote the world, the persons, both habit pools and the learner. Habits are anchored in a word or a
   pattern within an area, which suits a learner that generalises over lemmas. The test pool is new to the learner
   but not to its author.
2. Fresh phrasings keep the habit's word (in other forms and constructions); a habit expressed with entirely
   different words is out of scope here and would not be recognised.
3. Simulated corrections use general phrasings; real corrections are messier. Preferences are per person, not per
   situation; routines have two steps.
4. The general interpreter is trained on synthetic phrasings; its held-out accuracy is in-distribution.
5. CSI's part in the personal layer is lemmas and parts of speech; the size of the CSI ablation effect depends on how
   often fresh phrasings inflect the habit word, which the world's author chose.
6. Noise levels (inattention up to 8%, 4 one-offs) were chosen by the author after development run 1.
7. The leakage check looks for hidden shared state in one process; stores are separate by construction.
8. Deletion reaches the person's repository and lists registered copies; it cannot reach copies it was never told
   about.

### Results (test run 2026-11-03, 02:52-02:55 UTC, k3)

Registration commit `8db363b6` was pushed at 02:51:12 UTC. Before the run, the code on k3 matched it by sha256
(13 files), and `params.json` and the general model matched the registered hashes. Output: `results/test_result.json`
(sha256 `00176ef5...58456f`), `results/test_knowledge.jsonl` (what PPL learned, per person), `results/test_analysis.txt`
(every PPL error); per-item records stay on k3 (`runs/test/test_records.jsonl`, sha256 `3e477405...7769baf`).

12 persons, 1,317 fresh evaluation items per condition:

| condition | all | habit | control | control over-applied | one-off probe | one-off over-applied | ordinary | routine | open items | presentation | wrong actions | turns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 0.649 | 0.302 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.314 | 0.351 | 1.82 |
| B1 | 0.698 | 0.372 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.400 | 0.302 | 1.69 |
| B2 | 0.730 | 0.571 | 0.941 | 0.039 | 0.406 | 0.552 | 0.905 | 1.000 | 0.958 | 0.995 | 0.270 | 1.32 |
| **PPL** | **0.950** | **0.885** | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.050 | 1.06 |
| PPL-surface | 0.854 | 0.667 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.146 | 1.19 |
| PPL-noval | 0.863 | 0.885 | 0.941 | 0.000 | 0.125 | 0.875 | 0.951 | 1.000 | 1.000 | 0.996 | 0.137 | 1.16 |
| PPL-noctx | 0.941 | 0.875 | 0.961 | 0.039 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.996 | 0.059 | 1.07 |

(presentation = all applicable dimensions right, where the intent was right; wrong actions = acted on a wrong
intent; turns = turns to resolution when the same item is played through with learning frozen.)

| | result | |
|---|---|---|
| L1 | habits: PPL 0.885, B0 0.302 (+58.3), B2 0.571 (+31.4); p = 0.00049 for both (PPL above B0 and above B2 for all 12 persons) | **pass** |
| L2 | all items: PPL 0.950, B0 0.649 (+30.1), B2 0.730 (+22.0) | **pass** |
| L3 | over-application: controls PPL 0.000 (B2 0.039); one-off probes PPL 0.000 (B2 0.552) | **pass** |
| L4 | ordinary requests: PPL 1.000, B0 1.000 | **pass** |
| L5 | presentation: PPL 0.996, B0 0.314 (+68.2), B1 0.400 (+59.6) | **pass** |
| L6 | routines 1.000; open-item F1 1.000 | **pass** |
| L7 | 0 differences alone vs interleaved (12 persons); general interpreter and CSI (code, frozen artefacts, probe readings) unchanged and equal to the registered values | **pass** |
| L8 | 12 of 12 persons had validated knowledge on both habits; stop: all records kept, no references, behaviour back to general 1.000; delete: 0 of 100 deleted strings left in any store repository, no references, behaviour back to general 1.000; all other items unchanged; rebuild equalled live knowledge for every person | **pass** |

All eight criteria pass.

Per person, habit accuracy: PPL 0.79-1.00, B2 0.46-0.69, B0 0.12-0.50. Sensitivity (not a criterion): conversation
memory at the threshold that is best for habits *on this test* (tau 0.2) reaches 0.705 on habits, with one-off
over-application 0.708 and overall 0.691; PPL is 18 points above even that.

Per habit (PPL / B0 / B2): clean_code 1.00 / 0.00 / 0.44; kill_email 1.00 / 0.00 / 0.31; polish_docs 1.00 / 0.00 /
0.03; ship_docs 1.00 / 0.00 / 0.12; stash (learned in files and docs, tested in code) 1.00 / 0.00 / 0.07; tidy_files
1.00 / 0.00 / 0.08; ptal 1.00 / 0.00 / 0.55; long_docs 1.00 / 0.00 / 1.00; answer_email, arrow_move, arrow_share and
fyi_docs 1.00 for PPL and already 1.00 for B0 (the general reading was right); **look_code 0.33 / 0.00 / 0.72;
red_tests 0.00 / 0.00 / 0.62**.

**Where PPL failed.** All 66 PPL errors are habit items of two habits, and both are failures of the method, not of
the run:
- look_code: every training phrasing ("look at X", "have a look at X") contained *at*, so the learned trigger kept
  it (`ctx=code, has=code, tok=look, tok=at`). "Take a look at X" was recognised; "look into X" and "X needs a look"
  were not. Generalisation only drops what varied in training, so a habit seen in one construction stays tied to it.
- red_tests: in training, "The tests on X are red." was already read as *run the tests* by the general interpreter -
  because of the word *tests* - so there was no mismatch and nothing personal was learned. At evaluation "X is red
  again" lacked that cue. Learning only from mismatches cannot learn a personal meaning that the general reading got
  right for another reason. Conversation memory got 0.62 here by word overlap ("red again").
- Also two presentation misses (one person, research searches).

**What the ablations show.** CSI lemmas: without them the personal layer reaches 0.667 on habits instead of 0.885
(inflected forms such as "needs polishing", "can be killed", "get the report shipped"; and "is" is not recognised as a
light word without its lemma *be*, which is why long_docs drops to 0.00). Validation: applying what was seen once over-applies
the one-off meanings 87.5% of the time (overall 0.863 instead of 0.950). Context scoping: without it, habits leak
into controls (3.9%) - a smaller effect than in development (19%).

**Retention in practice.** Stop learning: 7-8 experiences per person kept as records, excluded from learning; the
knowledge derived from them disappeared on recomputation. Delete: 7 experiences per person removed from the working
tree, every commit, the reflog and the object store; none of the 100 deleted strings was found anywhere in the
person's repository afterwards. All 100 were still in the backup clones made before deletion - registered in
`copies.json` and reported, as designed: copies need their own deletion.

**Decision.** By the registered rule, the question is answered yes *for this controlled setting*: with Personal
Process Learning on top of CSI and a shared general interpreter, Clover learned each simulated person's habits,
preferences, routine and open items from interaction, and answered fresh scenarios better - 0.950 first-response
accuracy against 0.649 with general interpretation only and 0.730 with conversation memory, wrong actions 5.0%
against 35.1% and 27.0% - without applying one-off meanings or habits outside their areas, without changing CSI or
the general interpreter, and with stop-learning and deletion behaving as specified. The next step under the rule is
Phase 2 (runtime integration, off by default, and a consented pilot with real use), which needs the human's approval.

What this does not show, beyond the limitations stated before the test:
1. Development to test, habit accuracy fell from 0.993 to 0.885, all of it in two habits whose phrasings vary in a
   way no development habit did. Real people's habits will vary in both ways; the two failure modes above are the
   first things to address (e.g. learning which word carries the meaning from confirmations as well as mismatches,
   and asking when a known habit word appears in a new construction).
2. The evidence is about simulated persons written by the learner's author. Nothing here is evidence about real
   people; that needs the consented pilot.

### Phase 2: runtime integration (2026-11-03)

The human said "please continue" after the hand-back that asked whether to proceed with Phase 2. Built, off by
default (ARCHITECTURE.md section 8): `clover-ai-system/mobile/personal.py`, the hook in `semantics.interpret`, the
phone runtime's wiring (on only when `CLOVER_PPL_DIR` names the person's store) with `GET /api/personal` and
`POST /api/personal/retention|forget`, and the same in the laptop Studio's staging tree (a Settings switch; not
deployed).

Checked without a model: `test_personal.py` 43 checks - off by default and every frame unchanged; a habit learned from
two corrections, applied only when validated, shown as the person's; two retries switch it off; preferred form and
stated preferences reach the model; statements are evidence; declined tools stay declined, the question before
changing files stays, the vault stays off-limits, a failing store changes nothing, a folder holding other files is
never taken over; keep, stop learning, delete (no
deleted text left in any object; the habit that depended on it gone), a deleted conversation's experiences and
preferences gone, bad ids refused; the runtime learner identical to PPL-1's on shared experiences. On k3 the
runtime store deleted through git-filter-repo and through the rebuild path, leaving nothing. Existing suites unchanged:
semantics 78, mobile 13; laptop staging 110, 129, regression all pass.

Not done: the consented pilot (it needs the human's consent, a device to run on, and their permission to read the
pilot's records); end-to-end runs with a model; routines and open items in the runtime; the two failure modes
(a new experiment, PPL-2, with its own registration).

**On by default (2026-11-03).** The human: "please keep the learning on by default as I said this is research we can
just give option to disable not the other way around". Changed: the phone runtime keeps the store in
`<base>/personal` (or `CLOVER_PPL_DIR`); the laptop Settings switch `personal` defaults to on; the switch is kept in
the store, and while it is off nothing is read or recorded (view and delete still work); `CLOVER_PPL=off` turns it
off for a process. `test_personal.py` 48 checks (on once the store is known, the switch, the process switch); the
other suites unchanged. Records stay on the person's device; reading them for research is still the consented pilot.
