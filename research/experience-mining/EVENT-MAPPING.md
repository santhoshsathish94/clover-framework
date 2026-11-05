# Event mapping

How Clover's records map onto the four levels (events, sequences, transitions, outcomes), how they correlate, what is
persisted or truncated, and what they cannot show. Sources and counts: INVENTORY.md.

## 1. Identifiers that correlate records

| Identifier | Shape | Appears in | Join quality |
|---|---|---|---|
| session | `obs-<scenario>-<hex>`, Studio `[a-z0-9]{15}`, test names | sessions file name, outcomes `session`, execution log `<session>#<n>`, growth notes, git commit subjects, drive records | exact |
| turn | `n` (1-based per session) | sessions, outcomes, execution log, growth notes, commit subjects ("Turn n of ...") | exact within a session; the drive records number turns only by order (exact in these runs) |
| time | `at` ISO (framework UTC `Z`, Studio local offset), log `YYYY-MM-DD HH:MM:SS` local, commit time | all stores | ordering only; clocks and zones differ per machine |
| commit | one git commit per write | memory repositories | total order of writes, including deletions |
| memory | `MEM-USER-<date>-<time>-<hash>`, learning `LRN-...`, seeded `MEM-CORR-...` / `MEM-HYP-...` | knowledge files, outcomes `used` (Studio), layer traces `entries` | exact; a used id stays in outcomes after the note is removed |
| recall reference | `sessions/<month>/<session>.jsonl#<n>:<role>` | layer trace `s2_turn_recall.refs` (drive records) | exact pointer from a turn to the earlier turns it was given |
| tracer | `seq`, `turn` (per HTTP request), `thread`, `layer`, `fn`, `phase` | observation harness only | exact within a run |

## 2. The four levels

**Events (what happened).** Present: message received, frame (`understood`, act, route, discernment), tools offered,
tool call with arguments, tool result (ran / worked, excerpt), answer, alignment notes, label, memory saved, feedback,
permission asked and answered, settings changed, conversation deleted, note written. Where: Studio execution log and
outcomes; framework outcomes and drive records; tracer for layer-internal calls.

**Sequences (before and after).** Reconstructable inside a session from `n` and time: message -> frame -> tools ->
call -> result -> answer -> next message (which becomes the feedback for the previous turn). Across sessions, only
through memory: a note written in one session (MEM id, commit) and read in another (`used`, layer trace `entries`,
recall refs). The framework's execution log does not record calls or results, so its sequences need the drive records.

**Transitions (how the state changed).** Recorded as state, not as transitions: memory notes appear and disappear
(commits), settings change (execution log), the retry cycle's decision (`kept` / `rewritten` / `overturned` /
`corrected` / `unconfirmed`) and a turn's `result` (`answered` / `stopped`). No record of pending -> executing ->
completed for a tool; a call and its result are two lines with no status in between. Permission requests are the one
explicit wait state (asked -> answered).

**Outcomes (what the action achieved).** Three kinds exist, none is the task's real result:
1. Clover's own completion: `result: answered` - written whenever an answer was produced.
2. Tool outcome: `ran` and `worked` per call (Studio); a tool that ran and worked says nothing about the answer.
3. Inferred user feedback: the next message labels the previous turn `continued` (worked: true), `said_not_right`
   (false), `helpful` (button). This is a guess about the user's judgement, made from the fact that they continued.
The alignment label (`from the web · not checked`, `verified · Python`, `the tool failed`, `model knowledge`) states
how an answer is known, not whether it is right. Only the benchmark records hold a scored outcome.

## 3. Persisted, truncated, lost

| Information | Studio (laptop) | Framework | Observation harness |
|---|---|---|---|
| User text, answer text | full (sessions) | full (sessions) | full (drive records) |
| Frame: understood, route, discernment | execution log | not stored | layer trace |
| Tool arguments | execution log (full JSON) | not stored | tracer, bounded |
| Tool results | excerpt in execution log | not stored | layer trace `observation_preview` / `python_output` (preview) |
| Memory given to the model | `used` ids, `context` line | not stored | layer trace `entries`, recall refs |
| Retry cycle details | `checks`, `cycle` line, `lesson` | outcome only | `clover_cycle_info` (solutions, action) |
| Model internals (C1/C2/C3) | not stored | not stored | tracer (shapes, norms, timings) |
| Live K3 server turns | - | in process memory only | - |

The tracer cuts strings over 700 characters to head and tail. The Studio keeps result excerpts, not full results.
Deleting a conversation removes its session file and notes but not its execution-log lines or outcomes, and git keeps
the file (INVENTORY.md section 5).

## 4. What the records cannot show

- **Expected outcome.** What the user wanted is recorded only as Clover's reading of it (`understood`). There is no
  record of the intended result to compare with the actual one, outside the benchmarks.
- **Actual outcome.** Inferred feedback is weak evidence: in all four synthetic runs after fix A, "That's wrong, try
  again" was recorded as `said_not_right` for a correct answer ($30 for 45 pens), and a useless photo description was
  followed by "continued" (worked: true) because the next message was a follow-up.
- **Causality.** Records give order, not cause. The only designs that support causal statements are paired runs with
  the same inputs: the CSI off/on runs and the benchmark configuration ladder. Even there, replies differed on
  messages whose route did not change, so the runtime's own variation must be measured before attributing a change.
- **Why something happened inside a turn.** Only the observation harness records layer internals; ordinary use does
  not, so a real conversation can be followed step by step but not explained inside C1-C3.
