# Experience reconstruction: sample histories

Execution histories rebuilt from Clover's own records, each as Context -> Direction -> Execution -> Outcome -> Growth,
with the record every step comes from. Histories 1-3 use the synthetic messages of the k3 core-observation runs
(publishable); history 4 uses the laptop's real records as identifiers, counts and times only; history 5 uses the
benchmark records. Run numbers as in INVENTORY.md section 2.

## 1. "That's wrong, try again" on a correct answer (pens; run 5, tracer from run 3)

| Step | What happened | Record |
|---|---|---|
| Context | New conversation, empty memory except seeded knowledge | memory repository at start |
| Direction 1 | "A shop sells pens at 3 for $2. How much do 12 pens cost?" | `sessions/.../obs-pens-e927c45e.jsonl` n=1 |
| Execution | no tool; C1 with C2/C3 layers | outcomes n=1 `tools` (layer names only) |
| Outcome 1 | "12 pens cost $8 ..." (right); later labelled `continued`, `worked: true` | outcomes n=1; execution log |
| Direction 2 | "And how much would 45 pens cost?" | sessions n=2 |
| Outcome 2 | "45 pens would cost $30 ... 45 pens is 15 times that amount." (right) | sessions n=2 |
| Direction 3 | "That's wrong, try again" (scripted by the drive, deliberately false) | drive record; **not** in the session file |
| Execution | retry cycle starts before semantics reads the message (tracer: `run_retry_cycle` at seq 200, `semantics.interpret` at 248, act `retry`); three C1 solves; the compiler finds no code | tracer, run 3 |
| Outcome 3 | two "independent solutions" 15 and 16; kept **15**: "one of two independent solutions gives the same result, 15" | sessions n=3; drive record `clover_cycle_info` (`action: kept`, `solutions: [15, 16]`) |
| Growth | none: no lesson, no correction | outcomes n=3; `growth/` unchanged |

What the records say about this history:
- Turn 2 is recorded as `said_not_right` / `worked: false` **twice** (one record when the retry began, one with
  `by: 3` when it ended) although the answer was right. The inferred outcome is wrong, and double-counted.
- The session file stores turn 3's user text as the repeated question, not the words "That's wrong, try again"; the
  feedback survives only as an execution-log line and outcome records.
- The cycle "checked again" and kept a wrong number: it read the multiplier in "15 times that amount" as the answer.
- Identical in runs 2, 3, 4 and 5 (same replies, same solutions [15, 16], same `kept`). Run 1 (before fix A)
  returned an empty reply to "try again".

## 2. A tool that worked and a task that failed (Python; runs 2-5, tracer from run 3)

| Step | What happened | Record |
|---|---|---|
| Direction | "Use Python to compute 2**64 + 17" | sessions n=1 |
| Execution | semantics: act `action`, caps `python`; route offers `python_interpreter`; C2 parses a tool call; permission check; Python runs and prints `18446744073709551633` (right) | tracer seq 324-361; layer trace `python_output` |
| Outcome | reply: `2**64 + 17 = \boxed{18\,446\,744\,087\,370\,5516}33` (wrong digits copied from a right result); label `model knowledge · not checked`; outcome `result: answered` | drive record; outcomes |
| Growth | none | |

The tool call succeeded and the task failed; nothing in Clover's records marks the failure as one: the label means
"no evidence matched", and `answered` is written for every answer. The mismatch is visible only by comparing the
tool output with the reply, which the drive records allow and the framework's own records do not (they keep neither).
Same wrong reply in runs 2-5; run 1 did not call Python (it answered `4,194,304,001` "from memory").

## 3. An outcome that changed a later decision (remember -> dinner; runs 2-5)

| Step | What happened | Record |
|---|---|---|
| Direction A | "Remember that I'm vegetarian." | sessions (remember conversation) |
| Execution/Growth | note saved: `notes/MEM-USER-<date>-<time>-c83843.md` and `tasks/remember-that-i-m-vegetarian.md`; reply ends "(Saved to memory as MEM-USER-...)"; label `verified · saved` | commit "Remember MEM-USER-... in notes/"; alignment note "saved what you asked to remember" |
| Direction B (new conversation) | "Suggest a quick dinner I can cook tonight." | sessions (dinner conversation) |
| Context | the note is given to the model | layer trace `entries: [MEM-USER-...]`; label `uses your memory` |
| Outcome | "How about a quick veggie stir-fry? ... any protein you have on hand, like tofu or chicken." | sessions |

The saved outcome changed the next conversation's answer (run 1, with no note saved, suggested "garlic shrimp and
spinach pasta"), but only partly: chicken is still offered. Held in all four runs after fix A. In runs 4 and 5 the
remember task note (`tasks/remember-that-i-m-vegetarian.md`) was also given to an unrelated request ("Search the web
for why the Jaipur metro is not running today", label `uses your memory`): memory reaches decisions it has nothing to
do with.

## 4. Real use on the laptop: how remembered notes travel (identifiers and times only)

From the Studio's outcomes (`used`) and git history: 8 memory identifiers were given to later turns. The most-used one
was given 22 times in 17 conversations (2026-11-01 08:59 to 12:03) until the user removed it; a note with the same
content hash under a second identifier was used 11 times in 7 conversations; a learning (`LRN-...`) was used once and
removed by the user 29 minutes later. Over the same records: 256 answered turns, 85 feedback events (continued 76, said
not right 8, helpful 1), and after "not right" the retry cycle rewrote 3 answers, kept 3 and overturned 2. Which turns
the notes improved, and whether the removals followed a bad use, cannot be told without reading the conversations (not
done).

## 5. The same tasks across benchmark configurations (laptop `verify/`)

13 LFM3-Omni-2B configuration runs over the same 65 scored tasks (step 0 to the cycle runs), each a fresh execution:

| Pattern | Tasks |
|---|---|
| fail in all 13 | 7 (math 2, knowledge 2, instruction/structured 2, coding 1); all 7 also fail with LFM3-1.7B alone |
| pass in all 13 | 23 |
| mixed | 35 (about half of them pass in 6-8 of the 13) |

In the 7 "Try again" runs (base 47 of 65 passing), the cycle fixed 3-5 failures and broke 0-1 per run; the same two
tasks were fixed in all 7 runs and two of the seven always-failing tasks were fixed in 2 runs. The fix lasts for the
turn: no run started from what an earlier run learned.
