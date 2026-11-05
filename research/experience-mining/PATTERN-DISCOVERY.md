# Pattern discovery

Patterns seen in Clover's records, with how far each was tested across independent executions. "Run" = one framework
run of the synthetic messages from an empty memory (INVENTORY.md section 2); "configuration run" = one benchmark run of
the 65 scored tasks (laptop `verify/`). A pattern seen in one session only is listed as a candidate, not a finding.

## Patterns that held across independent executions

| # | Pattern | Evidence | Held in |
|---|---|---|---|
| P1 | The same input gets the same wrong outcome | Python result copied with wrong digits; photo "number" answered `246` (its mean colour); file outside the memory repository "not found"; "try again" keeps the multiplier `15` | 4 of 4 runs after fix A (runs 4-5 on newer code than 2-3); P1-file 3 of 3 since the read tool was routed |
| P2 | Some tasks fail whatever the configuration | 7 of 65 tasks fail in all 13 Omni configurations; all 7 also fail with LFM3-1.7B alone | 13 configuration runs + 1 other-model run |
| P3 | A tool that worked does not mean a task that worked | Python correct, reply wrong (P1); web search returned an unrelated page and the reply said it "cannot perform live web searches" (run 4), or gave an unsupported reason (run 5) | Python 4 of 4; search 2 of 2 different failures |
| P4 | Inferred feedback is not the outcome | a correct answer recorded `said_not_right` / `worked: false` (twice); a useless photo answer recorded `continued` / `worked: true` | 4 of 4 runs |
| P5 | The retry cycle's fixes are reproducible but do not last | the same 2 tasks fixed in 7 of 7 retry runs, 3-5 fixed and 0-1 broken per run; each run starts without what earlier runs fixed | 7 retry runs |
| P6 | A remembered outcome changes later decisions, partly, and also where it does not belong | dinner suggestion turns vegetarian after "Remember that I'm vegetarian" (but still offers chicken); the same note is given to an unrelated web request | 4 of 4 runs (dinner); 2 of 2 (unrelated request) |
| P7 | Recall brings unrelated earlier conversations into a turn | recall references to greeting, knowledge, pens and web conversations given to "Remember that I'm vegetarian" and to the delete request | 5 of 5 runs |

P1 and P2 partly reflect that the same model and inputs give the same reply (decoding repeats); they show systematic
failures, not chance. P5-P7 come from Clover's own mechanisms (retry cycle, memory, recall).

## Candidates (one source, or not yet testable)

- C1 A note the user keeps relying on: in real use one memory identifier was given to 22 turns in 17 conversations
  before the user removed it. Whether its uses helped is unknown without reading the conversations.
- C2 Learnings are rare and short-lived: one `LRN-` learning used once, removed 29 minutes later (real use).
- C3 Records rewrite or duplicate what happened: the session file stores the re-asked question instead of the user's
  "try again"; one feedback event becomes two outcome records (runs 2-5).
- C4 Seeded knowledge, not acquired knowledge: every new framework memory starts with the same research entries
  (`MEM-CORR-...`, `MEM-HYP-...` on caches and ontology traversal); no knowledge entry was written from an outcome in
  any run.

## What was not found

- No record of Clover changing an approach because the same failure happened before. Improvements between runs (run 1
  -> run 2: the garbled "capital of Australia" answer, the missing photo, the unsaved note) came from code changes made
  by people and agents (fix group A), not from Clover's experience.
- No transition records (pending -> executing -> completed); no expected outcome to compare with.

## How a pattern should be validated

1. State it before looking at new data (input class, expected recurrence, what would contradict it).
2. Test it on executions that share no session and no memory: fresh runs, other inputs of the same class (another
   picture with a number, another large computed result), other configurations, real-use records only with consent.
3. Count independent executions, not repetitions inside one conversation.
4. Separate determinism from recurrence: vary inputs or sampling, not only reruns.
5. Claim cause only from paired designs (same inputs, one difference), as in the CSI off/on and configuration runs.
