# Results: Clover Experience Mining 0.1

Read-only investigation of what Clover already records (2026-11-03). No runtime change, no learning algorithm, no
memory update. Evidence states as in AGENTS.md section 11: known, inferred, unknown.

## What the records can carry

- **Persisted:** every turn's text (sessions), Clover's completion and tool outcomes (outcomes), the user's next move
  as inferred feedback, memory written and (in the Studio) memory used, the retry cycle's decision, and in the Studio
  the frame, tool arguments and result excerpts (execution log). Git keeps every write in order.
- **Truncated or missing:** tool results (excerpts in the Studio, absent in the framework's own records), layer
  internals (only in the observation harness), the live K3 server's conversations (process memory only), the expected
  outcome (nowhere), transitions (only end states).
- **Correlation:** session + turn number join all stores exactly; memory identifiers and recall references link a turn
  to the earlier records it was given. Clocks and time zones differ between machines.
- **Sensitive:** real conversations, tool arguments, personal notes; deleted conversations remain in execution logs,
  outcomes and git history (INVENTORY.md section 5).
- **Causality:** records give order; only the paired runs (CSI off/on, benchmark configurations) support cause.

## The five questions

1. **Does Clover repeat the same mistakes across executions?** Known: yes, for the cases observed. Identical inputs
   produced the same wrong outcomes in 4 of 4 independent runs (copied digits, the photo "number", the retry that keeps
   a multiplier, the unreadable file), and 7 benchmark tasks fail in all 13 configurations. Unknown: how often this
   happens in real use, which would need outcome labels for real conversations.
2. **Does it discover better approaches through repeated experience?** Known: not in these records. Better outcomes
   between runs came from code changes by people and agents. Within a turn the retry cycle finds a better answer for
   3-5 of 18 failures, reproducibly, but no execution starts from what an earlier one found. One learning was used once
   in real use: too little to tell.
3. **Can we reconstruct why an execution succeeded or failed?** Known: partly. With the observation harness, down to
   the layer (Python printed the right number; C1 copied it wrong; the retry's answer reader took "15 times" as the
   answer). With the Studio's records, what happened (calls, result excerpts, answer) but not why inside the model.
   With the framework's own records alone, not enough (no calls, no results).
4. **Do outcomes change later decisions?** Known: through memory, yes - a remembered fact changed the next
   conversation's answer in 4 of 4 runs (partly), and in real use notes were given to later turns up to 22 times.
   Through the retry cycle, within a conversation. Not otherwise: no record shows a past failure or feedback changing a
   later route or approach. Memory also reaches unrelated decisions (5 of 5 runs).
5. **Has Clover acquired knowledge it never consolidated?** Inferred: the records hold stable experience - failure
   signatures that recur across independent runs, the conditions under which a tool or the retry cycle helps - that no
   Clover knowledge structure expresses (its corrections and observations are empty or seeded). That experience is
   consolidated only outside Clover, in people's and agents' notes. Clover does not use it: it is accumulated in the
   records, not acquired in behaviour.

## The central hypothesis

"Clover may already have accumulated more experience than its current knowledge structures can express." Supported in
a narrow sense (inferred): repeatable experience exists in the records and no knowledge structure holds it. Not
supported in the stronger sense that Clover has learned it: nothing in the records shows its behaviour changing from
that experience, except remembered notes the user asked it to keep.

## Limits

- The cross-run evidence comes from 15 synthetic messages (plus 7) and 65 benchmark tasks, with one model family.
- Real conversations were not read; real-use results are identifiers, counts and times.
- Feedback labels are not outcomes, so real-use success rates are unknown.
- Records of the 169 deleted conversations were counted, not read; using them would need the human's decision.

## What would make experience learnable (for the human's decision; not done)

- Record what the turn was for (the frame's reading at least) next to the outcome, and the user's real words on a retry.
- Record one feedback event once, and mark it as inferred.
- Keep tool results (or a reference) beside the reply, so "tool worked, task failed" is detectable.
- Give the framework's execution log the calls and results the Studio's already has.
- Decide what deletion should remove (derived records and history).

## Possible next steps (the human's choice)

1. Label outcomes for a sample of real conversations, with the human (or with their consent), to measure how often
   inferred feedback is wrong and how often real failures repeat.
2. Validate P1-P7 on new held-out executions in the k3 sandbox (other inputs of the same classes), as a pre-registered
   check.
3. Mine the development records (agent sessions and notes) for question 5 at the level of the people and agents
   building Clover.
