# Clover Experience Mining: working context

## Direction (human, 2026-11-03, key lines in their words)
- "I would not start by building another Experience Recorder. First, let's find out what Clover already records and
  what can be reconstructed from it." Four levels: events, sequences, transitions, outcomes; "A successful tool call
  does not necessarily mean a successful task."
- "Use existing, historical Clover records. Do not change the live runtime." "No new learning algorithm. No runtime
  changes. No automatic memory updates." "we must not treat a correlation in logs as proof of causation."
- Five questions (repeat mistakes; discover better approaches; reconstruct why; outcomes change later decisions;
  knowledge acquired but never consolidated). Hypothesis: "Clover may already have accumulated more experience than its
  current knowledge structures can express."
- First deliverable: the inventory and a reconstructed sample of real execution histories.

## Boundaries kept
- Read only on every machine. Laptop: scripts copied to its temp folder, run with the Studio's Python, then deleted;
  no `git status` (it can rewrite the index); the Studio was not started or changed. k3: the live K3 server was seen
  only through /proc and file metadata; another session's staging (`s23-omni2b`) and all vaults were not read.
- Real conversations were not read: real-use findings are identifiers, counts, categories and times. Detailed
  histories use the synthetic core-observation messages and public benchmark tasks.
- Records of conversations the user deleted were counted, not read.

## Status
- 2026-11-03: inventory (INVENTORY.md), event mapping (EVENT-MAPPING.md), five reconstructed histories
  (EXPERIENCE-RECONSTRUCTION.md), patterns with their cross-run status (PATTERN-DISCOVERY.md), answers to the five
  questions (RESULTS.md). Scripts that produced the counts: `scripts/`.
- Waiting on the human: whether to push this folder (public repository); whether real conversations may be read or
  labelled with them; whether development records (agent sessions) are in scope; which next step, if any.
