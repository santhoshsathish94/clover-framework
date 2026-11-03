# Personal Process Learning: working context

## Direction (human, 2026-11-03, key lines in their words)
- "Clover is a personal AI system. The objective is to improve the AI's ability to understand and work with each person
  over time: how they communicate, what they mean, how they approach problems, what they prefer, what they have
  learned, and how they want the AI to help."
- Three layers, kept distinct: CSI (general meaning), personal understanding (the individual), personal process
  learning (the relationship over time: "Interactions -> observations -> hypotheses -> validation -> adaptation").
  "One person's communication habits should not silently become universal language knowledge in CSI."
- "Clover must learn from the interaction, not just save the user's statements." The loop: reconstruct what was
  understood and why, find the mismatch, look for contextual patterns, form a provisional hypothesis, test it in later
  interactions, retain it only with sufficient evidence.
- Deletion: three user choices - Keep; Delete (remove from active stores and apply history/backup deletion); Keep
  history, stop learning. "Deletion should also invalidate derived personal knowledge when that knowledge depends on
  the deleted experience."
- Question: "Can Clover learn how one individual communicates and works, then demonstrably improve its assistance for
  that individual in future, previously unseen interactions?" Controlled interactions first, separate training
  experiences and fresh evaluation scenarios; "without simply memorizing previous conversations". CSI is the semantic
  foundation; its limits are measured, not extended here.
- "I want the full system this is research experiment so let's do it fully" (2026-11-03).
- Earlier (2026-11-01, core observation): "the user semantics should not corrupt the core semantics".

## Boundaries
- CSI stays frozen (v0.1r3, commit a00322e); its outputs are checked unchanged after personal learning.
- Controlled, simulated persons only; no real conversations are read.
- Research area only until the experiment decides; runtime integration (off by default) only after it passes.
- k3: writes only under /opt/clover-k3/meaning-core/ppl; the live K3 server (8765) is not touched.
- No Qwen-based components.

## Status
- 2026-11-03: system design (ARCHITECTURE.md); implementation in `ppl/`; experiment plan and results in EXPERIMENTS.md.
- 2026-11-03: general interpreter trained (held-out 1.000, in-distribution); development runs 1 (no noise) and 2
  (inattention + one-offs); dry run of the full test procedure on development persons; PPL-1 pre-registered
  (EXPERIMENTS.md) before any test-pool person was run.
- 2026-11-03: PPL-1 test (12 fresh persons): all eight criteria pass - first-response accuracy PPL 0.950, general only
  0.649, conversation memory 0.730; habits 0.885 / 0.302 / 0.571; no over-application of one-offs or habits outside
  their areas; retention behaved as specified. Two failure modes found (a habit tied to its training construction; a
  personal meaning masked by a general cue during training). Phase 2 awaits the human's decision.
- 2026-11-03: after the hand-back the human said "please continue"; read as going ahead with Phase 2. Runtime
  integration built, off by default (ARCHITECTURE.md section 8): `clover-ai-system/mobile/personal.py`, the semantics
  hook, the phone runtime's wiring (`CLOVER_PPL_DIR`), the laptop staging tree's wiring (Settings `personal`, not
  deployed). Not enabled anywhere, no real conversation read. The pilot needs the human's explicit consent.

## Where things are
- k3 `/opt/clover-k3/meaning-core/ppl/`: `code/` (uploaded LF copies, checked by `manifest.sha256`), `ppl-env.sh`,
  `ppl-stage.sh general|dev|dryrun|test`, `runs/` (general, dev-run1, dev, dryrun-1, dryrun, test; per-record JSONL
  stays there), `tools/fr/` (git-filter-repo 2.47.0, MIT, used as a tool, not vendored).
- Tests: `python tests/test_ppl.py` (no CSI or sklearn needed); store checks run when `PPL_FILTER_REPO` is set.
