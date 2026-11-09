# Pre-registration: held-out re-test of the OpenJev routing claim

Written before any held-out run. Review of `research/superagent-jev-dory` at `9b12f274`.

## Claim under test

From `docs/runtime-context/superagent-jev-dory/02-openjev-decision-engine-context.md`, lines 79-80:
`LFM3-1.7B` with `CloverUpgraded` ChatML and prior_alpha 0.55 (temperature 0.7) routes the
12-task benchmark at 100% (12/12) with ECE 0.0707.

## Why a held-out test is needed

- prior_alpha 0.55 appears only in the reported arm (the other arm uses 0.8, the class default is 0.7).
  No sweep and no train/test split is recorded. The 12 tasks that set the configuration also scored it.
- The same configuration scored 4/12 (ECE 0.4637) on `Qwen2.5-0.5B-Instruct` in the branch's own
  results file. The ledger table does not show that cell.

## Frozen before the run

- Code: the study script `run_openjev_live_experiments.py`, imported unchanged
  (sha256 prefix `0f703d99ca002716`, identical on k3 and in the branch). Its own
  `CloverUpgradedOpenJevBackend`, `compute_ece_and_brier`, and the unmodified OpenJev `HFBackend`.
- Request: the same `Choice` question, instructions and four criteria as the study (asserted
  verbatim against the script source at run time).
- Arms (pre-registered): A0 raw `HFBackend`; A1 Markdown, alpha 0.8; A2 ChatML, alpha 0.55
  (the reported configuration). Temperature 0.7 for A1 and A2, as in the study.
- Model: `/opt/clover-k3/models/LFM3-1.7B`, bf16, CPU, as in the study.
- Environment: `/root/hybridagent-venv` on k3 (Python 3.12.3, torch 2.11.0+cpu,
  transformers 5.13.0, pydantic 2.14.3); the only environment on k3 with all of them, and the
  study's bytecode is cpython-312.
- Tasks: `heldout_tasks.json`, 24 tasks written for this test, 6 per label. H1 (12) keeps
  the original style; H2 (12) is plain style without the status sentence. No label word or
  stem (act, gather*, clarif*, verif*) appears in any state (asserted).

## Gate G0 (liveness and reproduction)

Run the original 12 tasks first. The A0/A1/A2 per-task predictions on LFM3-1.7B must match the
recorded `per_task_comparison` (36/36). If fewer match, the environment differs from the study's;
held-out numbers then describe this environment and are reported with that caveat.

## Primary measure and decision rule (A2, LFM3-1.7B, 24 held-out tasks)

- Replicated: 21 or more of 24 correct.
- Partially replicated: 15 to 20.
- Not replicated: 14 or fewer.

## My predictions (point estimates, stated now)

- A2 about 17/24 (H1 about 10/12, H2 about 7/12): partially replicated.
- On held-out tasks A2 >= A1 >= A0; A0 predicts mostly `act`.

## Exploratory, no decision attached

- ChatML with alpha 0.0, 0.3, 0.8, 1.0 and Markdown with alpha 0.55 (separates the template
  effect from the alpha effect; shows whether 0.55 is special).
- The same arms on `Qwen2.5-0.5B-Instruct` (local HF cache, offline).
- ECE is reported but no ECE claim is made: 24 tasks over 5 bins is too few.

## Rules after the run

Gold labels are not edited. If a gold label looks arguable once the results are in, that is
reported as a separate note and the pre-registered score stands.
