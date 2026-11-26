# Edited-neuron-count study

## Result

Increasing the edited-neuron count did not improve on the64-neuron control.
All values below are tuning arithmetic answers out of144, not fresh held-out
accuracy claims:

| Edited neurons | Fixed strength | Same total edit norm |
|---:|---:|---:|
| 64 | 79 | reference |
| 128 | 79 | 79 |
| 256 | 77 | 78 |
| 512 | 77 | 76 |
| 1024 | 77 | 77 |

Free-response arithmetic32/36 and number selection48/48 stayed unchanged in all
nine arms. The original model scored74/144, but the64-neuron improvement itself
was not reliable in the earlier held-out test. No wider edit qualified for
fresh confirmation. All trial copies were restored, and the original remains
the retained model. See [CONTEXT.md](CONTEXT.md) for the full evidence record.

This tests whether editing more existing neurons improves the earlier sparse
LFM3-Omni intervention. It does **not** add neurons to the architecture or train
new capacity. The original checkpoint remains immutable.

## Controlled comparison

- Fixed layer22 MLP output tensor and the same discovered direction.
- Nested64,128,256,512,1024-neuron sets from the same discovery-only ranking.
- Fixed-strength track: alpha4 at each width.
- Norm-matched track: wider edits scaled to the64-neuron perturbation norm,
  with actual BF16-rounded norms recorded.
- Exact64-neuron checkpoint and output reproduction before wider trials.
- Same tuning prompts/scorer/decoding; no modification of earlier results.
- A fresh96-question/456-response confirmation set is frozen before trials,
  disjoint by question text from earlier discovery/tuning/final data.

Each trial changes one tensor in a physical copy and verifies the exact changed
columns, all untouched tensors/files, and byte-exact reversal. A candidate must
beat both original and64-neuron tuning results without protected-score losses
before confirmation is opened. Final retention additionally requires grouped
paired confirmation versus both baselines and unrelated/multimodal regression
checks. Changed speech requires independent content assessment.

## Files

- [CONTEXT.md](CONTEXT.md): direction, preparation correction, observed results and limits.
- [study.py](study.py): frozen ranking, norm controls, guarded copies and confirmation.
- [run.py](run.py): serial trial, candidate selection and conditional final checks.
- [verify.py](verify.py): result hashes, actual neuron counts, control identity and reversals.

The K3 result files are written under
`/opt/clover-k3/astra-weightspace-20261111/improvement/neuron_count`:
`frozen/`, `TRIALS.json`, `OUTCOME.json`, `RESULTS.md`, and, only when eligible,
`confirmation/` and `CONFIRMATION.json`. Trial outputs are under the parent
`improvement/runs/width-*`; copy audits/reversals are under `improvement/copies/`
or `neuron_count/restored_trials/` after confirmation copies are made.

Run the saved-results verifier using the existing isolated K3 environment:

```bash
/opt/clover-k3/astra-weightspace-20261111/venv/bin/python -B \
  /opt/clover-k3/astra-weightspace-20261111/improvement/neuron_count/verify.py \
  --root /opt/clover-k3/astra-weightspace-20261111
```

Do not rerun preparation or reuse existing copy/output names. Source, hypotheses,
neuron sets and confirmation cases are frozen; preserve a failed attempt before
repairing an instrument. Do not use an exposed confirmation result to choose
another width, strength or direction.

## Limits

This isolates the size of the edited subset and the size of the perturbation.
It cannot establish that adding architectural capacity or gradient training
would help. Tuning gains alone are not a model improvement, and a constructed
arithmetic test is not universal model quality. No production deployment,
commits or pushes are performed by this study.