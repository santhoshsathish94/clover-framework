# Before-and-after results

## Recommendation: Reject

The primary intervention does not improve held-out generated-answer accuracy.
Retain the original checkpoint. No experimental checkpoint is recommended for deployment.

The predeclared primary trial is H1 at 10%; 5% and 20% are sensitivity analyses, not candidates selected afterward from validation scores.

## Held-out generated answers

| Arm | Discovery /32 | Validation /32 | Fixed / broken validation | Exact paired p |
|---|---:|---:|---:|---:|
| Original | 15 | 13 | n/a | n/a |
| H1-alpha-05 | 15 | 13 | 0 / 0 | 1 |
| H1-alpha-10 (primary) | 15 | 13 | 0 / 0 | 1 |
| H1-alpha-20 | 15 | 13 | 0 / 0 | 1 |

Exact paired probabilities condition on discordant items (two-sided binomial/McNemar). Secondary adjusted probabilities are in COMPARISON.json. This sample is not a general model-accuracy estimate or an equivalence proof.

## Unrelated and multimodal regression

| Capability | Original | 5% | 10% | 20% |
|---|---:|---:|---:|---:|
| Unrelated text /16 | 15 | 15 | 15 | 15 |
| Controlled vision /12 | 12 | 12 | 12 | 12 |
| Controlled video /8 | 8 | 8 | 8 | 8 |
| Non-speech audio /2 | 0 | 0 | 0 | 0 |
| ASR exact /6 | 1 | 1 | 1 | 1 |
| ASR corpus WER (lower better) | 7.0% | 7.0% | 7.0% | 7.0% |
| Nonempty speech waveforms /4 | 4 | 4 | 4 | 4 |

Waveform output is not a speech-quality score. Greedy audio sampling differs from the publisher's sampled default. Naturalness and intelligibility require a separate independent assessment.

## Mechanism and controls

- No-op physical copy: 112/112 cases have identical inputs, generated tokens, letter logits and speech code/waveform hashes.
- No-op activation arrays identical: 1912.
- Each nonzero trial changed exactly one stored tensor and one BF16 gain coordinate, with all other checkpoint files unchanged.
- Every required upstream activation array remained bit-identical; downstream changes were observed, so the intervention was live.
- Each experimental copy was reversed and its complete model SHA-256 matched the original; the original itself remained read-only and hash-identical.

- H1-alpha-05: 84 cases with changed letter logits, 2 with changed generated token sequences, 4 with changed speech-code streams; mean failed-discovery margin change +0.00723805.
- H1-alpha-10: 99 cases with changed letter logits, 3 with changed generated token sequences, 4 with changed speech-code streams; mean failed-discovery margin change +0.0172335.
- H1-alpha-20: 101 cases with changed letter logits, 1 with changed generated token sequences, 4 with changed speech-code streams; mean failed-discovery margin change +0.0184972.

A margin movement is a measured numerical effect, not proof that a behavioural weakness was fixed. Observed associations and untested H2/H3 remain hypotheses.

## Boundaries and next evidence

No gradient training, tensor clipping, multi-component edits, production service changes, model deployment, merge or push occurred.
The experiment tests one coordinate and three strengths, not every possible weight-space intervention. The uniformly finite weights and statistical outliers alone do not diagnose defects.
Further work should preregister a new independent question rather than reuse this validation set for selecting another coordinate. Audio-control failures and speech fidelity need separate, appropriately grounded modality evaluations.
