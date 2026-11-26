# Initial-to-final LFM3-Omni outcome

## Decision: reject

The frozen final or regression gate did not pass.

The candidate was selected on tuning only. Its final set was not used to choose its layer, direction, columns or strength.

## Initial versus tested candidate

| Measure | Original | Candidate |
|---|---:|---:|
| Arithmetic choices | 134/288 | 137/288 |
| Arithmetic free response | 61/72 | 61/72 |
| Direct number selection | 96/96 | 96/96 |

Arithmetic-choice accuracy: 46.53% -> 47.57% (+1.04 percentage points).
Underlying questions improved/worsened: 12/13; one-sided exact sign p=0.654981.
Fixed/broken individual arithmetic-choice responses: 18/15. Four rotations of one question were treated as dependent, not four independent successes.

## Regression evidence

Newly broken previously correct unrelated/multimodal cases: [].
ASR word error worse: False. Changed speech-code streams: ['tts-000', 'tts-001', 'tts-002', 'tts-003'].
This is not a human naturalness assessment, a large natural-media benchmark, or proof of universal improvement. The reused regression set is a gate, not a fresh generalization claim.

## What changed

One tensor: `model.language_model.layers.22.feed_forward.w2.weight`, selected64 output columns, sparse directional subtraction at strength4. No gradient training; embeddings, final norm, output head and all other tensors unchanged.
Changed BF16 entries: 129166. The full tensor audit is in the candidate's PATCH_AUDIT.json.
The direction came from matched discovery arithmetic/reference states, restricted to problems with correct free answers and explicit number binding. It is not a Qwen direction or an edit justified by a weight outlier.

## Initial and final artifacts

Original checkpoint SHA-256: `76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`.
Tested candidate SHA-256: `0e428fe5808efa47e7f588078b8d348b8363001354a500e97df5a61110fa6078`.
Retained final SHA-256: `76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`.
Final directory on K3: `/opt/clover-k3/astra-weightspace-20261111/original`.
All exploratory trial copies were reversed exactly. A rejected final candidate is also restored; a retained candidate remains a separate physical copy. No original was overwritten and no model was deployed.

## Study accounting

Discovery114 responses plus96 matched references; original tuning228 plus six initial and two adaptive follow-up trials; final original/candidate456 each;112 regression responses. The stronger-dose follow-up was documented before it ran and used tuning only.
The primary final acceptance criteria were not weakened after observing results. All raw outputs, input hashes, per-question comparisons, frozen source/case hashes, hypothesis files and reversal records are retained.

## Remaining limits

Constructed arithmetic-choice validation is a limited domain. Even a retained candidate is not established to improve free-form reasoning, all factual knowledge, all languages, general vision/audio/video, or natural speech quality. Quantized exports and deployed Clover runtimes were not evaluated.
The result establishes only the measured scope. A failed gate means the original remains the final version; activity or tuning gains are not substituted for a verified improvement.
