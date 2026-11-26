# Decoder and Plan Interface Results

## Outcome

The follow-up completed on isolated K3 research copies. **No decoder policy
qualified as an improved model.** All weights, the completed composite experiment,
installed models and live services remain unchanged.

The five policies ran on the same 24 previously exposed diagnostic questions
and six formatting controls. None answered a question correctly. Native Omni's
recorded answer baseline is 24/24; these are diagnostic comparisons, not new
held-out accuracy claims.

| Decoder policy | Correct /24 | Question echoes /24 | Fallback windows | Format preserved /6 |
|---|---:|---:|---:|---:|
| Stock | 0 | 24 | 0 | 6 |
| Remove hard copy bonus | 0 | 24 | 20 | 6 |
| Remove all copy boosts | 0 | 24 | 29 | 6 |
| Raw logits, restricted vocabulary | 0 | 0 | 0 | 3 |
| Raw logits, full output head | 0 | 0 | 0 | 3 |

Fallback counts cover 29 active windows across the question/format cases. The
three preserved raw-policy controls were passthrough code, table and empty text.

## What Changed Our Understanding

Removing copy bonuses alone does not remove the echo: the content-coverage
fallback restores the input. Removing that fallback and decoding raw logits
does change the output, but does not recover correct answers. For example,
the full head returned empty text for `What is 14 plus 27?`; restricted raw
decoding returned `The` for the France-capital question.

All policies produced exactly the same upstream NCM latent state for each
corresponding input. The differences are therefore downstream of those states.

The answer-input controls distinguish copying from reasoning:

| Supplied input | Stock and both copy-reduction policies | Both raw policies |
|---|---|---|
| Native Omni's recorded answer | Preserve 24/24; score 24/24 | Preserve 0/24; score 0/24 |
| Another question's recorded answer | Preserve 24/24; score 0/24 | Preserve 0/24; score 0/24 |

The successful answer rendering follows supplied answer content. It is not
evidence that C2 or C3 solved the question. Neither a gold answer nor an
answer-aware vocabulary was injected into an answering policy.

## Verification and Limits

- 390 decoder runs and 24 new real C1 state passes completed on K3 CPU.
- Six policy controls, four selection-gate tests and four auditor controls passed.
- Stock outputs and traces reproduce the earlier frozen study exactly.
- The independent auditor recomputed all policy/interface scores and verified
  sampled raw IDs against text, frozen inputs, the checkpoint and metadata.
- All 197 prior source/copy files and 101 prior result artifacts retain their
  hashes. No training, weight substitution or deployment ran.
- No policy met the native-answer and formatting-retention gate; fresh
  confirmation was not opened and no policy was promoted.

The raw probes retain the existing output-position layout and NCM level schedule.
They are controlled readout experiments, not a reproduction of the NCM authors'
full generation recipe. The separately trained full C2 transformer was not
substituted. Canonical-token coverage fields do not prove that every alternative
tokenization of an answer is impossible. No speed, memory or device-quality
improvement is claimed.

The next useful investigation is the actual trained C2 planning path and its
NCM state/sequence contract, before further decoder changes or training. Do not
equate rendering an already-correct answer with improved reasoning.

## Reproducibility

Code: [policy.py](policy.py), [study.py](study.py), [verify.py](verify.py).
Cycle record: [CONTEXT.md](CONTEXT.md).
K3 evidence: `/opt/clover-k3/astra-weightspace-20261111/decoder_interface`.
Frozen protocol/cases/hashes, all individual JSON outputs and the independent
verification report are preserved there and in the dedicated local worktree.
The original composite checkpoint remains read-only in its earlier directory.

The evidence-only archive was SHA-256 verified before local extraction:
`c0a3bb628bdc017bc3447894c6961005c5bf12d09090ff109dfef2282c4e4d98`.
No model weights are included in that archive.