# Decoder and Plan Interface Follow-Up

## Direction and boundaries

On 2026-11-12 the human approved the proposed next test of the plan-to-answer
interface and decoder copy constraints. Use the existing dedicated worktree
research/astra-lfm3-weightspace-20261111 and a new decoder_interface/ experiment
directory on K3. The earlier composite checkpoint and results remain immutable.
No training, weight substitution, deployment or installed-model change is
approved. Runtime policy changes apply only to new isolated test instances.

## Local hypothesis

The forced positional copy bonus, input-derived vocabulary and content-coverage
fallback can hide changes in neural logits. Removing one restriction may expose
another; a less repetitive output alone is not a correct answer. Independently,
the current exported pipeline may require answer-bearing plan content rather
than deriving an answer from question states.

## Discriminating checks

1. Preserve the stock decoder as the exact control.
2. Remove only the hard +65 copy term; retain the learned copy term and all
   vocabulary/alignment/fallback behavior.
3. Remove all copy boosting, leaving vocabulary/alignment/fallback unchanged.
4. Decode raw neural logits with the old restricted vocabulary, without copying,
   content-coverage forcing or fallback.
5. Decode raw neural logits over the full output head, still without copying or
   fallback. This does not change the model's learned weights or output-position
   layout and is not claimed to reproduce the NCM authors' generation recipe.
6. For the interface check, supply native Omni's recorded model-produced answer
   as input and compare with a deliberately swapped model-produced answer. A
   correct rendered answer supplied as input is fidelity, not acquired reasoning.

The first cheap check is an instrument test: the +65 term must dominate a toy
score, disappear only in its named policy, and leave the frozen original method
unchanged. Raw decoding must be able to emit a non-input token in a toy case.
Reuse the earlier 24 questions only for diagnosis, never as new held-out evidence.
Any improved candidate needs a separately frozen confirmation before retention.

## Frozen execution and observed outcome

Six policy/instrument tests and four candidate-gate tests passed before freezing
the follow-up. The hypothesis snapshot preserves original bytes across Windows
and Linux newline handling. Source hashes, old cases/results, policies, input
controls and the selection rule were frozen before observing new outputs.

The isolated K3 service completed 390 decoder runs and 24 new real C1 state
passes in 63.809 seconds. Original model directories were inaccessible; the
earlier composite and its results were read-only. Stock question outputs and
NCM traces matched the earlier study exactly. All decoder policies preserved
the same upstream NCM latent state for each corresponding input.

| Policy | Correct questions /24 | Question echoes /24 | Active-window fallbacks | Formatting preserved /6 |
|---|---:|---:|---:|---:|
| Stock | 0 | 24 | 0 | 6 |
| Remove hard +65 only | 0 | 24 | 20 | 6 |
| Remove all copy boosts | 0 | 24 | 29 | 6 |
| Raw, restricted vocabulary | 0 | 0 | 0 | 3 |
| Raw, full output head | 0 | 0 | 0 | 3 |

The fallback totals include the formatting controls' active windows. There are
29 active windows across the 30 question/format cases. The three unchanged raw
format controls are passthrough code, table and empty text, not decoded successes.

For each of stock, no-forced-copy and no-copy: all 24 native model-produced answer
inputs were preserved and scored correctly; all 24 deliberately swapped answer
inputs were also preserved but scored incorrectly. Both raw policies preserved
none of these supplied answers and scored 0/24 in each interface arm. This is
evidence of answer-content copying, not a new reasoning capability.

The stock decoder's forced-copy and content-coverage fallback can independently
enforce the echo. Removing them exposes different text, but the frozen raw-head
probes do not establish useful answering. They retain the existing output-position
layout and NCM level schedule; do not generalize their failure to the NCM authors'
complete generation recipe or to the separately trained full C2 backbone.

No policy meets the native-answer and formatting-retention gate. No fresh
confirmation was opened, no policy was promoted, and no weight or deployment
change was made. The final evidence audit and handoff are next.

## Independent audit complete

The four auditor corruption controls passed. The full audit verified all 390
individual records, policy and interface scores, raw sampled token IDs/text,
upstream-state invariance, checkpoint integrity and package metadata provenance.
All 197 prior source/copy files and 101 earlier result artifacts retain their
recorded hashes. No frozen source or scoring control was changed.

Representative raw outputs were read directly: full-head question-00 produced
empty text, restricted-head question-08 produced "The", and no-copy question-00
used its explicit fallback to restore the question. The zeros are not just a
formatting mismatch hidden behind aggregate scores.

The test did not establish that all NCM generation methods fail or that training
is required. It establishes the measured limitations of these five readout
policies and the current exported planning interface. A next cycle should check
the actual trained C2 planner and the NCM state/sequence contract, not repeat
this completed policy grid or treat already-supplied answers as new reasoning.

## Handoff complete

The owned K3 test service exited successfully and no inference job remains
running for this follow-up. Frozen protocol/cases, all 390 raw records, summary
and independent verification are now local as well as on K3. The evidence-only
archive hash matched before exact-member extraction:
c0a3bb628bdc017bc3447894c6961005c5bf12d09090ff109dfef2282c4e4d98.
No model weights were downloaded or changed. The existing research branch and
separate worktree were retained; no commit, push, merge or deployment ran.