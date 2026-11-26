# Frozen pilot protocol

## Scope and evidence levels

This is a controlled exploratory study of one pinned LFM3-Omni-2B checkpoint on
K3, not an estimate of general model quality or a conventional fine-tuning run.
Weight observations, activation associations, mathematical predictions, and
experimentally verified changes must be reported separately.

The original snapshot is immutable. Analysis uses a network-disabled systemd
sandbox with its original directory mounted read-only, 8 CPU quota, nice 15,
24 GiB RAM ceiling, and no swap. Every complete phase verifies source hashes.
Model code and processor come from the pinned snapshot, not the deployed app.

## Frozen Cases

- 32 discovery text questions: addition, subtraction, multiplication and exact
  integer division, balanced by operation and A/B/C/D answer position.
- 32 separately generated validation questions, also balanced. No validation
  activations, answers, margins or scores enter candidate selection.
- 16 unrelated factual, language, logical and elementary programming questions.
- 12 generated, exactly labeled vision cases covering colors, spatial relations,
  counting and text recognition. These are controlled diagrams, not natural-image
  benchmark claims.
- 8 generated four-second, eight-frame videos, including order-reversed pairs.
  The actual frame sequences go through the native video processor/vision tower.
- 6 public LibriSpeech-dummy utterances selected by row order and 2-10 second
  duration, not model performance; 2 generated non-speech audio controls.
- 4 speech synthesis prompts through the native depth decoder and waveform codec.

Total: 112 cases. Separate pilot examples test APIs and observation hooks before
freezing. The case JSON, fixture bytes, source metadata, runtime, scorer, and this
protocol are hashed before the first baseline. No case or scoring changes are
allowed during a comparison; an instrument defect invalidates that comparison.
No expected answer or reference transcript is passed to the model except as one
of the presented multiple-choice options.

## Fixed Inference and Scoring

BF16 model, FP32 codec, CPU SDPA, eval mode, no gradients. Batch size one, new
request/cache per case, seed 20261111. Greedy text and speech; text temperature 1,
top-p 1, top-k 0, repetition penalty 1; audio temperature 0 and repetition penalty
1.2. This deterministic speech setting is not the publisher's sampled default.
Text limit 24 generated slots (96 for ASR and speech). Record limit hits, actual
input tensor hashes, generated IDs, first-token A/B/C/D logits, and wall times.
Timings are descriptive under resource caps, not performance improvements.

Primary accuracy: the generated answer's leading A/B/C/D label. A restricted
four-letter argmax is an auxiliary diagnostic, never substituted for generation.
ASR: normalized exact match and standard jiwer word error rate; preserve raw
transcripts. Speech: successful decoding, finite/nonempty waveform, duration,
clipping and exact code/waveform hashes. Waveform validity does not establish
spoken intelligibility or naturalness; those remain unverified without independent
listening/transcription. Explicitly retain that limitation in recommendations.

## Candidate Rule Before Looking at Behaviour

Observe layers 0, 2, 4, 11, 17 and 23, the final shared RMS norm, modality encoders
and projectors, and the speech-depth trunk. Hooks return no replacement values.
Check generated IDs with and without hooks on a separate pilot example.

The first eligible intervention targets exactly one coordinate of
`model.language_model.embedding_norm.weight`. Its source-defined location is
after the last shared backbone block and before the tied text head. For an
observed final hidden vector `h` and tied embedding rows `E`, the contribution
of coordinate `j` to the correct-versus-best-incorrect letter margin is
`c_j = (E_gold,j - E_competitor,j) * h_j`.

For failed discovery examples, reducing a coordinate with negative mean `c_j`
predicts a better first-token margin. Rank coordinates by
`-mean(c_j | failure) - 2 * max(mean(c_j | success), 0)`, eligible only when the
first term is positive. Choose the largest score deterministically, lowest index
on ties. Do not choose from statistical weight outliers alone. Preserve group
sizes, unstandardized and standardized activation contrasts, and label/operation
confounding checks. This supervised association is exploratory and subject to
multiple comparisons, not a causal explanation.

Proposed strengths are 5%, 10%, and 20% attenuation of that single original gain.
The predeclared primary strength is 10%; 5% and 20% are sensitivity analyses and
cannot be selected afterward because their validation score looks better.
No optimizer, gradients, learned adapter, alternate model, or other tensor edit.
If discovery has no successes/failures or no eligible coordinate, propose no
edit and report the eligibility failure rather than inventing a target.

Document the observed evidence, derivation, exact coordinate, original/new BF16
bytes, expected changes, regressions, and reversal before materializing a patch.
The first-order margin expression is exact in real arithmetic for a fixed
competitor, but actual BF16 norm/head rounding and future generation can differ.
Changing this norm can affect every modality, including speech-depth inputs.

## Validation and Decision

Apply one strength at a time to a fresh physical copy, never a hard link. Require
the original checkpoint digest, the hypothesis-document digest, exact original
bytes, and a one-tensor allowlist. Hash all untouched tensors. A no-op copy and
byte-exact reversal are controls. Rerun the complete frozen suite for each tested
strength under the same settings and compare paired outcomes and input hashes.

Report discovery, validation, and unrelated regression outcomes separately.
Compute exact paired sign/McNemar probabilities for discordant validation cases;
secondary strength results are exploratory and multiplicity-adjusted, not a way
to rescue a failed primary result. Report confidence limitations for this small
constructed sample and any missing independent modality-quality assessment.

Reject harmful or unsupported edits. A small score increase alone warrants
investigation, not adoption. Retain only with convincing held-out improvement and
adequate unrelated/multimodal regression evidence; otherwise recommend retaining
the original and rejecting the intervention or investigating further. Do not
publish/deploy any experimental weights.