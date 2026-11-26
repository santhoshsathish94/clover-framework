# C1/C2/C3 composite checkpoint experiment

## Direction

The human approved testing on 2026-11-11 and reiterated that it must use a
separate copy. Work remains on research/astra-lfm3-weightspace-20261111 in the
dedicated clover-astra-weightspace-wt worktree. K3 outputs belong only under
/opt/clover-k3/astra-weightspace-20261111/composite. Installed model files and
live services are read-only inputs, never execution or output locations.

This approval covers copied-artifact compatibility and composite testing, not
training, production replacement, deployment, or changing earlier frozen tests.
The no-gradient boundary and immutable original checkpoint remain in force.

## First hypothesis and check

The existing C1, C2, bridges and C3 parameter groups can be stored in one
namespaced checkpoint and executed without changing their tensor values or
stage outputs. First validate the binary export reader on known tensors and
malformed records; then copy and hash-pin actual inputs. Require exact tensor
round-trip equality before an inference comparison.

The reference uses real C1 inference states, not an embedding-table fallback.
Keep native Omni multimodal and speech modules present. Compare explicit
separate-stage execution with the combined entry point on identical inputs.
Probe the contribution of C2/C3 rather than treating unchanged echoed text as
proof of useful planning/decoding. Equality establishes faithful composition,
not improved quality or the validity of a shared-backbone substitution.

## Snapshot and tensor composition outcome

The nine preparation tests passed, including malformed indexes, independent
copy/overwrite boundaries and special-token collision detection. The real C2
export was inventoried successfully before snapshotting.

The sandboxed K3 snapshot copied 197 files totaling 5,660,259,611 bytes into
this experiment, with matching hashes and distinct inodes. Originals and input
directories are read-only in the execution sandbox; no live service changed.

The unified safetensors export contains 1,174 tensors, including original Omni,
the native speech codec, C2/bridge/C3 groups and the trained vocabulary tables.
Every tensor round-tripped byte-exactly. All 125,033 shared tokenizer IDs agree;
the trained tokenizer adds only <latent> and </latent>. Original and trained
token embeddings differ, so whole-backbone deduplication is not lossless. Both
tables are retained. This is tensor composition, not an averaging operation or
a demonstrated improvement. Inference parity and causal ablations remain next.

## Composite execution outcome

The local Python interpreter lacked PyTorch; the unchanged two loader tests
passed in the existing isolated K3 venv. No local dependency was installed.
The full loader then succeeded with the original, exported and copied input
weight directories inaccessible: original Omni, its native speech codec, both
C2 links and all 12 NCM blocks loaded solely from the unified checkpoint.
Real C1 inference returned six 2048-dimensional token states in the smoke test.

The evaluation cases and four implementation hashes were frozen before outputs.
The sandboxed comparison completed successfully in 168.788 seconds on K3 CPU:

- Native parity: 16/16 cases, 296 activation arrays exact. Modalities: four text,
	two image, two video, two ASR, two nonspeech-audio and four speech cases.
	Tokens and text match; speech codes and waveform hashes also match exactly.
- Strict pipeline parity: 30/30 cases and 177 C1/C2 state arrays exact against
	explicit separate-stage execution, including NCM state hashes and fallback
	traces. The long case exercised all 11 NCM levels.
- Diagnostic questions: native Omni 24/24 correct; both separate-stage and
	composite inference-only pipelines 0/24, echoing every question. All 30
	pipeline outputs, including six formatting controls, reproduced their inputs.
- C1-zero ablation changed NCM states on 28/30 cases, C2-identity on 23/30,
	and NCM-zero on 29/30. None changed the final text; all scored 0/24 questions.
- No explicit pointer fallback was taken in the 29 active decoder windows.
	Therefore the unchanged text cannot be attributed just to that fallback.
- Source and copy file hashes still matched after all evaluations. No original
	model, live service, previous study output or frozen test was changed.

Interpretation: a faithful one-checkpoint composite is feasible and preserves
the tested native multimodal paths. Consolidation does not repair the current
inference-only planner/decoder behavior. The output is insensitive to these
ablations on this diagnostic set despite real changes in internal NCM states.
This is not an improved model and must not be promoted as one. No backbone
substitution, gradient training, quantized-device testing or deployment ran.

Next: inspect the frozen decoder's output-vocabulary/copy constraints, verify
the saved records, and preserve the package and result evidence. Do not alter
the frozen scorer or rerun wider model edits to turn this result into a gain.

## Final audit and understanding

The independent saved-evidence verifier passed its four controls, including
deliberately corrupted scores, ablation labels and gold answers. It recomputed
the reported counts, checked saved activation structure/finite values, and
rechecked the frozen implementation, cases, unified checkpoint and all 197
source/copy file hashes. The paired activation equality was checked during the
run; only one matched copy of each array is archived, an explicit evidence limit.

The frozen source's _build_allowed_vocab_ids allows the input text, connector
tokens and word variants. _pointer_decode_from_zhat adds +65 to the copy scale
and -45 times squared position distance for ordinary non-semicolon lines.
These output constraints are consistent with the observed echoing despite
changed NCM states. No constraint was disabled and the frozen model/scorer was
not changed to rescue the result.

This composite uses original Omni and the existing exported C2/bridge/C3 modules,
not a substitution of the full separately trained C2 language transformer. The
nonidentical embedding tables remain separate. Consolidation is established;
improved answering, lossless backbone sharing, speed/memory gains and quantized
device behavior are not. Keep the research prototype, retain the original model
as the usable baseline, and make no production change.

## Handoff complete

The loader and frozen C2/C3 source were copied beside the package weights. A
second smoke from that final package location passed with the package read-only
and all original weight paths inaccessible; only the experiment cache was
writable. All owned test services exited successfully. No test remains running.

The evidence archive excludes safetensors/model copies and has SHA-256
cfe9fcb72abb7933dc1f7d7b3c63c46b42e9b17bd88e50061ed33d42edd043fa.
Its local download hash matched K3 before exact-member extraction. Frozen cases,
raw results, package metadata and VERIFICATION.json are preserved in the local
worktree. The 5,598,321,144-byte model remains only in the isolated K3 package.
No commit, push, merge, release, installed-model replacement or deployment ran.