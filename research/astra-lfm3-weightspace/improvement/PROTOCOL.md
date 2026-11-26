# Controlled initial-to-final LFM3-Omni evaluation

## Boundaries

The human approved controlled inference and reversible copied-checkpoint edits
after the weights-only examination, and requested continuous work through the
comparison. Originals immutable; no gradient computation, optimizers, training,
production changes, commits or deployment. One architectural component per trial.
Previously exposed pilot validation is not new held-out evidence.

## Data and comparisons

Discovery: the saved114-case rotated-choice/free-response diagnostic and96
matched number-binding references. Use only these for deriving candidate
directions and tensor locations. The references expose answers intentionally for
mechanistic discovery; they never substitute for test prompts.

Freeze two fresh disjoint sets before trials:
- Tuning:48 underlying problems,228 generated responses.
- Final validation:96 underlying problems,456 generated responses.
Each block of24 underlying questions contains six each of addition, subtraction,
multiplication and explicit number selection. Every question has four MCQ option
rotations; arithmetic questions additionally have a free-response version.
Numbers/options/labels are derived deterministically and checked. Underlying
question text must not duplicate discovery, tuning, or final groups.

Use the earlier frozen suite only as an additional regression gate, not as a
fresh accuracy claim. It covers unrelated text, controlled images/videos,
real-utterance ASR, non-speech controls and native speech output. Speech waveform
integrity is not independent intelligibility or naturalness evidence.

Keep the original native runtime, BF16 model/FP32 codec, CPU SDPA, deterministic
greedy decoding, identical input construction and24-slot caps for these compact
answers. Use original multimodal case limits for their regression reruns. No
prompt/scoring change between original and edited arms. Save actual token IDs,
input hashes and raw text, not just aggregate scores. Preserve protocol/source
and case hashes before execution.

## Candidate rule

First hypothesis: a localized MLP writing direction favors a default choice when
arithmetic must be bound to a label. From discovery cases with correct free
answers and correct explicit binding references, contrast failed arithmetic MCQ
states against their matched reference states. Test a sparse directional change
in ONE shared-backbone MLP output tensor; do not edit embeddings, output head or
final norm, nor copy Qwen's neuron indices or directions.

Document exact tensor, selected columns, direction, formula, strengths, expected
benefit, risks and byte-exact reversal before applying any edit. Use a fixed
small strength grid and independent controls; no iterative fitting to a loss.
If discovery disproves localization, return to context before choosing a new
hypothesis. Outlier magnitude alone cannot select a tensor edit.

Choose a candidate using tuning ONLY, requiring arithmetic MCQ improvement and
no loss in tuning free-response or direct-selection totals. Freeze its spec and
hash before executing final validation. Final comparisons report paired changes
by underlying question as well as answer count, because four rotations of the
same question are not independent samples.

## Retention gate

Require a positive final arithmetic-MCQ gain with one-sided exact paired sign
probability<=0.05 at the underlying-question level, no aggregate free-response or
selection loss, and no newly broken previously correct cases in the unrelated
multimodal regression gate. Report every changed output and remaining quality
gap. Changes in a logit or discovery score alone are not an improvement.

If no candidate passes, retain the original and report no verified improvement.
Do not keep searching the exposed final set or weaken a gate to force success.
If a candidate passes only the measured subset, call it a scoped improvement,
not a universally better model. Preserve original/final artifacts and their
precise reversible patch, plus all rejected trial records.