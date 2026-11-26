# Edited-neuron-count experiment

## Direction and fixed boundaries

2026-11-11: the human asked whether increasing the number of neurons improves
the result. This experiment increases the number of EXISTING MLP neurons whose
output columns are edited; it does not enlarge the model architecture. The
interpretation was stated before execution. Keep layer22, its existing discovered
direction, original checkpoint, decoding, scoring and single-component boundary.
No gradients, original writes, deployment, commits or changes to old evidence.

## Hypothesis and discriminating controls

The64-column patch may cover too few of the distributed writers associated with
the discovered binding failure. Use the SAME discovery-only ranking to choose
nested64,128,256,512,1024-column sets, subject to the same positive-score eligibility.
Reconstruct the previous64-column set exactly and require its strength4 tensor
to match the old audited candidate before comparing anything wider.

Two fixed tracks distinguish coverage from total perturbation magnitude:
- Fixed strength: alpha4 for every width.
- Norm matched: scale alpha so the pre-BF16 Frobenius delta equals the64-column
  alpha4 delta; record actual post-round norms instead of assuming exact equality.

No layer/direction/strength optimization beyond this specified normalization.
Do not silently include negatively ranked columns to reach a requested width;
report if a width exceeds eligible candidates. No additional widths selected
after observing results.

## Evaluation

Use the prior tuning cases to compare width against the original and64-neuron
control. Re-evaluate64 and require identical tokens to its earlier run. A wider
candidate qualifies only if its tuning arithmetic count exceeds BOTH baselines,
with no free-response or direct-selection loss. Break ties by smaller width,
then smaller actual perturbation norm.

Before width trials, freeze a fresh96-question/456-response confirmation set,
disjoint from prior discovery, tuning and exposed final questions. Keep it out of
selection. If a wider candidate qualifies, evaluate original,64 control and the
selected width on this set under identical settings. Require grouped paired
improvement versus BOTH original and64 (one-sided exact sign p<=.05), protected
free/selection totals, and unrelated/multimodal regression checks. A change in
speech requires an independent content check; no naturalness claim.

All copies are physical, one tensor changed, original hashes and unchanged
columns/tensors audited, rejected/trial copies reversed byte-exactly. If no width
qualifies on tuning, do not expose confirmation just to hunt for a result.

## Current evidence

The preceding64-neuron study did not establish a reliable held-out improvement.
Its exposed final set is not new validation. This experiment has not yet run;
record the actual width/quality/norm tradeoff and outcome without assuming that
more edited neurons must help.

## Preparation correction

The first preparation reconstructed the64-column ranking/direction but stopped
before its manifest/trials: the prior batch-level question builder exhausted
its retry limit while excluding all earlier questions. Direct-selection questions
draw from a small77-number space, so rejecting a whole24-question batch for one
collision is inefficient. Preserve the partial output as preparation-v1-failed.

Select unused questions individually from the same seeded generator, retaining
24 underlying questions per family, four balanced rotations, and the same free
versions:96 groups/456 responses. No model outputs were viewed and no scoring
or acceptance criterion changed. A test explicitly checks disjointness from all
previous splits. Freeze the hypothesis snapshot and experiment source with the
successful preparation so later context updates cannot redefine the experiment.