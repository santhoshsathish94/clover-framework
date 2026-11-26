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

## Successful preparation

The corrected preparation succeeded and is frozen. Original discovery ranking
and direction reconstructed exactly;2549 neurons meet the same positive-score
eligibility rule. All requested widths are available, yielding nine specs.
Fresh confirmation has96 balanced underlying questions/456 responses disjoint
from earlier discovery/tuning/final questions; its outputs remain unopened.

Measured target-tensor perturbation norms after BF16 rounding, before trials:
64fixed1.996355;128fixed2.283481;256fixed2.639010;512fixed3.096140;
1024fixed3.623604. Norm-matched widths are1.996200,1.996147,1.996129,1.996099,
within0.013% of the64 control. Matched strengths are3.496989,3.025846,2.579033,
2.203484, respectively. Record these actual norms, not assumed exact equality.

The serial run first requires the64-neuron checkpoint to match the old candidate
SHA0e428fe5808efa47e7f588078b8d348b8363001354a500e97df5a61110fa6078
and reproduce its tuning outputs exactly. All wider copies require exact
requested changed-column counts, one changed tensor and byte-exact reversal.
No trial outcome is established at this context update.

## First completed trials

The64-neuron control passed exact checkpoint-hash and tuning-output reproduction:
79/144 arithmetic MCQ,32/36 free response,48/48 direct selection. The128-neuron
fixed-strength trial also scored79/144,32/36,48/48. Both changed exactly the
requested columns of one tensor and were reversed after their runs. Doubling
the edited count has not yet improved the score; the remaining fixed-width and
norm-matched comparisons are still pending. Do not infer their outcomes.

## Completed count experiment

All nine trials finished successfully. The evidence verifier checked2052 tuning
responses, all frozen specifications/source hashes, exact64-control identity,
actual changed-column counts, norm controls and all trial reversal records.
The completion subscriber exited and its terminal was cleaned up. No inference
job remains running for this experiment.

| Edited neurons | Fixed alpha4 arithmetic /144 | Norm-matched arithmetic /144 |
|---:|---:|---:|
| 64 | 79 | same reference |
| 128 | 79 | 79 |
| 256 | 77 | 78 |
| 512 | 77 | 76 |
| 1024 | 77 | 77 |

Every arm retained free response32/36 and direct selection48/48. The unedited
original tuning baseline was74/144,32/36,48/48. The earlier64-neuron edit did not
establish reliable held-out improvement; matching its tuning score does not
turn either edit into a validated upgrade.

No wider candidate exceeded the64-neuron control while preserving the other
criteria. Therefore no fresh confirmation was opened, no new multimodal or
speech-retention test was required, and no wider model was retained. All nine
physical trial copies were restored byte-exactly; original weights untouched.

Understanding: increasing the edited subset up to1024 did not fix the limiting
factor for this direction/layer. Controlling total perturbation size did not
reveal a hidden benefit from broader coverage. This is a bounded negative
result, not proof that every larger subset or a physically wider/trained model
would fail. Do not escalate widths or retune the old exposed final set to force
a positive result. Any different mechanism needs a separately documented test.

The first attempted frozen-artifact download returned no recorded outcome;
checking the local directory confirmed it was absent before retrying transfer.
Completed evidence is under frozen/,TRIALS.json,OUTCOME.json,RESULTS.md and the
parent improvement/runs/width-* and improvement/copies/width-* audit records.

The evidence archive (no model tensor files) was hash-verified before extraction:
2fc4ca4d01388b13fbc46980317bdf956fcbc2f606ca37692694f8ccc538e231.
Windows tar rejected GNU --wildcards before extraction; using exact member paths
succeeded without replacing existing source or old study records. The final
service is inactive with exit0. The fresh confirmation set remains unused.