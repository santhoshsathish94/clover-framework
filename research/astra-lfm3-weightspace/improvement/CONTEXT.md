# LFM3-Omni controlled improvement study

## Current human direction

2026-11-11: continue without returning after every intermediate step; seek an
actual improvement to LFM3-Omni and show the initial-versus-final difference.
The human explicitly approved ending the earlier weights-only restriction and
running controlled inference plus reversible weight-edit trials on isolated
copies. Keep the original immutable, no gradients/training, no deployment, and
one architectural component changed per trial. Keep the existing dedicated
branch/worktree and K3 research area. Do not modify earlier frozen evidence.

Read the complete canonical Clover AGENTS.md again for this continuation. The
goal is measured improvement, not a forced favorable result. Do not relabel
geometric structure as a defect, tune against exposed validation data, or weaken
regression checks. If no candidate survives, report that outcome honestly.

## Established context

Original: LiquidAI/LFM3-Omni-2B, revision
21d75123f86c1f7ffb4bef1ea202710c4e7ebe67, checkpoint SHA-256
76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763.
The native BF16 CPU/FP32-codec runtime is already verified in this study. Reuse
it without altering the prior frozen implementation. Original artifacts are
under `/opt/clover-k3/astra-weightspace-20261111/original`.

Weights-only steps showed a common low-norm embedding family, preferential
first-projection gain for that direction, and predominantly single-tap-dominated
convolution filters. None alone proves a behavioral fault or justifies an edit.
The earlier ad-hoc final-norm intervention did not improve its held-out set;
that exposed set will not select or validate a new candidate.

## First discriminating diagnostic

Hypothesis: some wrong arithmetic-choice answers reflect position sensitivity
instead of uniformly missing arithmetic capability. Test fresh, deterministic
problems in four answer rotations and free-response form, plus simple direct
number-selection controls. Preserve actual tokens, outputs and selected
intermediate states; never replace generated answers with a restricted argmax.

A position account is weakened if free responses fail equally, rotations stay
consistent by answer value, or easy selection controls fail because of an input
or scoring defect. Validate generation/scoring plumbing before any tensor edit.
This is discovery only, not final held-out evaluation.

## Subsequent gates

After the diagnostic, document a localized causal hypothesis and exact proposed
tensor change before application. Use independent tuning and final validation
problems, frozen before trials. Select a candidate on discovery/tuning only,
then evaluate final data once under identical inference settings. Test unrelated
text and all supported modalities; name any unverified speech-quality limits.
Reject unsupported or harmful candidates. Preserve initial/final copies,
per-tensor hashes, byte-exact reversal and raw paired records. No main checkout,
production services, original checkpoint, or previous frozen study is changed.

## Outcome

Fresh diagnostic completed:114 cases, no runtime errors, original hashes intact.
MCQ60/96; free response14/18. All6 addition and6 subtraction free responses were
correct, but rotated choices yielded addition8/24 and subtraction11/24. Direct
number selection controls passed24/24. Multiplication was2/6 free,17/24 MCQ.
15 of24 problem groups changed their selected numeric value under rotation.
Predicted labels were A52,B17,C13,D14. Thus the pattern cannot be explained only
as uniformly missing arithmetic ability or an inability to select labels.

Next discriminating reference: keep the exact options and labels for each
discovery MCQ but ask which option equals the computed numeric result. These
are teacher-style discovery references, not benchmark prompts or held-out data.
Capture the same residual/MLP states and compare only cases where the unassisted
free answer and explicit numeric binding are independently correct. This may
localize default-position behavior, but prompt/content confounding remains and
must be tested by actual reversible interventions plus unrelated controls.
No weights have been modified in this continuation.

Matched references completed96/96 correct with balanced labels and no rotation
value disagreement. Fresh tuning/final suites frozen at228/456 responses on48/96
underlying questions. Original tuning: arithmetic MCQ74/144, free32/36,
selection48/48. Final outputs have not been generated or used for selection.

Proposal computation used56 eligible discovery MCQs (31failed,25successful)
whose free answers and explicit binding references were correct. Ranked matched
residual contrasts and MLP writes selected layers22 and23,64 columns each. The
documented sparse output-direction edit is tested in ONE tensor per trial at
strengths0.5,1,2. The final-RMS/embedding/head remain unchanged. Post-MLP scaling
is used only as an approximate alignment; the RMS denominator and prompt
confounds remain reasons the actual trials can falsify the hypothesis.

The first proposal job failed before execution because its ReadWritePaths mount
named a nonexistent output subdirectory (status226/NAMESPACE). The journal
confirmed the cause; rerunning the unchanged code with the existing improvement
parent mount succeeded. No model/file mutation occurred in that setup failure.

Next: six serial physical-copy tuning trials, full tensor audit and exact reversal
after each, selection by frozen tuning criteria only. Retain every trial outcome.

Initial tuning observed: layer22 strengths0.5/1 left arithmetic74/144; strength2
raised it to75/144, with free32/36 and selection48/48 unchanged. Layer23 strength
0.5 fell to73/144. This is weak evidence, not a verified improvement.

Before any follow-up results, documented a conditional extension in
DOSE_FOLLOWUP.md: only if layer22 remains the best initial qualifying candidate,
test the identical direction/64 columns at4 and6. No new tensor location or
selection data; no further automatic escalation. Initial proposals and writer
implementation remain unchanged. The extension is disclosed as adaptive tuning,
not independent evidence. Frozen scoring/cases and final retention criteria do
not change. All trial copies still require single-tensor audits and reversal.

The final driver can freeze either the original-grid or combined follow-up
selection before viewing final outputs. It preserves initial/final checkpoint
hashes and refuses to count the four rotations as independent significance
samples. A passed narrow text gate still cannot establish universal model or
speech quality; changed speech requires an independent check before retention.

Initial six-trial grid completed and all copies reversed. Best qualifying score
75/144 came from both layer22 and23 at strength2; the fixed tie-break chose22.
The conditional follow-up then completed: layer22 strength4 scored79/144,
strength6 regressed to65/144, both free32/36 and selection48/48. Both copies
reversed exactly. This non-monotone response rejects "more subtraction is better"
and prohibits further escalation under the documented plan.

Selected for final validation: layer22, same64 columns/direction, strength4.
Selection is based only on tuning and is preserved in FOLLOWUP_TRIALS.json.
The final driver freezes it before evaluating the untouched456-response set.
No held-out improvement is claimed yet. Independent Whisper-tiny.en speech
content assessment is prepared conditionally for changed speech streams; it
does not modify LFM3 or replace the primary/regression criteria.

## Final validation outcome

The frozen candidate completed the untouched456-response comparison and112-case
regression rerun. Initial -> tested candidate:
- Arithmetic MCQ134/288 ->137/288 (+1.0417 percentage points).
- Free arithmetic61/72 ->61/72.
- Direct number selection96/96 ->96/96.
- Exact one-sided sign test over underlying arithmetic questions: p0.654981.

The final reliability gate FAILED. The small aggregate difference is not evidence
of consistent improvement across underlying questions. No previously correct
unrelated/multimodal regression case broke; unrelated text15/16 ->16/16, controlled
vision12/12 and video8/8 unchanged, ASR WER0.07 unchanged. Speech-code streams
changed for all4 outputs, which remained valid/nonempty. Those facts do not
rescue the failed primary gate or establish broad speech quality.

The final candidate was automatically reversed and its complete checkpoint hash
restored to the original. All eight tuning trial copies were also reversed.
No candidate is retained or deployed; the final model remains the initial
original. The independently pinned speech observer was acquired but is NOT run
to justify retention after the primary gate failed. Its presence is not evidence.

This cycle does not meet the human's desired improved-model outcome. It does
establish a real arithmetic-to-choice inconsistency, but the tested sparse
late-MLP direction was not a robust correction. Do not continue tuning against
the now-exposed final set or weaken the acceptance rules. Further research needs
a genuinely different causal hypothesis and fresh final data, not escalation
of the rejected dose or relabeling tuning gains as final improvements.

Finish by preserving the initial/candidate/final hashes, all raw paired results,
trial reversals and the explicit negative recommendation. The user requested
continuous work, not permission to manufacture a favorable outcome.

Final artifact/evidence checks completed. Exactly72 underlying arithmetic
questions:12 better,13 worse;18 individual answers fixed,15 broken. Eight trial
audits/reversals and2964 tuning/final response records verified. The verifier's
first pass encountered integer-versus-string JSON object keys in the grouped
data; canonicalizing JSON representation fixed that check without changing any
counts or criteria. A corrupted-file and changed-count control still fails as
required. All owned experiment services exited0/inactive.

Retained model hash is the initial original:
76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763.
Rejected tested candidate hash:
0e428fe5808efa47e7f588078b8d348b8363001354a500e97df5a61110fa6078.
Evidence archive (no model tensors), verified before extraction:
eb476cd8803c30fabcccef3c28d0ac2f5eb57d753f762b8aa4ca9307950f6f63.
The local README indexes initial/candidate/retained-final results and raw evidence.
No original, main checkout, deployed service, commit or remote branch was changed.