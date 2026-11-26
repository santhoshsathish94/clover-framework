# C2/C3 Weight Observation and Intervention

## Direction

On 2026-11-12 the human redirected the work from C1 toward observing the trained
C2 HybridAgent and C3 NCM weights, then changing and testing isolated copies
before suggesting improvements. Keep the dedicated research branch/worktree and
new K3 c2c3_weights/ directory. Original models, live services and all earlier
studies remain unchanged. No conventional gradient training or deployment is
authorized by this direction.

The full Omni planner was located at
/opt/clover-k3/models/LFM3-Omni-2B-HybridSFT-S3, including model.safetensors,
adapter.pt, training metadata and an earlier inference smoke record. The raw C3
checkpoint exists at /opt/clover-k3/s23-omni2b/ncm/lfm3_ncm_core_bf16.pt, with a
local archived counterpart and NCM runtime under the prior training artifacts.

## First falsifiable hypothesis

The learned adapter/bridge/head scales may explain weak or mismatched signals
that exported-runtime copying previously hid. Tensor shape, finite-value and
norm observations can identify a concrete candidate, but an outlier is not itself
a defect. Inventory the raw checkpoints before selecting an intervention.

The cheapest initial check is a weights-only, bounds-checked inventory of the
actual C2 adapter and C3 tensor groups, with known-value numerical tests and
source/copy hashes. Then read the relevant training/runtime contract for the
measured candidate, test a reversible change on a physical copy against an
unedited/no-op control, and validate with data not used to select that change.
Preserve unfavorable results; do not reuse old question scores as held-out proof.

## Raw observation and first loader check

Physical copies were created for the full C2 backbone (286 floating tensors),
C2 adapter (8) and C3 core (362). The observer's three numerical controls passed;
all source hashes remained unchanged. C2 post-LayerNorm weights have RMS 1.0.
C3 native-link output scale is 0.013660; combo-bridge scale 8.49218; pointer copy
scale 18.53294. These are observations, not proof that any value is defective.

The full C2 checkpoint loaded successfully. C3's first strict load exposed 22
EMA buffers omitted from the inference export. The copied quantizer source
shows EMA updates under self.training and uses actual codebook embeddings for
inference. The loader now permits exactly those named training-only buffers,
initializes them without changing any inference parameter, and retains strict
loading for all other state. Training remains disabled. Preserve this first
failed load as an export-contract finding, not a model failure.

## Runtime baseline and preregistered interventions

The corrected strict load passed with originals inaccessible: full C2 backbone,
2048-dimensional link, and C3 VQ/transformer/pointer/native-link/bridge modules.
Only the 22 inactive EMA buffers were initialized; all inference weights loaded
from copies. On the smoke input, C2 link output RMS was 0.99669 and token-input
embedding RMS 0.023439. The roughly 43x magnitude mismatch is a hypothesis about
feedback calibration, not proof that matching norms improves planning.

Interventions use separate physical checkpoint copies and before/after hashes:
- C2: scale only post_ln.weight and post_ln.bias by a nonnegative least-squares
	gain or an RMS-matching gain fitted on 20 discovery texts. Compare the two
	actual rounded checkpoints and an unedited/no-op copy; select by discovery
	feedback NLL. Test the selected copy on 20 disjoint texts against original,
	zero-input and true-next-token-embedding controls. The endpoint is token t+2
	likelihood after one link-feedback input for token t+1 through the full C2 LM.
	This is analytic calibration, not gradient fine-tuning or proven end-to-end
	planning improvement. Require paired significance against original AND zero.
- C3: scale only lfm3_combo_bridge.scale by 0.5, 1 or 2. Select on discovery code
	NLL, then require held-out paired NLL improvement and no loss in generated
	nonpad-token accuracy. Use the saved 48+48 packed prefix/mask, all 11 levels,
	64-token target, native generation code and VQ decoder. No copy bonuses or
	pointer fallback can turn a generated error into a reported success. Report
	the oracle VQ encode/decode accuracy as a ceiling, not as generated accuracy.

The first discovery text was used for a layout smoke only. It passed both
objectives; C3 generated 2/14 target tokens, with oracle reconstruction 9/14.
The two 20-text sets are distinct but deliberately similar-domain short notes;
held-out means held out from this calibration, not guaranteed unseen in original
model training. No held-out outcome has been viewed before freezing selection.

## Completed weight trials

The frozen run completed in 504.075 seconds on K3 CPU. Separate physical
checkpoints were written and audited: C2 changes only post_ln.weight/bias;
C3 changes only lfm3_combo_bridge.scale. No-op copies have no changed tensors.

C2 discovery fitted least-squares gain 0.00836702 and norm-matching gain
0.02350780. Discovery feedback NLL selected norm matching before held-out text
was evaluated. Held-out mean next-token NLL: original 11.49832, norm-matched
5.34387, zero input 8.06574, true-token input 0.08853. Per-text comparisons:
19 better/1 worse versus original (one-sided sign p=0.00002003), 14 better/6
worse versus zero (p=0.057659). The predeclared gate required p<=0.05 against
BOTH, so it failed. The strong improvement against the original is a promising
calibration observation, not permission to waive the zero-input control.

C3 discovery selected doubled bridge scale. Held-out code NLL worsened from
1.297129 to 1.319102; 5 texts improved and 15 worsened (p=0.994091). Generated
nonpad-token matches increased from 118/231 to 124/231, while VQ oracle
reconstruction was 171/231 in both arms. The joint gate failed; the token-count
increase cannot override worse held-out code likelihood.

These feedback texts ask the model to restate a supplied note and then examine
continuation positions; they do not measure solving a new planning problem.
Both weight changes remain unaccepted. A separate readiness probe will check
whether the natural HybridAgent loop actually uses the C2 adapter. That probe
does not reopen the failed calibration gate or turn this into end-to-end proof.

## Natural planner usage probe

The frozen upstream HybridAgent loop ran two deterministic prompts with the
original and norm-matched adapters, a 96-step budget, latent limits 4..32, and
zero permitted tool calls. Both prompts ended at EOS for both variants. All
four runs took zero latent steps and the paired output texts were identical.
The check completed in 18.139 seconds; no tool executed. It is a readiness
probe on two simple prompts, not evidence that the adapter is never used.

This explains why the one-step calibration result alone cannot be presented
as improved planning. Before adopting a gain change, establish representative
natural latent usage and a fresh task-level improvement, while preserving the
failed zero-input gate result above.

Terminology qualification: the recorded VQ encode/decode result is an oracle
reconstruction reference, not a mathematical upper bound on every possible
generated code sequence. It is useful for diagnosing representation loss but
does not prove that a generator cannot exceed that reconstruction score.

## Audit and reversal complete

The independent auditor passed its three corruption/aggregation controls and
verified discovery-based selection, held-out means and paired statistics,
declared changed tensors, and actual factor-scaled values. All 18 provenance
records (raw checkpoints and runtime metadata/copies) retained their hashes.
Every one of the six trial files was restored byte-exactly, with REVERSAL.json
beside its preserved PATCH_AUDIT.json. No-op files remained identical throughout.

Natural planner outputs were read: both prompts correctly returned 43 and Rome,
but with zero latent steps for either adapter. Recommendation: C2 gain matching
is a promising calibration direction needing fresh task-level/zero-control
validation; the tested C3 bridge gain is rejected. Investigate VQ reconstruction
and generalization before further C3 gain edits. No model upgrade was retained.

## Provenance and handoff

The raw K3 C3 checkpoint matches the local archived Omni-trained checkpoint
byte-for-byte: SHA-256
9a03a682b1b614a239f9e6c92276a96ad1e188e2d8bfe4f90c945cd752d16255.
C2 backbone SHA-256 is
581239236b5966bc5f3883d28d81a061dca3718dfbdba5e11738fdefb97e9ad2;
its adapter is f5f3ee236dfcc4ffb895726960824ae28f2201c998da88d39fdcdcc83035fd05.

The evidence-only archive was verified before local extraction:
9cebb5dc05d2a0e73c5d5a729faaa08a56c6b337e0cd578df9f95d9cc1742530.
Frozen cases/source hashes, results, planner usage/source, and patch/reversal
audits are preserved in the local worktree and K3. No weight files were included
in the handoff archive. All owned jobs finished successfully after the recorded
loader correction. No inference remains running, and no commit, push or deploy
was made. Restored trial files contain original weights, not accepted upgrades.