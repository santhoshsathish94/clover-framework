# Astra: LFM3-Omni-2B weight-space study

## Current Direction: Other HybridAgents, Sequentially

The human requested the same investigation for the other two trained HybridAgent
models, one at a time, with a final branch and dataset locations for real
end-to-end evaluation. Confirm the intended checkpoints before model-specific
execution. Keep originals immutable, use separate copies, and do not equate
improved proxy losses with an improved full model. No production changes or
gradient training are authorized by this continuation.

Correction: the prior LFM3 g0.92 report overstated "zero degradation." Entry CE,
exit cosine loss and forced-rollout exit rank worsened; the frozen backbone-loss
significance gate was not met. Split membership in the original training run
was not verified. It remains a research candidate, not an accepted upgrade.
Raw records and frozen evaluation files are unchanged; all trial weights were
restored. The next models must not inherit the overstated acceptance claim.

## Latest outcome: C2-only weight improvement & non-degradation study

On 2026-11-12 the human directed focusing exclusively on C2 (`LFM3-Omni-2B-HybridSFT-S3`) to see if it can be improved without degrading it. Full report: [c2_improvement/README.md](c2_improvement/README.md); cycle record: [c2_improvement/CONTEXT.md](c2_improvement/CONTEXT.md).

Key findings across the official evaluation split (`eval32`, 31 trajectories), an unseen held-out split (`heldout32`, 32 trajectories), and 24 downstream natural + 4-step forced-latent QA cases:
- **Why aggressive `post_ln` shrinking (`0.0225x`) failed**: shrinking $R(h_t)$ by 44x degrades self-conditioned Pass-2 `backbone_loss` (`0.7181 -> 0.7834`) and `cos_loss` (`0.5941 -> 0.7454`) because the 24-layer LFM3 backbone was trained in Stage 3 on unit-RMS (`~1.0`) latent vectors.
- **Single-component `g0.92_only` (`adapter.pt` `post_ln` $\times 0.92$, backbone untouched)** improved selected averages but did not establish strict non-degradation:
  - Improved `loss`, `hybridlink_loss`, `cos_loss`, `mse_loss`, and `exit_mse_loss` on **58/58 held-out latent trajectories (`0` worse, $p = 1.86\times 10^{-9}$ on both `eval32` and `heldout32`)**.
  - Reduced mean `backbone_loss` (`-0.00159` and `-0.00163`), mean `text_ce` (`-0.00050`, $p = 0.0121$, and `-0.00051`), and mean `exit_ce` on both splits.
  - Preserved **19/24** natural QA (`0` broken, bit-identical) and **22/24** 4-step forced-latent QA (`0` broken).
- **Combined `combined_anchor1.5_g0.92`** passed 11/12 gates ($p \le 1.28\times 10^{-6}$ on `loss`, `backbone_loss`, and `hybridlink_loss` on both splits; `19/24` natural QA), but failed the forced-latent QA gate (`22/24 -> 21/24`) and was rejected.
- All physical trial copies were audited and restored byte-exactly (`c2_improvement/VERIFICATION.json`).

## Previous outcome: raw C2/C3 weights changed and tested

The human requested observation and copied-weight tests of trained C2 HybridAgent
and C3 NCM on 2026-11-12. Result: [c2c3_weights/README.md](c2c3_weights/README.md).
The actual full Omni HybridSFT-S3 checkpoint, adapter and raw NCM were copied,
inventoried and run with their relevant native contracts. C2 gain matching
reduced held-out feedback NLL from 11.4983 to 5.3439, but missed the predefined
zero-input significance gate. A two-prompt natural planner probe used no latent
steps and produced identical correct answers. C3 doubled bridge gain worsened
held-out code NLL despite a small positional token-match increase. Neither
change was accepted; all six trial copies were restored byte-exactly. No C1,
installed-model, gradient-training or deployment change was made.

## Latest outcome: decoder and plan interface follow-up

The human approved the next decoder/interface test on 2026-11-12, retaining the
separate-copy, no-training and no-deployment boundaries. Result:
[decoder_interface/README.md](decoder_interface/README.md).

Five frozen decoder policies completed 390 runs. Removing copy boosts alone
triggered the content-coverage fallback; raw decoding removed question echoes
but still answered 0/24 diagnostic questions and regressed formatting. Supplied
native answers were copied correctly; swapped answers were copied incorrectly.
No policy qualified for fresh confirmation or retention. All 197 source/copy
hashes and 101 prior result artifacts remain unchanged. The completed composite
and its frozen source/results were not modified.

## Latest outcome: copied C1/C2/C3 composite tested

The human approved the proposed tests and explicitly required a separate copy.
Execution is confined to the existing dedicated research worktree and a new
composite/ directory on K3. Record: [composite/CONTEXT.md](composite/CONTEXT.md).
No gradient training, installed-model changes or deployment is approved.

Completed: [composite/README.md](composite/README.md). The 1,174-tensor unified
checkpoint round-trips exactly; 16 native multimodal cases and 30 pipeline cases
match their references. However, the strict inference-only pipeline echoes all
24 diagnostic questions (0/24 correct versus native Omni 24/24); C1/C2/NCM
ablations do not change final text. All 197 source/copy hashes remain unchanged.
The prototype is research-only and was not promoted as an improved model.

## Previous question: combine C1, C2 and C3 into one model

On 2026-11-11 the human asked whether C1 (LFM3-Omni inference), C2
(LFM3 trained with HybridAgent for planning) and C3 (LFM3-trained NCM decoding)
can become a single model. This cycle is a feasibility assessment, not approval
to train, merge checkpoints, replace a runtime or deploy. The completed
edited-neuron-count experiment remains unchanged in improvement/neuron_count/.

Read the current Studio omni_c2_c3.py loader and computation paths in the shared
repository without modifying them, the recorded Omni training handoff, and the
existing K3 export manifest at
/opt/clover-k3/s23-omni2b/ncm/precomputed/manifest.json. Local installed model data
also has separate trunks/, hybridagent/, bridges/ and ncm/ directories.

The local structural hypothesis was that these are complementary modules, not
three interchangeable Omni state dictionaries. The discriminating check was
the loader's parameter namespaces/dimensions against the export manifest's
component groups. The source uses 2048-wide C1 states, native and Stage-3
HybridLink modules, bridges to 384-wide representations, a 768-wide 12-block
NextConceptTransformer, an 11-level VQ-VAE and a 384-wide three-block pointer
decoder. The manifest lists separate plan, bridge, VQ, NCM and pointer groups.
This supports architectural composition, not direct three-way weight averaging.

Distinguish three outcomes:
- One distributable checkpoint and custom model graph can preserve C1, C2,
  bridges and C3 as submodules without claiming that their weights were averaged.
  Packaging alone establishes no quality, memory or speed gain.
- Sharing one language backbone for inference and planning is a candidate design.
  Only identical tensors can be deduplicated losslessly. A fully fine-tuned C2
  backbone is not automatically identical to C1; equal shapes are insufficient.
  Compatible same-base language-model deltas could be tested separately, while
  keeping HybridLink and NCM modules and their required execution logic.
- Absorbing all three capabilities into the unchanged native Omni architecture
  is not established by these artifacts. Distillation or joint training is a
  plausible route, not an authorized action or a guarantee of capability or
  NCM-style parallel decoding. The earlier no-gradient boundary remains in force.

The NCM text decoder is not the Omni speech codec. Preserve original vision,
audio and speech paths in any proposed composite; text-only planner changes can
still alter multimodal behavior through the shared backbone.

Limits: no new inference, weight merge, training or deployment ran. Historical
training/provenance reports were read, not reverified by hashing every raw
checkpoint. Exact base revisions, current tensor/tokenizer compatibility,
feature-space alignment, multimodal retention and combined quality remain open.
Next useful check, if the human chooses implementation: pin the three actual
artifacts, inventory tensor overlap and state/tokenizer contracts, establish
the existing pipeline baseline, then verify a composite's parity and ablations
before attempting shared-backbone substitutions or learning.

## Previous direction: continue through controlled improvement validation

The human now requests continuous work toward a better LFM3-Omni and an
initial-versus-final comparison, without returning for approval after each step.
The human explicitly approved controlled inference and reversible edits on
isolated copies, ending the earlier weights-only restriction for this new phase.
Keep original weights immutable, no gradients, no deployment, one architectural
component changed per trial, fresh held-out validation and regression checks.
Active record: `improvement/CONTEXT.md`. Prior studies and their frozen evidence
remain unchanged; the Qwen task stays superseded.

## Previous direction: LFM3 weights only, one step at a time

The human redirected this task on 2026-11-11: use the paper-derived understanding
to inspect LFM3 open weights ONLY, step by step. No new inference, activation,
prompt, tokenizer-label or benchmark evidence in this examination. First step
and ongoing understanding are in `weights-only/CONTEXT.md`. Original model
weights remain immutable and the existing branch/worktree isolation is retained.

Stopped this session's remaining Qwen original inference service in response to
the redirection, preserving its logs. Its completion notification must not resume
the superseded paper task or be mistaken for a completed paired inference result.

## Previous direction: reproduce the research before extending it

Human correction, 2026-11-11: first obtain the research paper and its code on
K3, do the actual work to understand the method, and preserve the evolving
understanding in a context file. Reading or summarizing is not reproduction.

The earlier tensor survey and H1 experiment were an exploratory pilot designed
in this session, not a reproduction of a named paper. They establish only the
recorded measurements and negative H1 result; they do not establish that a
published intervention method has been understood or tested.

The initial missing source was resolved by the human:
https://rikiyat.github.io/blog/weight-space-programming/ . The article and its
official experiment repository are now acquired on K3. The reproduction record
is `reproduction/CONTEXT.md`, with pinned source commit and execution checkpoints.
The authors explicitly disallow final-norm edits in the relevant math experiment;
the earlier final-RMS pilot must not be represented as reproducing their method.

Once identified: download the paper and clone its official code into the isolated
K3 research area; pin source revisions and hashes; inspect the implementation
and required data; reproduce an applicable authors' experiment unchanged before
adapting it. If the original method requires training, additional hardware,
access, or changes outside the established boundaries, surface the conflict
before execution. Record the initial interpretation, exact run, observed result,
what that result changes in the interpretation, and the next discriminating
experiment after each meaningful cycle. Keep the existing worktree isolation,
immutable originals, copied-checkpoint controls, and frozen pilot evidence.

## Direction and boundaries

The human requested this study on K3 on 2026-11-11 under Clover. The objective
is to identify weight-space structures associated with measured behavioural
weaknesses and evaluate reversible, non-gradient interventions.

The human subsequently required a new branch and independent worktree.
Branch: `research/astra-lfm3-weightspace-20261111`.
Worktree: `C:\personal\oss\clover-astra-weightspace-wt`.
Base commit: `2ef6786b7634b5100f08a2d6b43754a89f782217`.
Study path: `research/astra-lfm3-weightspace`.
K3 isolated directory: `/opt/clover-k3/astra-weightspace-20261111`.
The shared main checkout was not switched or edited. This session's bootstrap
artifacts were moved from `C:\personal\scratch-agents\astra-lfm3-weightspace`;
the inventory checks passed again at the new location. No commits or pushes.

- Original checkpoint files are immutable inputs. Never overwrite them.
- Start with read-only tensor and activation analysis. Outliers are observations,
  not defects. Document an evidence-linked mathematical hypothesis before edits.
- Test one tensor/component intervention at a time in an independent copy.
- Freeze inference settings, test cases, scoring, and discovery/validation splits
  before comparing arms. Do not tune against held-out outcomes.
- No production service changes, trained Clover substitutions, customer data,
  credential reads, gradient training, or deployment. No commits unless asked.
- Use an isolated K3 research directory with bounded CPU/memory consumption.

## Context established

Read the complete canonical Clover AGENTS.md (22 sections) this session, and
the relevant stored K3 and LFM3 context. Connected to SSH alias `k3` read-only.
`/opt/clover-k3/s23-omni2b` exists but contains a derived phone runtime, pruned
weights, and NCM artifacts. It is not yet established as the original checkpoint.
The existing interpreter `/root/ncm-cpu/venv/bin/python` is available by pathname;
its packages and the live machine resources have not yet been checked.

## First cycle

Hypothesis: safetensors headers and checkpoint configuration can supply an
exhaustive stored-tensor inventory without loading or modifying model weights.
Discriminating check: validate byte coverage, sizes, tensor names, and shard-index
agreement, plus self-tests with known and deliberately malformed headers.

Inventory self-tests passed 6/6, including malformed headers and source-directory
write rejection. Editor diagnostics reported no errors. No checkpoint tensors
have been modified.

K3 is an AMD Ryzen 9 7950X3D (32 logical CPUs) with native AVX-512 BF16. Observed
available RAM approximately 80 GiB and disk free approximately 702 GiB. Existing
production processes are present and remain untouched. Plan: 8 analysis threads,
nice 15, memory cap 24 GiB, and a read-only filesystem sandbox for analysis.

Checked the known model stores and found a trained Omni HybridSFT-S3 checkpoint,
but not the original Omni snapshot. A pinned public `LiquidAI/LFM3-Omni-2B`
snapshot will be fetched into the study directory, with upstream Git/LFS hashes
verified and a sealing manifest. Config reports 24 text layers, 14 vision layers,
17 audio-encoder layers, and 3 speech-depth layers; source/runtime mapping remains
to be checked. The terminal API display was truncated and could not be parsed;
`prepare.py` preserves and parses the response directly on K3 instead.

## Original source established

Pinned revision: `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.
Main model: 4,333,941,640 bytes, upstream SHA-256
`76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`.
Audio codec: 147,143,540 bytes, upstream SHA-256
`45984a872fdcf0b94cd0ec68822bbf26f0c6fcb580cfaba0f7fa3ee60355137f`.
Metadata/source files were hash-verified before code inspection. The capped
snapshot-fetch unit exited 0; the sealing manifest is the next verification.

Source findings, not behavioural claims: the shared text embedding is tied to
the text output head; image and video use the same vision tower/projector;
audio input uses a separate encoder/projector; speech output uses a three-layer
depth decoder with 12 individually tied codebook embeddings/heads, and a separate
waveform codec. The text blocks have sandwich RMS norms, so rescaling a projection
before a norm is not generally equivalent to scaling its residual contribution.

## Next cycle

Run the inventory self-test, locate the original complete Omni snapshot, inspect
its matching source/configuration and available resources, then record the
immutable input manifest before analysis. Do not reuse application or trained
Clover benchmark scores as a baseline for the original checkpoint.

## Inventory outcome and survey correction

K3 inventory completed successfully in 2.10 seconds within a read-only source
mount. Main checkpoint: 665 tensors, 2,166,928,384 stored parameters. Codec: 203
tensors, 73,559,090 stored elements. All main tensors have a source-mapped
component. Inventory self-tests: 8/8; numerical survey self-tests initially 5/5.

The first full survey failed in fingerprint sampling: float32 `torch.linspace`
rounded 49,569,791 to 49,569,792 and indexed past the last element. This is a
research-instrument defect, not a model defect. Sampling indices now use int64
arithmetic, with a regression test at the failing tensor size. Preserve the
failed `reports/weight-survey` directory; rerun as `weight-survey-v2`.

The independent venv's media-dependency installation exited 0. The shared venv
was only copied, never installed into. Weight analysis has not yet completed;
no distribution finding or behavioural improvement is established.

## Completed read-only weight analysis

Corrected survey completed: 868/868 tensors, 33.40 seconds on K3 at 8 threads.
Original file hashes matched before and after; an attempted write-only open
without truncation was refused by the read-only mount. Nonfinite elements: 0.
The survey includes exact moments, row/column norms and six-standard-deviation
concentration counts, sampled quantiles and correlations, and exact small-matrix
or explicitly truncated randomized large-matrix spectra. These are descriptions,
not defects or improvements. Artifacts: `weight-survey-v2/` locally, matching
`reports/weight-survey-v2/` on K3. The failed v1 run remains on K3.

The first real-model smoke stopped before inference: PyPI supplied torchvision
0.26.0 with CUDA dependencies to CPU-only torch 2.11.0. `ldd` showed missing CUDA
libraries. Installed torchvision 0.26.0+cpu only in the independent venv, without
changing torch; native-operator availability and AutoProcessor import now pass.
Rerun smoke in a separate result directory. This was an environment defect, not
a model failure. No behavioural baseline or weight intervention has run yet.

The second smoke loaded all 665 model and 203 codec tensors with no reported
missing/unexpected weights. Its first generation exposed an observer bug: the
custom generator calls `lm_head` on one row (1D output), while the hook assumed
the 3D full-forward shape. The hook now takes the last vector independent of
leading dimensions; a test covers 1D and batched output. Preserve the v2 failure
log and retry as v3. Do not classify either instrument/setup failure as a
checkpoint weakness. No baseline cases have been run.

Text smoke v3 passed: prompt `What is 7 plus 8? Answer with only the number.`
returned `15` (token IDs 885, 124900), identical with/without hooks. All selected
text hooks recorded activations, parameter version counters stayed unchanged,
and original hashes matched after inference. This establishes the text runtime
and one observational neutrality check, not general capability.

Built 112 materials on K3: 80 text (32 discovery, 32 validation, 16 unrelated),
12 vision, 8 video, 6 ASR, 2 non-speech audio, 4 speech. Fixture self-tests 4/4;
benchmark accounting tests 3/3. Speech data is publicly pinned and selected only
by source order, rate and duration. Benchmark case/media hashes are recorded;
the full runtime/scorer seal waits for the independent API pilot. Codec inference
is explicitly FP32 (exact widening of stored BF16), as preregistered; source
bytes remain immutable. No target intervention has been chosen.

Media pilot v1: image, real-utterance ASR, and native speech-code/waveform output
ran successfully; video was rejected before inference because sampled frame
indices were absent from metadata. The native processor requires `frames_indices`
even when `do_sample_frames=False`. Supply the exact indices 0..7 for the unchanged
2-fps eight-frame fixture, and add a clock-metadata test. No question, answer,
scoring rule, or media bytes changed. Retain v1 results and rerun as v2 before
sealing. `systemctl show` confirmed both original and fixtures read-only mounts.

## Frozen baseline and documented candidate

Media pilot v2 completed all four paths with no errors. Sealed benchmark SHA-256:
`0815a634fa3ee4cff61a707ab6934728f036daa4150014b317a4648fde557470`.
The 112-case baseline completed in 60.89 seconds, no runtime errors, no token-cap
hits, original hashes unchanged. Discovery arithmetic 15/32, validation 13/32;
unrelated text 15/16; vision 12/12; video 8/8; non-speech audio controls 0/2;
ASR 1/6 exact, word error rate 0.07; all four native TTS waveforms nonempty.
Waveform integrity is not independent intelligibility or naturalness evidence.

The preregistered discovery-only rule selected final RMS gain coordinate 36,
original 3.03125. Failed-example mean margin contribution -0.0835954; successful
examples +0.00641696. Predicted 10% changes: +0.00835954 and -0.000641696.
This is a small exploratory association across 2048 coordinates, not a defect
diagnosis. HYPOTHESES.md documents the derivation, indices, risks, strengths and
reversal before patch specs were generated. H2/H3 are untested lower-ranked
alternatives; they will not be tried to rescue a failed H1 validation.

At that point the original model tensors were still untouched. The next cycle
was the no-op and H1 5/10/20% validation described below. The sealed scorer,
cases, environment and inference code were not changed during comparison.

## Final controlled outcome

Completed original + no-op + three strengths: 560 case evaluations, no runtime
errors, no token-limit hits in the complete suite. No-op matched 112/112 inputs,
generated tokens, letter logits and speech code/waveform hashes, plus all 1,912
activation arrays. Each H1 strength changed exactly one tensor/one BF16 gain;
1,680 upstream activation arrays remained identical, with 211/227/231 changed
downstream arrays at 5/10/20%. The intervention was live, not a no-op.

All arms retained discovery 15/32 and validation 13/32, with zero fixed/broken
validation answers (exact paired p=1). Unrelated text 15/16, vision 12/12, video
8/8, non-speech audio 0/2, ASR exact 1/6 and WER 0.07, nonempty speech 4/4 were
unchanged. Primary 10% edit changed letter logits in 99 cases and all four
speech-code streams; mean failed-discovery margin rose 0.0172335. That numerical
change is not a verified behavioural improvement.

Each experimental copy was reversed immediately after its run. Full checkpoint
SHA-256 matched the original after each reversal and again at final reporting.
Other checkpoint files and original source hashes remained unchanged. The owned
experiment service exited 0 and is inactive; no production services were touched.

Final recommendation: **retain the original; reject H1 at the tested strengths**.
H2/H3 remain untested, not alternative successes. No commits, pushes, merges or
deployments. No gradients, conventional fine-tuning, or multi-component changes.

Limits: this is one pinned model, one CPU/inference configuration and a small
constructed benchmark. Speech intelligibility/naturalness were not independently
scored; vision/video are controlled fixtures; audio has only six real utterances
and two synthetic controls. Discovery success/failure contrasts are confounded
by answer position (A: 8/8 correct, C: 1/8). Last-token layer-0 contrast is zero;
the shared prompt suffix and the first short-convolution layer's limited receptive
field are consistent with this, not evidence that all early representations are
constant. The experiment does not rule out other weight-space interventions.

Growth: describe statistical structure separately from defects; preserve the
small predicted margin effect as a hypothesis, not an accuracy promise. Do not
choose H2/H3 against this now-exposed validation set. A next study should first
separate answer-position bias from arithmetic competence using new counterbalanced
and free-response examples, and separately assess audio/speech fidelity.

Evidence archived from K3 (weights excluded), SHA-256:
`c89554b9e7a9fac4f0f55c47c7c6a42c865203b2d1f318c93d97c37821e226bd`.
Transferred via temporary path and hash-checked before extracting reports,
fixtures and copy audits into this worktree. The README indexes all seven
deliverables; `verify_handoff.py` checks the preserved artifacts independently of
model execution. Large local artifacts are Git-ignored, not deleted.