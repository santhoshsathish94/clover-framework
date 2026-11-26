# Weight-Space Programming: reproduction and developing understanding

## Superseded direction

The human redirected the task to LFM3 weights-only inspection, step by step.
The remaining owned Qwen original inference service was stopped on2026-11-11;
its logs were preserved, but no completed paired conclusion is claimed. Do not
resume inference from the old completion notification. Current work and evidence
boundary: `../weights-only/CONTEXT.md`. Earlier cycles below remain historical.

## Direction

The human supplied https://rikiyat.github.io/blog/weight-space-programming/ and
asked to clone the research onto K3, do the actual work, and update understanding
from execution rather than equating reading with understanding. Continue in the
existing isolated branch/worktree. Preserve all earlier frozen pilot evidence.
No production changes, original-checkpoint writes, or gradient training.

## Acquired primary sources

- Article: "Sparks of Weight-Space Programming: LLMs can natively read & write
  neural network weights", Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab,
  published 2026-11-02. This source describes itself as a blog post, not a
  separately identified archival paper.
- Official release: https://github.com/RikiyaT/lm_understand-logs
- Cloned on K3 at
  `/opt/clover-k3/astra-weightspace-20261111/reproduction/lm_understand-logs`.
- Pinned commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.
- Article bytes downloaded on K3 beside the clone as `article.html`.
- Requested sparse checkout: experiments 5 and 8 plus the shared harness.
- Current K3 headroom: approximately 80 GiB available RAM. Use bounded CPU jobs,
  an independent reproduction environment, and read-only mounts for references.

## Interpretation before doing the reproduction

These are statements read from the authors' source, not reproduced findings:

- Weight-space programming specifies what a network should compute, diagnoses
  its observed behaviour, and changes weights to implement that computation.
  It is not a rule to remove statistically extreme weights.
- Experiment 5 supplies exact one-shot edit blocks and resulting 600-32-32-6
  tanh MinAtar policies. Their deterministic forward equations and environment
  source are disclosed. Published evaluation uses MinAtar 1.0.15, sampled
  softmax actions, sticky actions 0.1, 1000 episodes, 10000-step cap. PPO baseline
  checkpoints and the authors' edit parser/evaluation runner are not listed as
  released files; their reported PPO scores are not our measurements.
- Experiment 8 identifies repeated reasoning in Qwen3.5-0.8B and targets two
  MLP output directions. It explicitly forbids final-norm and embedding/head
  edits. The earlier LFM3 final-RMS coordinate pilot did not follow this method.
- Published Qwen evaluation uses thinking, vLLM, temperature 0.6, top-p 0.95,
  top-k 20, presence penalty 1.5 and 32768 new tokens, with multiple samples.
  A shorter, greedy, CPU or smaller-sample test must not be labeled the same
  benchmark. The 50-pass edit touches columns in two MLP layers (15 and 19).
- The release includes exact Qwen edit diffs and checked reconstruction code,
  but excludes the original interactive agent/forward-pass harness. Replaying
  released artifacts cannot establish that we reproduced Astra's discovery.
- The Gemma follow-up reports no established math improvement for ordinary
  reasoning errors without extensive looping. This is a published negative
  observation, not a conclusion about LFM3.

## First executable hypothesis and discriminating check

Hypothesis: the published Breakout edit block deterministically reconstructs
the supplied six float32 arrays, and the resulting feed-forward network acts
as a landing-position controller when run in the unmodified MinAtar environment.

First check: parse the disclosed edit language into a new zero network and
require exact per-array identity with `networks/breakout/hand.npz`. Then run
the reference and reconstruction on identical observations, verify logits and
sampled-action behaviour, and evaluate actual episodes under declared settings.
Do not claim the published mean was reproduced merely from reading its JSON.

For Experiment 8, inspect and run the released reconstruction/score-merge code,
then choose an execution budget based on measured cost. The original constraint
against simultaneous multi-component edits required clarification for the
published two-layer reference. The human explicitly approved on 2026-11-11:
"go ahead use the qwen to understand the research fully" in response to the
request to replay that exact reference on an isolated Qwen copy. This authorizes
the published Qwen reconstruction and evaluation, not integration into Clover,
production changes, or relaxing the single-component LFM3 experiment boundary.
Original Qwen and LFM3 snapshots remain immutable; no gradient training.

## Outcome and growth

### Cycle 1: exact Breakout construction and local computation

Executed `replay_network.py --root .../reproduction` on K3 after four parser and
channel-mapping controls passed. The script accepts only literal numeric tensor
assignments via Python's AST, never executes the historical answer as code, and
rejects any invalid statement instead of silently dropping it.

Observed: all 92 published statements reconstructed all six arrays bit-for-bit
against `networks/breakout/hand.npz`: 20486 parameters, 815 nonzero. The saved
result includes every tensor hash, the source answer hash and reference hash at
`results/breakout-reconstruction/result.json`.

Ten controlled forward probes held the ball at row 7/column 4 and its trail at
row 6/column 3, varying paddle position from 0 through 9. Both published and
reconstructed networks produced identical logits. The network moved right for
paddle columns below 5, stayed for columns 5 and 6, and moved left above 6,
matching the predicted next catch interval. This is an observed computation,
not merely the author's explanation repeated back.

What changed in understanding: the weight program combines a spatial lookup
table, subtraction of paddle position, saturated direction signals, and tanh
gates. Large coefficients implement gates; their being statistical outliers
does not make them defects. The computation can be traced to specific units and
tested on inputs. This does not yet establish game-playing performance or
transfer to LFM3.

MinAtar 1.0.15 installed in `reproduction/venv`, independently copied from the
existing environment. The frozen parent venv was not modified. Next: validate
the installed game source against the source supplied in the published prompt,
then actual seeded episodes under the published sticky-action/scoring rules.

### Cycle 2: real environment smoke and Qwen source acquisition

Executed the independent evaluator for two seeded Breakout episodes. Installed
MinAtar 1.0.15's game AST exactly matches the Python source in the author's
prompt (source SHA-256
`35fd842751f4314a427623da24c3ca36dc07dfcbbd979e9c72c1079b933bcf5d`).
Reconstructed policy scores: 491 and 727 (mean 609, SE 118); zero-network control
scores: 0 and 0. This demonstrates live game-playing and an effective negative
control, not reproduction of the 1000-episode published mean. References were
unchanged. Next run uses 1000 episodes; the author's unreleased seed schedule and
evaluator mean this is a disclosed-settings independent behavioral reproduction.

Acquired Qwen/Qwen3.5-0.8B at the authors' exact base revision
`2fc06364715b967f1860aea9cf38778875588b17`. Its single shard is named
`model.safetensors-00001-of-00001.safetensors`, 1746942600 bytes, SHA-256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
All 12 selected Git/LFS files verified; original sealed under
`models/qwen-original`, manifest `models/QWEN_SOURCE.json`.

The published p50c diff declares 65260 and 64928 changed BF16 entries in layer
15/19 MLP down projections respectively, shape 1024x3584, 64 columns each. Its
130188 changed entries are fewer than the 131072 touched entries because some
changes round back to the original BF16 value. That distinction must survive
our reconstruction and reports. Next: run the authors' unmodified hash-asserting
`apply_edit.py` on the original shard into a separate output file.

### Cycle 3: full Breakout evaluation and exact Qwen patch replay

The 1000-episode Breakout run completed on K3 in 133.00 seconds. Measured mean
502.88, SE 7.7474, versus published mean 517.411, SE 7.4170: difference -1.35
combined standard errors. 100 zero-network control episodes averaged 0.41, SE
0.0621. Same released float32 weights, source-verified MinAtar 1.0.15, sticky
actions 0.1 and 10000-step cap; independent seeded evaluator because the original
seed schedule/runner were not released. This supports the high-scoring neural
program's actual behavior, not exact equality of sampled means or a rerun of
the authors' PPO baselines. Trace and animation are saved with per-episode rows.

Ran the authors' unmodified `exp8_math/code/apply_edit.py` against the pinned
original Qwen shard, writing only `models/qwen-p50c.safetensors`. Both base and
edited tensor assertions passed. Layer 15: 65260 changed BF16 entries, edited
hash `82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53`.
Layer 19: 64928 changed BF16 entries, edited hash
`8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2`.
This is exact evaluated-tensor reconstruction, not yet new Qwen behavior evidence.

The official evaluator imports an unreleased `weightenv` package, and its native
backend is GPU/vLLM. Its math checker also imports an unreleased multiple-choice
loop helper. Do not silently fabricate these dependencies or describe a CPU
adapter as the authors' original evaluator. Preserve the source grading
functions where possible, explicitly test the adapter, and label backend/sample
differences. Next: original-model CPU timing smoke, then bounded paired tests
with the author's math prompt/sampling settings and exact reconstructed weights.

### Cycle 4: independent Qwen audit and full-cap paired run

The independent tensor audit found 488 tensors in the original full release:
486 unchanged, exactly the two declared MLP tensors changed, 130188 BF16 entries.
Each changed-column list exactly matches the author's selected-neuron manifest.
Recovered leading edit directions have absolute cross-layer cosine 0.8490049:
they are related layer-specific directions, not one identical global vector.
Direction spectra and vectors are saved in `results/published-audit/`.

Executed the authors' unchanged `merge.py` on copied score records: unedited,
p10c, p25c and p50c merged JSON exactly matched the release. Thus the reported
MATH-500 7.9% -> 48.1%, GSM8K 13.8% -> 67.1%, IFEval 21.0% -> 62.3% are backed
by internally consistent released counts. This is record recomputation, NOT
newly reproduced benchmark inference. The final unedited GPQA number is 22.7%,
not the blog figure's interim 24.5% for 149/198 questions.

The author final report gives the precise formula
`W' = W0 - 6 * outer(u, mask64 * (u @ W0))`, with each layer's unit u derived from
four matched productive/repetitive original contexts, and neurons selected using
original fwd4. Alpha=6 means over-subtraction, not orthogonal projection with
alpha=1. Test that interpretation against the actual BF16 diff next.

Qwen CPU timing smoke loaded its native conditional-generation model without
missing/unexpected parameters; first published diagnostic `gsm8k_train_2805`,
128 tokens in 8.697 s (14.72 tok/s), cap hit before thinking closed. This is not
an accuracy measurement at the paper's cap. Published grading functions were
extracted as unchanged AST definitions (unreleased mcqa helper not fabricated),
with math-verify 0.9.0; positive/negative checker and generated-token-only presence
penalty tests passed. Removed unused inherited Hydra/OmegaConf from the separate
reproduction venv to resolve ANTLR conflict; `pip check` passes. Parent frozen
venv unchanged.

Running paired original/edited Qwen on that same first diagnostic, n=1, full
32768-token cap, thinking enabled, temperature .6 / top-p .95 / top-k20 /
presence penalty1.5, seed0. Native Transformers CPU SDPA, not original GPU/vLLM;
RNG/backend differences and single-item scope must be explicit. Units:
`astra-paper-qwen-original-full-70cb`, `astra-paper-qwen-edited-full-70cb`.
Logs: `qwen-original-full.log`, `qwen-edited-full.log`; results folders of same
name without `.log`. Both capped at90 minutes, 8CPU quota,24GiB,no swap,nice15,
network disabled, all models and author source mounted read-only. At1024 tokens
both took about92-93 seconds concurrently. Do not claim either completed yet.

### Cycle 5: test the operation rather than repeat its description

Executed `audit_published.py --operator` after a hand-derived alpha-six control
and rank-one/full-rank tests passed. From the exact BF16 diff, fitted alpha:
6.000321 (layer15), 6.000173 (layer19); post-edit directional ratios:
-5.000321 and -5.000173. Relative error of the alpha6 formula is 0.002348 and
0.002944; orthogonal-change fractions 0.002337 and 0.002936. Every unselected
column is bit-identical; inactive selected neurons give zero local output delta.

Understanding changed: "remove the direction" is an imprecise shorthand here.
The published final patch sparsely reverses/amplifies the directional component
on selected writers; it does not simply project it to zero or delete neurons.
The measured geometry supports the documented alpha6 operation, with small BF16
rounding leakage. Behavioral improvement still requires the generation test;
operator fidelity alone is not a loop-reduction result.

The exact original fwd4 request has been retrieved for inspecting the diagnostic
contexts used in writer selection. Do not infer unavailable activation values
from descriptions; regenerate applicable probes and compare measured directions.

### Cycle 6: real matched-context activations and a corrected interpretation

Teacher-forced replay ran all eight disclosed raw texts on original and exact
edited Qwen. Token lengths exactly matched the authors' recorded fwd4 lengths.
Recovered repetition directions agree with edit-derived directions at cosine
0.9999839/0.9999777. Observed mean last36-token log-likelihood changes: productive
-0.0114771, repetitive -0.1091974 nats/token. Each of the four repetitive branches
decreased more than its matched productive branch. This supports selective local
effect on the disclosed diagnostic contexts, not a held-out generation score.

The first replay matched only58/64 and51/64 published writer columns. Returning
to the recorded code exposed MY interpretation error: direction construction
uses the last36 residual tokens, but the FINAL top64 writer selection uses the
last24 MLP tokens. The initial code used36 for both. Do not attribute that gap
to CPU/GPU differences. Preserve `results/qwen-teacher-probe` and rerun as v2
with36 for directions and24 for writers, recording both window lengths. This
is why actual reconstruction is a stronger understanding check than reading.

Corrected v2 outcome: 64/64 selected columns recovered at layer15,63/64 at layer19.
Direction cosines and productive/repetitive likelihood deltas are unchanged.
One remaining selected-column mismatch is reported, not silently replaced by
the known manifest or attributed to a backend without evidence.

Next discriminating control, specified before running it: seeded random direction
orthogonal to each measured edit direction (seeds20261111+layer), same published
64 columns, same Frobenius perturbation norm before BF16 rounding. Measure the
post-round norm explicitly. Evaluate the same eight contexts; no seed or control
is selected by observed outcomes. This tests whether generic perturbation of
similar size explains the selective likelihood effect. Only in-memory Qwen
research instances change; original and reconstructed disk checkpoints remain
read-only. Preserve v1/v2 and run this as a new named result.

### Cycle 7: orthogonal control and remaining column discrepancy

The fixed orthogonal, norm-matched control completed on all eight contexts.
Mean log-likelihood change (nats/token): targeted edit productive -0.0114771,
repetitive -0.1091974; control productive +0.00696614, repetitive -0.000648015.
This supports directional specificity on these four disclosed diagnostic pairs;
it is not a population estimate or a held-out accuracy measurement. Pre/post-BF16
control norms are saved in `results/qwen-teacher-control/result.json`.

The one remaining layer19 ranking difference is concrete: our ranking includes
132 instead of the published2248. We did not substitute the known answer or
claim exact column-selection reproduction. Layer15 matches64/64, layer19 63/64;
the actual applied published checkpoint still matches ALL declared tensor hashes.

Next: re-grade released MATH-500 first-sample text using the unchanged selected
published checker functions, comparing every result to the recorded first-sample
label. This validates recorded-answer evidence separately from score aggregation
and from the new original/edited CPU generations still running at the full cap.

### Cycle 8: independently re-grade released answers

Re-graded all available MATH-500 first-sample text with the unchanged selected
published extraction/equivalence functions: unedited15/200 correct, p50c92/200,
zero disagreements with recorded labels in either arm. Runtime1.36 seconds.
Coverage is200 traces per arm (20 per50-problem shard), NOT all500 first samples
or all2000 samples per arm used for the aggregate. The release README's broad
trace description must not make us claim evidence that is not present.

Together with exact score aggregation, this validates a substantial subset of
recorded-answer evidence. It does not turn the published7.9%/48.1% aggregate into
a new inference measurement. The new full-cap CPU pair remains a separate test.

A report builder now requires completed per-arm summaries, checks identical
inputs/settings and result hashes, and rechecks the source/weights before writing
the final reproduction report. It will not label active runs complete.

### Cycle 9: edited full-cap generation finished, original still pending

The edited service exited0. New CPU sample `gsm8k_train_2805` finished on EOS
after18023 generated tokens in2311.01 seconds, thinking closed, original hashes
verified afterward. It was NOT correct: published boxed-answer extractor returned
None, gold23. Inspection of the actual response end shows an unboxed final35;
the model repeatedly considered the correct subset total23 but chose the total35.
Thus this sample has both answer-format and semantic interpretation failure.
Do not claim "loop fixed, therefore math improved" from completion alone.

The original full-cap service had not yet emitted its completion event at this
update. Its result must be read before constructing the paired conclusion.
An event-only journal subscription is active in terminal
`47e3f30d-e002-47ea-a76d-ccfca56fe94c`, waiting for both services' completion;
the edited completion was observed. No polling/sleep loop, no run was shortened
or restarted. After original completion: run report_reproduction.py on K3,
preserve the final evidence, and update this context with the observed pair.

The completed edited-arm raw data has been retrieved locally at
`reproduction/qwen-edited-full/`. Other completed local evidence is at
`reproduction/breakout-reconstruction`, `breakout-1000`, `published-audit`.
K3 remains the canonical complete result tree under `reproduction/results/`.