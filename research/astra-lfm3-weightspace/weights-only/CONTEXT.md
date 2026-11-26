# LFM3 weights-only understanding, step by step

## Current direction

2026-11-11: the human redirected the work to understanding LFM3 open weights
only, using what was learned from the paper, under Clover, one step at a time.
This step studies the pinned original LFM3-Omni-2B checkpoint on K3. Stay in
branch `research/astra-lfm3-weightspace-20261111` and its existing worktree.

Only tensor header information (names, shapes, dtypes) and stored numeric values
may support claims here. No model forward pass, generation, prompts, activations,
benchmark outcomes, tokenizer labels, or weight changes. The assistant has prior
exposure to earlier experiments, so this is not claimed to be a blind study;
those outcomes are explicitly excluded from this step's evidence and calculations.

The remaining owned Qwen original inference service was stopped in response to
the redirection, preserving logs. Its administrative deactivation is not a
completed paired inference result. Do not resume the old task from its terminal
notification. No unrelated service or model file was changed.

## Step 1: the stored embedding table

Source tensor: `model.language_model.embed_tokens.weight` in the original
checkpoint at `/opt/clover-k3/astra-weightspace-20261111/original/model.safetensors`.
Pinned file SHA-256:
`76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`.
The name suggests an embedding role; runtime weight tying and token identities
will not be asserted from this name alone.

Question: does the table contain shared geometric structure and repeated rows,
or is its apparent structure explained by the distribution of individual values?

Falsifiable check: compute exact row/column norms, the shared mean-vector energy,
and exact repeated-row groups. On a fixed sample of 4096 rows, compare centered
top singular values with a control that independently permutes coordinates within
each row. The control preserves each sampled row's values/norm but removes their
coordinate alignment. It is a temporary derived array for statistics, not a
checkpoint edit or neural-network execution. Approximate spectra and sample
limits must be explicit.

Use hand-derived duplicate/rank-one controls to establish that the instrument can
distinguish structure. Read the original file read-only, verify its hash before
and after, and make known prior reports, reproduction data, fixtures and tokenizer
files inaccessible to the analysis process. No network access during the run.

## Meaning and limits

Identical rows mean identical stored vectors. Aligned variation can show shared
directions. Neither fact alone names a concept, identifies a behavioral weakness,
establishes a defect, or justifies a weight edit. Preserve observations separately
from interpretations, and stop after this first component before following a
specific structure into adjacent weights.

## Outcome

Step1 ran successfully on K3: 3.72 seconds of measurement, 8.74 seconds for the
sandboxed command including integrity checks. Three instrument controls passed
(exact duplicate rows, shuffle preserving row values, rank-one discrimination).
The sandbox refused access to the specified old reports/reproduction/fixtures,
tokenizer and config; the original checkpoint was read-only. Its complete hash
matched before and after. No model was instantiated or run; no tensor edited.

Observed from `step1/result.json` and `step1/arrays.npz`:

- Shape128000x2048, BF16,262144000 values. All finite and nonzero.
- All128000 rows are bitwise distinct; no all-zero or exactly duplicated rows.
- Row-norm min0.17231, median0.85471, max1.52750. Coordinate values range
	-0.27344 to0.21484, mean-0.0000820, std0.018206.
- An exploratory low-norm cutoff0.25 identifies8440 rows (6.59%);3188 occur
	among the last4096 row indices. This threshold was chosen after inspecting
	the norm summary and is descriptive, not a defect criterion.
- Sampled rows125823 and127426 have cosine0.9994224 and norms0.172872/0.172970.
	They are nearly parallel with similar magnitude, but not identical. Other
	sampled close pairs also exist. This is not a complete clustering of all rows.
- The shared mean vector accounts for1.8768% of total squared weight norm.
- On4096 fixed sampled rows, centered leading-direction energy is1.0895%,
	compared with0.13865% for an independently row-shuffled control preserving
	every row's values. Ratio approximately7.86. This shows coordinate-aligned
	structure beyond row value distributions; it does not isolate covariance
	from coordinate-scale differences or assign any semantic meaning.
- Approximate leading16 centered directions capture4.3273% of sample energy
	(control2.0749%). No obviously small dominant subspace is revealed by this
	sample. The leading direction itself is distributed: its largest coordinate
	carries0.8426% of its squared norm, not most of it.

## Current understanding

The stored table is dense and heterogeneous, with shared coordinate structure
that the row-shuffled control does not reproduce. It is not a collection of
identical placeholders. Some distinct entries have extremely similar stored
representations, alongside a substantial small-norm subset concentrated toward
the end of the row range. Those are mathematical properties of these weights.

Working hypothesis, not established: part of the table may be weakly
differentiated. We have not shown that all8440 low-norm rows share one direction,
nor that they are unused, rare, newly initialized, redundant in the complete
model, or related to any capability. Small norms do not prove small downstream
influence, especially without studying adjacent transformations. We do not know
which tokens or concepts these row indices denote and have not consulted their
labels. No weakness, behavioral cause or justified intervention is identified.

The paper-derived lesson used here is to read a concrete structure and test a
mathematical interpretation, not to import Qwen's repetition direction into LFM3
or reinterpret old benchmark failures as evidence from weights.

## Stop point and next question

This turn stops after the embedding component, as requested. The single next
weights-only question is whether the low-norm near-parallel examples belong to
one shared row family or several unrelated groups. Establish that from their
vectors before assigning meaning or following a selected direction into the
first-layer weight projections. No inference or intervention is authorized by
the present findings.

## Step 2: one low-norm family or several?

The human approved this next weights-only step. Keep the same evidence boundary
and unchanged original checkpoint. Question: do all8440 rows below norm0.25 form
one near-parallel family, several compact groups, or a connected but diffuse cloud?

Before observing results: normalize rows for directional comparison; compute all
within-population cosine similarities; examine connected components at fixed
thresholds0.90,0.95,0.99. Connectivity alone is not compactness: report minimum
within-group pair similarity and similarity to each group's mean direction.
Use a same-sized seeded sample of rows outside the low-norm subset as a geometric
reference. Group labels carry no token meaning. The low-norm cutoff remains
exploratory; no threshold will be selected to imply a defect.

Discriminating controls: recover two known distinct directional families;
detect a bridge-chain whose endpoints are not close; distinguish shared direction
from shared magnitude. Store every row ID and component assignment, verify saved
norms against freshly read weights, and check the original hash before/after.
No model loading, forward pass, tokenizer, activation or benchmark access.

### Step2 outcome

Step2 completed on K3 in4.32 seconds of measurement (sandboxed command9.33s).
Three synthetic controls passed, including the chain-versus-compactness case.
All8440 low-norm rows were compared exhaustively, with a seeded8440-row sample
from outside the subset as reference. Saved step1 norms matched newly read
weights. Original checkpoint SHA-256 matched before and after; excluded evidence
remained inaccessible, no network or model forward pass, and no weight edits.

Observed directional structure:

- Across ALL low-norm rows, mean non-self pair cosine0.855286; reference0.020105.
- Against the low-norm population's mean direction: minimum cosine0.612569,
	median0.985599. Thus a substantial common direction exists, but not every row
	is equally close to it.
- At cosine threshold0.90: one connected component of6004 rows plus2436
	singletons;47.9225% of all within-population pairs meet the threshold.
- At0.95: one component of5231 rows plus3209 singletons;35.7305% qualifying pairs.
- At0.99: one component of4173 rows plus4267 singletons;20.9035% qualifying pairs.
- There were no other multirow components at those thresholds. This does NOT
	rule out finer groups inside the main component or structure below0.90.
- The reference sample had no pair reaching0.90,0.95 or0.99. This statement is
	about the fixed sample, not every other row in the checkpoint.

Connectivity was not mistaken for a clique: the6004-row component's minimum
pair cosine is0.812571; the5231-row component's is0.899643. Even the4173-row core
contains some pairs below its0.99 edge threshold, but its minimum is0.970334,
its minimum cosine to its own mean direction is0.982420, and median is0.997021.
That core's norms span0.172310..0.176816 (median0.174819);3108 of its rows lie in
the final4096 table positions. These are numeric row positions, not decoded
token identities.

### Updated understanding

The earlier near-parallel pair was not an isolated coincidence. There is one
large, compact common-direction core, plus less-aligned rows extending away from
it. The observations support a common-template-plus-deviations interpretation
of this portion of the table. They do not support several separate tight families
at the tested resolutions, nor the stronger statement that ALL8440 rows are
near duplicates. Each row remains bitwise distinct as established in step1.

We still do not know whether this structure arises from initialization, training,
rare or unused entries, regularization, or something else. We did not decode
labels or infer semantic equivalence. Similar starting vectors can be separated
by later transformations, so this is not a functional redundancy or defect
finding and does not justify merging, pruning or changing weights.

Evidence: `step2/result.json` and `step2/arrays.npz`, including all row IDs,
norms, component assignments at each threshold and group centroid directions.
The cutoff0.25 remains exploratory from step1; similarity thresholds were fixed
before step2. Pair counts are exhaustive within each analyzed population;
reported pair quantiles use65536 seeded non-self draws. Float32 cosine values
can carry small numerical roundoff near1.

### Step2 stop point

Stop here for this approved step. Next weights-only question: how does the first
layer's projection-weight geometry align with the common embedding direction
and the deviations around it? That is a question for static weight analysis,
not permission to run inference or conclude that small input differences remain
small in the whole model.

## Step 3: first input-projection weight geometry

The human approved this next step. Read only the first-layer tensor headers:
`conv.in_proj.weight` is6144x2048 and `operator_norm.weight` is a matching2048
vector. This analysis uses those stored values and the established4173-row core.
It does not inspect runtime/model code or evaluate a neural-network layer.

Before results: decompose the core's embedding matrix into its common direction
and orthogonal row deviations. Compute the input projection's Gram matrix and
measure its squared gain along the common direction, average gain for isotropic
directions, and energy-weighted gain for the actual weight-derived deviations.
Test both bare P and algebraic composition P diag(s), where s is the named scale
vector. This is not a claim that it equals the complete layer's computation:
normalization denominators, gates, convolution and residual paths are excluded.

Inspect the three equal2048-row blocks without assigning them gate meanings from
their order. Report projected cross terms, because orthogonal input components
need not remain orthogonal under a learned matrix. Keep all original files and
earlier evidence immutable and the same forbidden-evidence mounts.

Discriminating controls: identity preserves common/deviation gain; a known null
direction is detected; an anisotropic diagonal matrix amplifies deviations by
the hand-derived ratio and responds correctly to column scaling. The question
is whether first-projection weights suppress, preserve or favor the common
family direction relative to its small differences. No behavior is inferred.

### Step3 outcome

Step3 completed on K3 in3.41 seconds including its final source hash
(sandboxed command6.41s). Original checkpoint hash matched before/after; prior
reports, reproduction data, fixtures, tokenizer/config and network were blocked.
No model was instantiated or executed and no stored tensor changed. Analytic
identity/nullspace/anisotropic controls passed; a cross-term control also checks
that energy accounting does not assume projected components stay orthogonal.

For the4173-row core, orthogonal deviations from the measured common direction
account for0.696633% of the input squared-weight energy. Measured static gains:

| Operator | Common-direction gain | Isotropic RMS gain | Deviation RMS gain | Deviation/common gain |
|---|---:|---:|---:|---:|
| Bare input projection P | 5.882263 | 1.662637 | 1.696163 | 0.288352 |
| Algebraic P diag(s) | 1.907274 | 0.331826 | 0.341236 | 0.178913 |

Thus the common direction is favored by3.5379x relative to isotropic RMS gain
for P, and5.7478x for P diag(s). The latter is a static weight composition, NOT
evaluation of the full normalization operation or a model activation.

Relative to the separate common/deviation component energies, deviations shrink
from0.696633% at input to0.0582953% after P and0.0224505% after P diag(s).
These fractions use the sum of the component energies; cross terms are reported
separately rather than hidden. Full projected energy has cross term-0.076762
out of4379.160494 for P, and+0.002478 out of460.238076 for P diag(s).

All three equal output-row blocks favor the common direction. Their bare
common/isotropic ratios are3.9117,2.5367,3.9619; deviation/common ratios are
0.25768,0.41265,0.25386. Under column scaling, common/isotropic ratios are
6.0837,4.8431,6.1785. No gate meanings were assigned to block order.

### Updated understanding after step3

The first input projection does not behave like a simple linear removal of the
common embedding-family component. It gives that component substantially larger
gain than either average directions or this family's actual differences. The
bare matrix still amplifies deviations in absolute norm (1.696x): the observed
attenuation is RELATIVE to the common part, not deletion of all differences.

This strengthens the common-template interpretation at this particular linear
operator, but does not explain its purpose. We cannot call the entries unused,
equivalent, defective or behaviorally indistinguishable. A complete layer also
contains normalization, nonlinear interactions, temporal convolution, output
projection and residual paths, which were not run or silently approximated.
We have not established their ordering or gate roles from tensor values alone.

Evidence: `step3/result.json` and `step3/arrays.npz`, with source-tensor hashes,
the exact row family/direction, full and per-block metrics, and explicit limits.

### Step3 stop point

Stop at this approved component. Next weights-only question: what do the stored
seven-tap convolution coefficients imply about constant versus varying signals?
Examining their static coefficient sums/frequency geometry could reveal whether
a persistent component would be passed or suppressed, without assuming that the
common embedding direction is actually constant in any real token sequence.

## Step 4: seven-tap coefficient response

The human approved this next step. Inspect only
`model.language_model.layers.0.conv.conv.weight`, whose stored shape is2048x1x7.
Treat each seven-value row as an analytical filter coefficient vector, without
instantiating a layer or feeding it any sequence. Do not infer channel-routing,
gate ordering, tap time direction, or actual token behavior from the tensor name.

Before observing the result: compute constant gain=sum(taps), alternating gain=
sum(taps*alternating signs), and frequency magnitude from the coefficient rFFT
with FFT length2048 (1025 nonnegative-frequency bins). Compare constant gain with coefficient L1 (cancellation)
and L2 (frequency RMS by Parseval). Predeclare strong DC cancellation as
abs(sum)/L1<=0.10; this is descriptive, not a defect criterion. Separate zero and
flat-magnitude filters before categorizing frequency preference. Record raw
coefficients, per-filter metrics, examples and hashes.

Discriminating controls: averaging passes a constant; a first difference rejects
it; a single-tap filter has flat magnitude and no frequency preference; a zero
filter is handled explicitly; Parseval and tap-reversal magnitude identities
hold. Frequency response is computed from weights, not obtained by running a
model on generated test sequences. Same read-only and forbidden-evidence sandbox.

### Step4 outcome

Step4 completed on K3:0.422s coefficient analysis,7.11s sandboxed command with
hash checks. Five analytical controls passed (average, difference, flat, zero,
Parseval/reversal). Original checkpoint SHA-256 matched before/after; forbidden
evidence remained blocked, no network, no sequence/model execution, no edits.

Measured2048 filters x7 coefficients (14336 BF16 values), tensor SHA-256
`467b2148e7ebe82d480605f1f46add83c07c352a074d29e2f676aa4d68fe6bfd`:

- No exactly zero filters and no exactly flat magnitude responses at the fixed
	tolerance. One extremely small filter, index1025, has L2 norm1.15694e-35;
	it is the only row below an exploratory1e-6 cutoff. This was checked after
	the scale summary and is not a defect criterion or an inferred runtime state.
-1763/2048 filters (86.08%) peak below0.125 cycles per discrete step;93 peak in
	[0.125,0.25),41 in[0.25,0.375),151 in[0.375,0.5]. Peaks are grid measurements.
-1497 filters have constant/DC magnitude at least99% of their grid peak.
-Only8/2048 meet the predefined strong DC-cancellation criterion
	abs(sum taps)/sum(abs taps)<=0.10. Median ratio0.95701: little sign cancellation
	for a typical row.
-236 have DC magnitude below their frequency-RMS gain;211 have alternating-sign
	gain exceeding constant gain. Those minorities prevent an all-filters claim.
-Median DC/frequency-RMS ratio1.12734; median alternating/frequency-RMS0.88935.
	The typical low-frequency preference is modest, not an ideal sharp low-pass.
-Stored tap index5 carries91.2973% of the whole bank's squared coefficient norm;
	it is the largest-magnitude tap in1850/2048 filters (90.33%). Mean absolute
	share for this tap is72.8123%. The bank is not generally uniform averaging
	over seven positions. No temporal direction/lag is assigned to this index.
-Every absolute DC gain is below1, median0.12790 and maximum0.395996. Favoring
	constant components RELATIVE to other frequencies is not absolute amplification.
-1001 signed DC gains are positive and1047 negative; no exact zero sums.
	Sign reversal alone is not rejection of a constant component.
-Parseval maximum absolute discrepancy6.94e-18 and reversal-magnitude
	discrepancy1.67e-16. Static transfer-function arithmetic passes these checks.

Representative coefficient-only contrasts: filter953 has DC-0.002733 versus
alternating gain0.070105 (contrast-sensitive); filter1135 has DC-0.184509 versus
alternating-0.002258 (constant-favoring). They are examples, not edit candidates.
All coefficients, per-filter spectra and examples are retained in
`step4/result.json` and `step4/arrays.npz`.

### Updated understanding after step4

These weights predominantly encode single-stored-position-dominated linear
mixing with smaller neighboring coefficients and a modest low-frequency bias.
They do not describe a universal mechanism that subtracts away every persistent
component. A small contrast-sensitive minority exists. This fits a more varied
filter bank than either "all averaging" or "all differencing".

We cannot infer that the common embedding-family component is actually constant
in any sequence, which projection block enters these filters, how gating changes
the signal, or which lag index5 represents. The preceding projection findings
and these isolated filter responses are NOT a complete composed model result.
The tiny filter is not sufficient evidence of a dead channel or a justified
pruning action. No capability, failure cause or improvement is established.

### Step4 stop point

Stop after this approved component. Next weights-only question: what structure
does the stored convolution output-projection matrix have, and how does its
column geometry reweight or mix the corresponding filter-index directions?
Keep any channel-alignment assumptions explicit; do not fill missing nonlinear
routing information with a plausible story or execute inference.