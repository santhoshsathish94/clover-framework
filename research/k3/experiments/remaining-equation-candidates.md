# Remaining equation candidates

User direction (2026-09-29): list the remaining equation work after the thirteen
integrated items and their speed/RAM measurement. This inventory interprets the
request as fourteen further candidate areas, numbered14 through27. It is a
research queue, not authorization to implement all items or a promise of savings.

## Evidence and boundaries

Read the [model equation](../model/k3-model-equation.md) and the actual
[final integrated source](reverse-integration/step1/candidate.c). The objects
below exist in that source and have not received these proposed consumer
representations in the thirteen-item integration. Some proposals extend an
existing equation/codec to a different object; they are not fourteen invented
mathematical identities.

**Every item below is untested as a new representation.** Exact reconstruction
is the acceptance condition, not an observed result. A reversible description
need not be smaller, and avoiding a buffer can cost recomputation. Preserve
learned values, float32 rounding nodes, reduction order, full outputs and routes.
Do not restart the failed normalization/Gram shortcuts under another name.

The current source already has one-pass ordered top16 selection, expert reuse
across positions, staged weight I/O, and finite MXFP4 lookup tables. Those are
baseline capabilities, not new candidates. Knowing routed IDs does not make
expert weights resident.

## Parameters and stored weights

| Item | Target and possible representation | Actual code anchor | First discriminating check |
|---|---|---|---|
| 14 | **Expert-code residuals between groups/rows.** Predict a32-code group from a previous group and encode exact nibble XOR residuals, potentially by exceptional positions/values. | `Xm`, `coverage_replace`, packed `p1/p3/p2` records | Read one contiguous8-row by4-group tile of a known expert matrix. Compare512 original code bytes with predictor, residuals, tags and metadata; require exact reconstruction. Correlation is not established by the existing noncontiguous sample. |
| 15 | **Scale structure across rows.** Represent repeated base/width/usual-offset headers or scale-row templates jointly, instead of restarting each row descriptor independently. | `scale_prepare`, `exception_prepare`, row scale pointers | Use saved complete-row traces to count repeated headers/templates and reconstruct every scale; include template IDs, exceptions and boundaries. Adjacent-row prediction would require a fresh contiguous sample. |
| 16 | **Int8 weight-code tiles.** Apply exact bitplane, histogram or predictive coding to signed8-bit projection weights, beyond the MXFP4 expert matrices. | `Q`, `Qm`, I8R rows | Decode one small weight tile and preserve signed bytes, then compare one affected projection using the original16 FMA lanes and fixed tree. Count complete descriptor cost. |
| 17 | **Int8 projection row scales.** Encode the sequence of float32 scale words by exact dyadic fields or bitwise prediction. These are not E8M0 scale bytes. | `Q`/`Qm`: four-byte scale at each row start | Reconstruct a bundle of scale words bit-for-bit and keep multiplication after the dot product. Measure headers and payload before claiming a reduction. |
| 18 | **Learned BF16 embedding/head tables.** Encode blocks of the actual16-bit learned values, not merely references to the existing rows. | embedding `tab`, `Bf`, `LMW` | Reconstruct a bounded BF16 tile exactly; then check an embedding consumer and/or affected output-head row. Row references from items5/6 did not remove these tables. |
| 19 | **Normalization parameter vectors.** Extend exact scalar/block encodings to learned BF16/F32 norm weights and reconstruct them at their existing consumers. | `slot_vec`, `rmsnorm`, `rmsnorm_blocks`, `win/wpost/won/lnw/mn` | Check one parameter vector's stored and widened bits and its complete norm output. Do not absorb weights into the inverse or change the multiply order. |

## Temporary vectors

| Item | Target and possible representation | Actual code anchor | First discriminating check |
|---|---|---|---|
| 20 | **MLA query expansion.** Retain the1536-value normalized query latent plus learned projection reference; generate the96x192 expanded query coordinates at consumption. | `qnm`, `Qm(...WQB...)`, `qs`, attention `qh` | Regenerate one192-coordinate head exactly and compare its scores. Preserve the int8 projection tree and the full192-coordinate score order. This is an alternative storage/consumer form of existing query algebra. |
| 21 | **Routed expert input latent.** Represent the3584-value expert input as the exact ordered `WDN` projection of the7168-value `x2` source. | `Qm(zlm, ..., WDN)`, `zin` | Generate the full latent for one position, then feed one expert's gate/up projections. Compare all latent and projection bits; account for retained source and repeat-generation cost. |
| 22 | **SiTU intermediate activations.** Carry gate/up source expressions or exact descriptors and generate each nonlinear output coordinate when the down projection needs it. | `situ`, expert `go/uo`, shared `sgm/sum2`, dense gate/up vectors | Begin with one expert's3072 activation values and one down-projection row. Preserve the uncapped gate input to sigmoid and every tanh/multiply rounding; do not approximate saturation. |
| 23 | **KDA convolution history and outputs.** Describe each channel's current input and three preceding raw projected values, then regenerate its ordered convolution/SiLU output. | `convbuf`, `rawm`, `cv` | Compare one channel over the first five positions, including initial zeros and history shifts, then normalized q/k. The stored history is raw input, not post-SiLU output. |
| 24 | **KDA decay and gate vectors.** Carry generating inputs/parameters for alpha, beta and output gates rather than separately materializing every nonlinear result vector. | `betam/beta`, `zzm/alpha`, `gtm/gtf`, `ah/dtb` | Reproduce one head's alpha/beta values and gated output exactly. Retain per-head versus per-channel indexing and exp/sigmoid evaluation order; do not collapse consecutive rounded projections. |

## State and attention mixtures

| Item | Target and possible representation | Actual code anchor | First discriminating check |
|---|---|---|---|
| 25 | **KDA recurrent state as ordered updates.** Explore a checkpoint plus a generating history of diagonal decay and rank-one updates, with the original rounding at each state operation. | `St[H][D][D]`, delta-rule loops | Compare all128x128 values of one head after every position, then its output. Rounded updates need not preserve low rank; history can become larger than the dense state. Do not assert an exact low-rank closed form. |
| 26 | **MLA expanded KV cache from retained latents.** Retain the512 normalized latent values plus64 shared rope-slot values, and regenerate the expanded key/value projections when consumed. | `ccm`, `Qm(...WKB...)`, `mla_klat/mla_v/mla_rp` | Regenerate one position's expanded KV and compare attention, then a saved-prefix continuation in a new experimental cache format. The current code stores expanded keys/values; the learned WKB and its rounded evaluator still remain. |
| 27 | **MLA attention-output mixture.** Represent each128-wide head output as an ordered source expression over value-cache rows with the original rounded softmax coefficients. | `scv/ex/z/pr`, `accm`, `gbf`, `SA` | Check one head across causal positions, including scores, coefficients, weighted sums and gated output. Keep double score/normalizer accumulation and position-major float32 value accumulation. This transfers the source-expression approach; it does not remove cached values by itself. |

## Recommended next item

**Item14, expert-code residuals**, is the next small representation experiment.
In the already measured unique sample, code payload is3287040bytes while the
adaptive scale payload is23830bytes. That identifies a larger remaining data
target; it does not establish compressibility or predict runtime benefit.

For a declared deterministic predictor, the relation is exact:

$$c_{r,g,i} = \widehat c_{r,g,i} \mathbin{\oplus} d_{r,g,i}.$$

The question is whether the residuals have enough structure to pay for the
predictor/reference, positions, values, tags and row/group boundaries. Compare
against original packed codes on the same tile, with unchanged scale values.
Preserve the original code/scale bits before attempting a consumer replacement.
No tile was read or candidate tested as part of this list-only cycle.

## Separate follow-through

Not counted as new equations: extend successful sampled consumers to all
required sites; create an actual persistent encoded store; move reference
comparisons into independent test execution without weakening them; test longer
contexts and generation; remeasure the resulting implementation. These are
implementation and verification tasks with their own approval/coverage needs,
not mathematical reductions. The current measured harness retains original
data and duplicate computations, so its speed/RAM result does not settle these
remaining candidates.