# Vectors as equations

## Current inventory

Inventory requested on 2026-09-29, after the adaptive scale-exception model
integration. These are the items in this representation study, not a claim
that each is independent, novel or deployed. Later forms sometimes replace
earlier intermediate formats.

**Status key:** "model pass" means an actual consumer replacement passed
the recorded full norm/logit comparison within its stated sample. "Data check"
means reconstruction of recorded data, not a model-integrated implementation.
The installed reference, learned weights and correctness gates remain unchanged.

### Vector items

| Item | Representation or purpose | Evidence status |
|---|---|---|
| Exact component equation | Sign, power of two and odd-integer code: x = sign * 2^exponent * (2*code+1) | Initial data check on 465 vectors; subsequently consumed in the scoped cumulative model pass (section 18) |
| Shared-exponent integer blocks | Exact integers and a shared exponent for blocks of vector components | Initial data check on 470 vectors; both32/128 formats subsequently consumed in the scoped cumulative pass |
| Lossless vector byte layouts | Raw zlib and byte-plane grouping plus zlib | Initial data check on 470 vectors; both layouts subsequently consumed in the scoped cumulative pass |
| Rounded source-vector expression | Generate a mixed vector from retained snapshots/residual and coefficients, preserving rounding nodes | Model pass for layer-84 position-0 router consumer; source vectors remain necessary |
| Entry row-reference consumer | Token ID plus learned BF16 embedding-row reference generates first RMSNorm inputs | Model pass on five-token entry, independently and with tail |
| Tail source-expression consumer | Generate rounded aggregate/normalized coordinates inside the BF16 output head | Model pass independently and with entry; head weights remain necessary |

The source-expression and endpoint equations, worked arrays, comparisons and
limits are in sections 1-9. Earlier byte-layout measurements are retained in
the [vector representation study](vector-representation.md).

### Expert code and scale items

| Item | Representation or purpose | Evidence status |
|---|---|---|
| Four-mask weight group | Four 32-bit masks plus scale generate 32 MXFP4 weights through the exact value equation | Model pass for one expert group; remains 17 payload bytes |
| One mask as count and rank | Population plus combination rank reconstructs one 32-bit mask | Model pass for one mask/group; example has 30 meaningful bits in four bytes |
| Whole group as counts and arrangement | Sixteen counts, exact code-order rank and scale | Model pass, then broader selected-group coverage; example grows to 24 bytes |
| Constrained histogram rank | Stars-and-bars ranks sixteen counts summing to 32 in 40 bits | Integrated model pass with arrangement rank over the sampled expert groups |
| Joint histogram/arrangement rank | One 128-bit rank carries the complete ordered code sequence; scale remains separate | Integrated model pass; fixed 17-byte group, equal to original packed payload |
| Scale-row base and offsets | Shared base and exact offsets across a complete row of group scales | Model pass on 1926 sampled rows; payload saving, not measured store/RAM reduction |
| Adaptive scale exceptions | Usual offset, exception count/positions/values, or fixed offsets selected by a stored tag | Latest integrated model pass on those sampled rows; further scale-payload saving |

These items are described in sections 10-17. Exception positions reuse
combinatorial ranking; exception values use an ordered radix rank. They are
parts of the adaptive scale representation, not additional separately
model-tested techniques. The joint rank supersedes separate histogram/order
fields as the fixed-size code representation in the latest path.

### Findings not accepted as replacements

| Item | Outcome |
|---|---|
| Absorbing an aggregation denominator into RMSNorm | Algebraic identity with epsilon adjustment; tested numerical replacement failed the original output gate and this route was stopped |
| Direct Gram/transformed-source normalized projection | Exact real-arithmetic identity; long-double consumer changed router values and final outputs. Failed, not superseded into a claimed pass by the rounded-expression repair |
| Small reusable integer-factor dictionary for later vectors | Exact dictionary reconstruction studied, but most later components require distinct factors; no compact reusable whole-vector generator or model replacement established |
| Nonlinear expression propagation through gating/SiTU | Prospective algebra discussed; no new nonlinear consumer replacement tested |
| Streaming top-16 selection | Proposed only, paused after revisiting known route IDs versus expert-weight residency; not implemented |

Failure and scope records remain in the earlier sections and
[normalization experiment results](normalization-reduction-results.json).
Existing DQ lookup, prefix reuse, route caching and staged expert I/O are
baseline capabilities, not new inventions of this study.

### Combined tests and present boundary

- Entry and tail were tested separately and together, not together with every
  later expert-representation experiment.
- Expert-code integration broadened to 5778 unique sampled groups from 214
  layer/expert pairs, across gate/up/down at layers 1, 48 and 92.
- Full-row scale integration covers 1926 unique sampled rows, 205440 scales
  and 6574080 weights using those scales. Joint-code substitution within those
  rows remains limited to the sampled 5778 groups.
- Latest adaptive runs on two five-token prompts passed 4320 affected projection
  comparisons, full routing traces and every final norm/logit value.

Latest unique-sample scale payload progression is **205440 original bytes ->
39914 fixed-offset bytes -> 23830 adaptive bytes**. The 3287040 code bytes are
unchanged in this accounting. Headers/tags/padding are included; persistent
index/container overhead is not. Original data and diagnostics remain present
in the harness, so neither deployed checkpoint nor process RAM reduction has
been measured.

A cumulative build now combines the thirteen listed items at the sampled sites,
following thirteen separate reverse-order gates (section 18). No test covers
all model groups, the entire checkpoint, all prompts, longer contexts or
generation/decode. No next candidate is selected by this inventory.

## Direction

The user set the current objective on 2026-09-29: express vector data through
equations, substitute those into the model equations, and establish the
mathematical relationships before choosing an implementation. Existing C
kernels do not bound the representation. A slower intermediate implementation
is acceptable; immediate performance improvement is not an acceptance gate.

**Verification sequencing update, 2026-09-29:** after the broader expert-group
tests, the user directed that the remaining equations can be identified before
the final complete-run verification. Do not require another full-model campaign
merely to consider the next equation. Preserve each derivation's assumptions,
existing evidence and untested scope. Identical results remain the target;
the original correctness gates are unchanged, and an integrated replacement
must not be called verified before its complete-run comparison. Earlier
per-step sequencing statements below describe the completed experiments;
this update governs subsequent equation identification.

**Acceptance correction, 2026-09-29:** only performance is deferred, not
correctness. A replacement must produce the same results as the reference
before this work advances to another stage. Retain the existing full final-norm
and logits comparison; the same selected token alone is insufficient. Mathematical
identities support a candidate, but do not substitute for this result check.

**Current status: scoped router and endpoint replacements verified; Gram shortcut
failed.** Section 7 records the rounding-aware layer-84, position-0 router
replacement. Section 8 records separate entry-only, tail-only and combined
endpoint tests. Every accepted case passes its checked stages and the complete
final norm/logit comparison on the five-token reference prompt. The endpoint
build does not include the router replacement. These results do not validate
all vector consumers, other prompts or the Gram form. Capture-only checks alone
did not establish replacement correctness.

Keep the learned weights and all 93 layers. Study the 7168-component vector
before experts or trunk representation. This does not restart the stopped
normalization-denominator experiment.

Distinguish mathematical equivalence, information accounting and runtime.
A valid representation can be a useful intermediate step without immediately
reducing stored information or execution time. No final minimal representation
has been established.

## 1. Generate a component without storing its integer table

Every nonzero finite float32 value has the exact form

$$x_i = s_i 2^{e_i}(2c_i+1),$$

where $s_i$ is either -1 or 1, $e_i$ is an integer exponent, and
$0 \le c_i < 2^{23}$. Factor the largest power of two out of its integer
significand; the remaining positive integer is odd. Thus its table is itself
an equation, $T[c]=2c+1$, and need not be stored.

This identity also covers nonzero subnormal float32 values. Zero, signed-zero
bit patterns and non-finite values require separate cases. The studied vectors
contained no such components.

The [table checker](vector-table-equation.py) reconstructed all 465 captured
layer-input vectors byte-for-byte, and their squared norms agreed exactly
under rational arithmetic. See [results](vector-table-equation-results.json).
The five embeddings each use 128 distinct odd factors. Later vectors use a
median of 7165 distinct factors and need a median of 23 bits for the universal
odd-integer code. These observations do not rule out relationships between codes.

This is an equation for each component. A shared equation for the whole vector
is a separate question.

## 2. Use the vector's actual construction

At snapshot aggregation the real-arithmetic model already supplies a whole-vector
generator. Let the ordered source vectors be the columns of $B$:

$$B=[b_1\ \cdots\ b_k]\in\mathbb{R}^{7168\times k},\qquad
  a\in\mathbb{R}^{k},\qquad x=Ba.$$

Equivalently,

$$F(i;B,a)=\sum_{j=1}^{k}B_{ij}a_j.$$

The sources are the existing snapshots and the live residual. The coefficients
come from the model's aggregation scores, not a fitted approximation.
At the captured layer-84 pre-MLP site, $k=9$.

The representation is the pair $(B,a)$, not $a$ alone. If $B$ is already
retained, the aggregate can be described by nine coefficients and nine source
references instead of another explicit 7168-component array. The nine source
vectors still contain 64512 components. This is relative representation of
an intermediate value, not compression of arbitrary vectors into nine numbers.

$B$ is a source matrix; its columns need not be independent. Each reference
denotes the value at a particular position and stage, not a mutable buffer
whose later contents could silently change the equation.

## 3. Substitute into the consuming equations

For any compatible real matrix $W$, associativity gives

$$Wx=W(Ba)=(WB)a.$$

For the squared norm, define the Gram matrix $G=B^\top B$:

$$\|x\|^2=(Ba)^\top(Ba)=a^\top G a.$$

For RMSNorm, let $D_w$ denote the diagonal matrix of learned gains, and
retain the model's positive epsilon. Then

$$\operatorname{RMSNorm}(Ba;w,\epsilon)
  =\frac{D_wBa}{\sqrt{a^\top Ga/7168+\epsilon}}.$$

Combining normalization and projection gives

$$\boxed{
W\operatorname{RMSNorm}(Ba;w,\epsilon)
=\frac{Ha}{\sqrt{a^\top Ga/7168+\epsilon}},
\qquad H=WD_wB,\quad G=B^\top B.
}$$

Proof: expand $x=Ba$ in the norm definition, substitute
$(Ba)^\top(Ba)=a^\top B^\top Ba$, and move the scalar denominator outside
the linear projection. Epsilon is unchanged. Neither orthogonality of $B$
nor coefficients summing to one is required. Positive epsilon keeps the
denominator defined even if $Ba=0$.

This equation does not require explicitly constructing the mixed vector $x$.
It still requires $B$, or the information needed to obtain $H$ and $G$.
Their construction and reuse are later implementation questions, not reasons
to reject the identity. $H$ depends on $W$, $w$ and $B$; changing those does
not leave $H$ unchanged. The learned weights themselves are not replaced.

## 4. Carry the representation beyond a linear stage

This section records prospective algebra only. The limited result in section 7
does not establish any of these nonlinear replacements.

The following rules describe exact real-arithmetic composition:

$$x+y=[B\ C]\begin{bmatrix}a\\b\end{bmatrix}
\quad\text{when }x=Ba,\ y=Cb,$$

$$Wx=(WB)a,\qquad \lambda x=B(\lambda a).$$

A pointwise nonlinearity can remain an explicit generating expression:

$$z_i=\phi\!\left(\sum_j B_{ij}a_j\right).$$

That is valid, but it need not lie in the same fixed linear span. For example,
with $B=(1,2)^\top$ and $a=0$, elementwise sigmoid returns
$(1/2,1/2)^\top$, which is outside the span of $B$. This is a counterexample
to automatic fixed-span closure, not a claim that K3 encounters this example.

The next representation may therefore be a nonlinear generating expression
rather than another vector with the same nine coefficients. How to simplify
that expression is open. Merely nesting all original operations gives a valid
description, but does not by itself establish a reduced computation.

## 5. Exact arithmetic and the current numeric evaluator

Sections 2-4 preserve real-arithmetic functions with fixed decoded checkpoint
coefficients. The reference C also specifies rounding after particular products
and sums. An identity over the reals does not automatically preserve that
separate rounded function.

The AX102 checker keeps both questions visible:

- Repeating the reference float32 aggregation reconstructed its captured output
  byte-for-byte.
- Computing the mixture as an exact rational vector satisfied the Gram and
  linear-projection identities exactly.
- That exact mixture differed from the staged float32 sum in all 7168 components,
  with maximum absolute difference about 1.06218e-7 at this site.

The last observation is a numeric distinction, not a rejection of the real
identity. It is nevertheless relevant to the required equality of results:
the identity alone does not clear that check. An evaluator's precision and
rounding need explicit specification and verification against the reference.
Section 7 records the subsequent result of implementing a scoped replacement;
the algebra and capture checks alone do not establish that result.

## 6. Evidence and limits

AX102, 2026-09-29, five-token France prompt:
`1008,10484,318,15383,387`. A separate capture-only build recorded the first
nine-source AR call at layer 84: pre-MLP, position 0. Captured fields were the
nine coefficients, nine full source vectors, the learned scoring direction
and the reference aggregation output.

The [capture patch](vector-basis.patch) changes only an experimental copy;
the installed source, binary and gate remain untouched. Build flags were
`-O3 -march=native -ffp-contract=off -fopenmp -DNPOS=5 -lm`.
The capture build passed the preserved logits checksum
`23d162dcefb18211a7540ef12948f1eb`.

The [checker](vector-basis-equation.py) uses Python Fraction arithmetic on
the actual captured values. [Results](vector-basis-results.json) retain the
exact rational quantities and capture hash. It verifies the squared RMS
denominator, not a floating approximation to its square root. The projection
check uses the captured learned AR scoring direction as one real linear
functional; it is not a run of the next Q or router kernel. The general matrix
identity is established by the algebra in section 3.

Remote artifacts are under `/opt/clover-k3/vector-basis-20260929-a`:
`baseline.c`, `captured.c`, `captured`, `basis.bin`, `logits.bin`, run logs,
the patch, checker and `results.json`. The earlier table study is under
`/opt/clover-k3/vector-representation-20260929-a`, with the v2 checker and
`table-equation-v2-results.json`.

This establishes one exact representation and its linear/norm compositions.
It does not establish a final representation, a runtime improvement, a fixed
nine-dimensional space across layers, or a new nonlinear closure theorem.
Performance is deliberately deferred for this stage of the research. Correctness
is not deferred. The actual replacement experiment follows below; it must not
be confused with the capture-only checks above.

## 7. Actual replacement: failed shortcut, passing source expression

The first consumer chosen was the layer-84, position-0 router, immediately
after the captured nine-source pre-MLP aggregation and RMSNorm. All 896 rows
of its learned matrix are evaluated by the candidate. Candidate values feed
the original sigmoid, expert selection and weights, followed by the remaining
model. The original normalized vector is still computed for other consumers
and reference diagnostics; it is not the input to the candidate projection.

### Mode 1: direct Gram form

The candidate computes the Gram matrix, transformed source projections and
normalization in `long double`, then casts the projection to float32 before
the original sigmoid. This is a numerical realization of section 3, not an
exact rational calculation.

Result: **FAIL**, despite the same emitted token 17374.

- 224 of 896 projected router values differ; 224 sigmoid values differ.
- 6547 of 7168 final norm values differ.
- 143038 of 163840 logits differ, maximum absolute difference
  3.814697265625e-06.
- Output md5: `59afac21c3ae080d3707b126ab62d13c`, not the reference.

The [failed result](vector-basis-router-gram-results.json) remains preserved.
More accurate evaluation of a real-arithmetic expression is not automatically
evaluation of the reference's rounded expression.

### Mode 2: explicitly rounded generating expression

Keep the same sources and coefficients, but make the rounding nodes part of
the representation. Write $R_{32}$ for float32 rounding. Each coordinate is
generated on demand by

$$v_i^{(0)}=0,\qquad
v_i^{(j)}=R_{32}\!\left(v_i^{(j-1)}+R_{32}(a_jB_{ij})\right).$$

Using $v_i=v_i^{(k)}$, the norm uses the reference's ascending-coordinate
double sum of squares and one float32 reciprocal-square-root result $\eta$.
Each normalized coordinate is then generated by

$$n_i=R_{32}\!\left(R_{32}(w_iv_i)\eta\right).$$

The router accumulates $W_{oi}n_i$ in ascending-coordinate double arithmetic,
casts once to float32 and calls the existing sigmoid. These are the reference
arithmetic operations, now evaluated from the retained source expression.
The candidate does not read the existing `x2b` normalized-vector array or
substitute the separately calculated reference score.

This is **not** the Gram shortcut: the rounding nodes cannot simply be removed
when the target function includes them. The new expression is still a
representation of the vector through source references and coefficients.
No claim of fewer arithmetic operations is made.

Result: **PASS**, with zero changed bits in every checked output:

| Check | Identical values |
|---|---:|
| Router projection | 896 / 896 |
| Router sigmoid output | 896 / 896 |
| Final normalized vector | 7168 / 7168 |
| Logits | 163840 / 163840 |

Output md5 is `23d162dcefb18211a7540ef12948f1eb`, the original gate, and token
17374. Both revisions' disabled modes match the unchanged reference. The
reference run repeated after the first candidate also matches its initial run.
See the [passing result](vector-basis-router-rounded-results.json).

### Implementation and boundaries

The [patch](vector-basis-router.patch) connects the isolated
[helper](vector-basis-router.h) to the reference source. `K3_BASIS_ROUTER=0`
disables it, `1` selects the Gram candidate and `2` selects the rounded
expression. Enabled modes require `K3_BASIS_REPORT` for per-row comparison
data. All candidate rows feed the model; reference values are diagnostics.

AX102 artifacts are in `/opt/clover-k3/vector-basis-router-20260929-a` for
the failed revision, and its `rounded/` subdirectory for the passing revision.
Each retains source, helper, patch, binary, outputs, logs and results. The
result JSONs identify both the main source and included helper hashes; the
helper is part of the build. The installed source, binary and gate are unchanged.

This establishes the requested same-result outcome for one actual consumer,
one position, one layer and one prompt. It does not establish a whole-model
source-expression evaluator, other prompts, decode or memory savings. Source
references are valid during this consumer's execution; a wider implementation
must preserve their value lifetimes. Performance remains deferred. Any extension
of this representation needs its own same-result check before further progress.

## 8. Entry and tail endpoint tests

The user next authorized testing the same representation principle at the
token-to-vector entry and vector-to-token exit. Both were tested independently,
then together, in a fresh experimental copy of installed Clover. The router
modification from section 7 is not included in this build.

### Entry: row reference consumed directly by RMSNorm

Define a coordinate generator for token $t$:

$$e_i(t)=\operatorname{BF16toF32}(W_E[t,i]).$$

The candidate retains the token ID and embedding-table reference, computes the
ascending double sum of $e_i(t)^2$, then generates normalized components using
the reference's float32 reciprocal and multiply association:

$$\eta_t=R_{32}\!\left(1/\sqrt{\operatorname{sum}_{64,i} e_i(t)^2/7168+\epsilon}\right),
\qquad x_{1,i}=R_{32}\!\left(R_{32}(w_i e_i(t))\eta_t\right).$$

Layer-0 attention consumes this generated normalized vector. The candidate
RMSNorm calculation reads the BF16 row directly, not the materialized embedding
array. The original embedding array is retained for snapshot/residual consumers
and comparisons. The normalized output is still materialized for attention.
Thus this is a tested row-reference consumer, not removal of all entry arrays.

### Tail: source expression consumed directly by the output head

The original tail computes nine mixture coefficients from its eight snapshots
and last-position residual. The candidate retains those coefficients and source
references, generating $v_i$ and $n_i$ by the explicitly rounded equations in
section 7. The candidate head reads those generated $n_i$, not the existing
normalized-vector array.

For each vocabulary row it widens the BF16 weights exactly, accumulates products
in the same 16 double lanes, and uses the reference tree:

$$U_c=(C_c+C_{c+4})+(C_{c+8}+C_{c+12}),\quad c=0,1,2,3,$$

$$\ell_v=R_{32}\!\left((U_0+U_1)+(U_2+U_3)\right).$$

Each lane receives coordinates congruent to its index modulo 16, in ascending
order. Generated logits replace the original head results and feed the ordinary
argmax. Reference logits remain only long enough for an exact comparison; there
is no fallback to them. The head is still a projection, not a reverse embedding
lookup. Learned embedding and head weights remain unchanged.

### Same-result checks

AX102, 2026-09-29, fresh prefill of token IDs
`1008,10484,318,15383,387`. Each test passed before proceeding to the next.

| Mode | Endpoint checks, all byte-identical | Complete final norm/logits |
|---|---|---|
| Disabled control | No endpoint substitutions | 171008 / 171008 |
| Entry only | 35840 embedding + 35840 entry-normalized components | 171008 / 171008 |
| Tail only | 7168 aggregate + 7168 normalized + 163840 head logits | 171008 / 171008 |
| Entry and tail | All entry and tail checks above | 171008 / 171008 |

All four modes retain output md5 `23d162dcefb18211a7540ef12948f1eb` and token
17374. The final norm comprises 7168 values and the logit vector 163840 values;
acceptance is not based on the token alone. An unchanged reference run after
the combined test also matches the initial reference run. Installed source,
binary, gate and model index hashes remain unchanged.

The [helper](vector-endpoints.h), [patch](vector-endpoints.patch) and independent
[output checker](vector-endpoints-check.py) are preserved along with the
[full results](vector-endpoints-results.json). `K3_VECTOR_ENDPOINTS` selects
0 (disabled), 1 (entry), 2 (tail), or 3 (both). Every run requires a separate
`K3_ENDPOINT_REPORT` path for checked counts and bit mismatches. The checker
also verifies expected stage counts and compares complete output files.

Remote artifacts: `/opt/clover-k3/vector-endpoints-20260929-a`, containing the
baseline/candidate source and binaries, helper, patch, checker, all four mode
outputs and stage reports, reference-before/after outputs, logs, intermediate
check results and final `results.json`. Build flags and environment settings
are recorded in the results; the build uses `-ffp-contract=off`.

Scope: one five-token prompt, fresh prefill, not decode or general prompt
coverage. Reference computations are retained for diagnostic comparisons and
other consumers. This establishes that these endpoint consumers can use the
generating expressions with identical results, not a memory or performance
improvement, a complete removal of vector materialization, or a test of the
router and endpoints enabled together. Performance remains deferred.

## 9. A real 7168-component vector example

This example comes from the AX102 captures used above, for position 0,
token ID **1008**, text **`The`**, in the prompt `The capital of France is`.

The [complete example](vector-example.json) contains both full 7168-value
arrays, all nine coefficients, source labels and the first-coordinate
calculation. The [exporter](vector-example.py) checks the capture identity,
reconstructs every intermediate component, and verifies that exporting the
arrays to JSON preserves their float32 bit patterns.

### Original embedding

The initial vector is the token's learned BF16 embedding row widened exactly
to float32. These are its **first eight components only**:

```json
[
  0.032470703125,
  -0.0024566650390625,
  -0.0361328125,
  -0.00008106231689453125,
  0.009033203125,
  0.0233154296875,
  -0.01416015625,
  -0.06982421875
]
```

The other 7160 components are in `embedding.values` in the complete example.

### Intermediate vector

At the same position, layer 84's pre-MLP aggregation produces the following
**first eight components**, before RMSNorm:

```json
[
  0.1127423495054245,
  -0.0872378945350647,
  -0.09514196217060089,
  0.07357168197631836,
  0.1025756448507309,
  0.0418103002011776,
  -0.1276521384716034,
  0.17316222190856934
]
```

The full array is `intermediate.values`. These two arrays are different stages
of one token's computation, **not** the same vector before and after compression.

### The intermediate vector as a source expression

The ordered sources are the snapshots taken on entry to layers
0, 12, 24, 36, 48, 60, 72 and 84, followed by the live residual after
layer-84 attention. Their nine coefficients, in the same order, are:

```json
[
  0.004975033458322287,
  0.018128681927919388,
  0.000512637838255614,
  0.00029077011276967824,
  0.0018231947906315327,
  0.002290245844051242,
  0.03026803955435753,
  0.039532385766506195,
  0.9021790027618408
]
```

Rather than reading a stored intermediate component, generate it from the
corresponding component in each source:

```text
value = 0
for source = 0 through 8, in order:
    contribution = float32(coefficient[source] * sources[source][coordinate])
    value = float32(value + contribution)
```

For coordinate 0, the first source is the original embedding value
`0.032470703125`. Its rounded contribution is `0.00016154283366631716`.
After the first eight sources, the rounded running sum is
`0.07156509906053543`. The live residual's coordinate is
`0.04564199596643448`; multiplying by its coefficient and rounding gives
`0.04117725044488907`. The final rounded addition produces
**`0.1127423495054245`**, exactly the intermediate array's first component.

All **7168 of 7168** intermediate components reconstructed byte-for-byte, not
just this coordinate. The complete example retains each step of coordinate 0
under `source_expression.first_coordinate_steps`.

The nine coefficients describe a mixture of retained source vectors; they do
not replace the information in those vectors. Likewise, the embedding and
output-head weight matrices remain required. This example illustrates a
verified generating representation for an intermediate value, not elimination
of the learned tables. It uses existing captures; no additional model run was
needed to export it.

## 10. First selected-expert weight group

After clarifying that known expert IDs and resident expert weights are different
objects, the user authorized a bounded expert-weight representation study.
The first item is **one group of 32 weights** in a selected expert, not routing
selection or an entire expert. Only performance is deferred; all result checks
remain mandatory.

### What Clover already represents

The existing expert format stores 32 four-bit E2M1 codes in 16 bytes, with one
E8M0 scale byte for the group. The 17 bytes generate 32 decoded float32 weights.
The existing DQ table already tabulates all 4096 code/scale combinations. That
small-domain observation is prior work, not a new reduction here.

### From one expert down to 17 bytes

One routed expert contains three learned matrices; shapes below are output
rows by input columns. This does not include the separate shared expert or
the layer's trunk.

| Matrix | Shape | Weights | 32-weight groups | Packed codes + scales (bytes) |
|---|---|---:|---:|---:|
| w1, gate | 3072 x 3584 | 11010048 | 344064 | 5849088 |
| w3, up | 3072 x 3584 | 11010048 | 344064 | 5849088 |
| w2, down | 3584 x 3072 | 11010048 | 344064 | 5849088 |
| Entire routed expert | Three matrices | 33030144 | 1032192 | 17547264 |

Each matrix contains 5505024 packed-code bytes and 344064 scale bytes.
The expert's total payload is **17.547264 MB (16.734375 MiB)**, excluding
file/index metadata, I/O alignment and execution buffers.

For the gate matrix, the hierarchy is:

```text
One expert: three matrices, 17,547,264 bytes total
  Gate matrix: 3,072 rows, 5,849,088 bytes
    One row: 3,584 weights = 112 groups, 1,904 bytes
      One group: 32 weights = 16 code bytes + 1 scale byte
```

The down matrix instead has 3584 rows, each with 3072 weights: 96 groups
or 1632 bytes per row. It has the same total size as each other matrix.
These are payload accounting units: codes and scales can occupy separate
tensor ranges; the original format need not interleave 17-byte records.

If expanded into float32, one group would occupy 128 bytes and the entire
expert 132120576 bytes. Clover already avoids that persistent expansion with
MXFP4. The new masks do not introduce that existing saving: both the original
packed group and the alternative mask descriptor carry **17 bytes**.
Only one of the expert's 1032192 groups was substituted in this experiment.

### The alternative group equation

Transpose the four code bits into four unsigned 32-bit masks $M_0,\ldots,M_3$.
For coordinate $i\in\{0,\ldots,31\}$, define

$$b_{k,i}=\left\lfloor M_k/2^i\right\rfloor\bmod 2,\qquad
c_i=b_{0,i}+2b_{1,i}+4b_{2,i}+8b_{3,i}.$$

The mantissa bit is $m_i=b_{0,i}$, the two-bit exponent is
$e_i=b_{1,i}+2b_{2,i}$, and the sign bit is $b_{3,i}$. Generate the unscaled
magnitude without the E2M1 lookup table:

$$a_i=\begin{cases}
m_i/2,&e_i=0,\\
(1+m_i/2)2^{e_i-1},&e_i>0.
\end{cases}$$

For scale byte $s<255$ the weight is

$$w_i=R_{32}\!\left((-1)^{b_{3,i}}a_i2^{s-127}\right).$$

The implementation applies the sign with `copysignf`, including negative zero,
and the power with `scalbnf`. Scale byte 255 produces signed zero as in the
current Clover decoder. This is the reference's special-case behavior, not a
claim about every external interpretation of E8M0. Overflow/subnormal outcomes
and signed zeros are part of the bit comparison.

Thus the group is described by $(s,M_0,M_1,M_2,M_3)$ and a shared equation.
Its learned payload remains **17 bytes**, exactly the original size. The masks
encode the original codes; they are not constants shared by all experts. No
learned information has disappeared. A complete store also needs the group's
location, just as the original representation does.

### Apply the equation to the model's projection

At row 0 of the selected gate projection, substitute the generated $w_i$ for
coordinates 0 through 31 in the usual weighted sum. In real notation:

$$y_0=\sum_{i=0}^{31}F(i;s,M)x_i+
       \sum_{i=32}^{3583}w_i x_i.$$

The candidate does **not** evaluate these as two separate sums: it inserts the
generated weights into the original four-by-four double lanes and preserves
the original unfused multiply/add order and reduction tree. The remaining
weights in the row use the original DQ decoding. This avoids changing the
numerical function while changing the representation of one group.

The selected row is recomputed and replaces its reference output before the
original SiTU and subsequent expert/model operations. Reference computation
is retained for diagnostic comparison, not substituted for the candidate.

### A real group and its results

AX102 selected **layer 1, expert 498, w1/gate, row 0, group 0**, used at
position 0, rank 0 of the five-token France prompt. Its new descriptor is:

```json
{
  "scale_byte": 121,
  "masks": [2964563821, 2011846492, 3356231810, 3458434101]
}
```

The scale is $2^{121-127}=1/64$. These are the first eight decoded weights:

```json
[-0.0078125, 0.03125, -0.0234375, 0.0234375,
 -0.015625, -0.0078125, 0.0234375, 0.03125]
```

For coordinate 0, the masks produce code 9: sign bit 1, exponent 0, mantissa
1. Its value is $-\tfrac12\times\tfrac1{64}=-0.0078125$.
The [full result](expert-group-results.json) retains all 32 codes and decoded
weights, the original packed bytes, and the tested projection output.

| Check | Outcome |
|---|---|
| All code/scale combinations against current DQ | 4096 / 4096 bit-identical |
| Every code at every position in a mask group | 512 / 512 exact |
| Actual group codes reconstructed into packed bytes | Exact |
| Actual decoded weights | 32 / 32 bit-identical |
| Affected full-row projection output | 1 / 1 bit-identical |
| Final normalized vector | 7168 / 7168 bit-identical |
| Final vocabulary logits | 163840 / 163840 bit-identical |

Fresh reference, disabled candidate and enabled candidate all retain md5
`23d162dcefb18211a7540ef12948f1eb` and token 17374. A repeated unchanged
reference also matches. Installed source, binary, gate and index hashes are
unchanged. These results cover one actual group and one prompt; the exhaustive
decoder check covers the finite code/scale domain, not all model workflows.

### Artifacts and next boundary

The [helper](expert-group-equation.h) and [patch](expert-group-equation.patch)
implement the isolated replacement. `K3_EXPERT_GROUP=1` enables it;
`K3_EXPERT_GROUP_REPORT` names its stage report. The
[independent checker](expert-group-check.py) verifies mask/code reconstruction,
actual weights, projection values and complete output files.

The [serialized descriptor](expert-group-mask.bin) is four little-endian
uint32 masks followed by the uint8 scale, exactly 17 bytes. It was serialized
and checked after the model run; the model experiment constructed its masks
from the loaded packed group. Loading an alternative on-disk expert format
has not been tested.

AX102 artifacts remain under `/opt/clover-k3/expert-group-equation-20260929-a`:
both sources and binaries, helper/patch/checker, stage report, descriptor,
reference/disabled/enabled/repeated outputs, logs and final results. Build
flags and environment are recorded in the results.

This is the first result-preserving weight-group representation in this study.
It does not remove the original weight files, the other DQ consumers, or any
of the group's 136 payload bits. Neither an entire row nor all groups of an
expert were changed. Further relations among the masks or across groups are
open questions, not established reductions. Performance remains deferred.

## 11. Next candidate: one mask as a count and combination rank

The expert/matrix/row/group byte breakdown in section 10 remains the starting
record: 17547264 bytes per routed expert, 1032192 groups, 17 bytes per group.
The next candidate stays inside the already-tested group and asks whether
one of its four 32-bit masks has a shorter exact generating description.
This is a representation study, not permission to change the result.

### The equation

A 32-bit mask with exactly $k$ set bits is specified by its ordered set-bit
positions $0\le p_1<\cdots<p_k<32$. There are exactly $\binom{32}{k}$ such
masks. Its zero-based colexicographic rank is

$$R=\sum_{j=1}^{k}\binom{p_j}{j},\qquad 0\le R<\binom{32}{k}.$$

Here $\binom{p}{j}=0$ when $p<j$. The empty mask has $k=0,R=0$.
To regenerate a mask, work from $j=k$ down to 1: choose the largest permitted
position $p_j$ with $\binom{p_j}{j}\le R$, subtract that binomial coefficient,
and require the next position to be smaller. The decreasing positions give
the original mask exactly. This is a standard combinatorial number-system
representation, not a new model-specific identity.

Once $k$ is known, the rank needs

$$b_R=\left\lceil\log_2\binom{32}{k}\right\rceil$$

bits, with zero bits when there is only one possible mask. A general count
$k\in\{0,\ldots,32\}$ requires six bits. The count must not be silently
treated as free learned information.

### The real group

Applied to the same layer-1 expert-498 gate group from section 10:

| Plane | Original mask | Set bits $k$ | Combination rank $R$ | Rank bits | Count + rank bits |
|---|---:|---:|---:|---:|---:|
| 0 | 2964563821 | 18 | 289269275 | 29 | 35 |
| 1 | 2011846492 | 20 | 71379515 | 28 | 34 |
| 2 | 3356231810 | 8 | 10235410 | 24 | 30 |
| 3 | 3458434101 | 14 | 413656102 | 29 | 35 |

Plane 2 is the smallest candidate in this group. Its set positions are:

```json
[1, 7, 11, 18, 19, 27, 30, 31]
```

It can therefore be generated exactly from **count 8 and rank 10235410**.
Those fields need 30 bits rather than the raw mask's 32, before accounting
for any choice of representation. The other three masks do not individually
shrink under this count-plus-rank format.

### Worked reconstruction: count and rank back to the mask

This is a concrete example of the mask-decoding test that passed, not a
new model-run claim. Start with the same group's plane-2 mask:

```text
Original mask: 3356231810 (32 bits)
New description: count = 8, rank = 10235410
Regenerated set positions: [1, 7, 11, 18, 19, 27, 30, 31]
Regenerated mask: 3356231810
```

The rank follows directly from the eight positions:

$$R=\binom{1}{1}+\binom{7}{2}+\binom{11}{3}+\binom{18}{4}
  +\binom{19}{5}+\binom{27}{6}+\binom{30}{7}+\binom{31}{8}$$

$$R=1+21+165+3060+11628+296010+2035800+7888725=10235410.$$

Decode from the largest position down, subtracting each chosen binomial term:

| Step $j$ | Chosen position $p_j$ | Subtracted $\binom{p_j}{j}$ | Remaining rank |
|---|---:|---:|---:|
| 8 | 31 | 7888725 | 2346685 |
| 7 | 30 | 2035800 | 310885 |
| 6 | 27 | 296010 | 14875 |
| 5 | 19 | 11628 | 3247 |
| 4 | 18 | 3060 | 187 |
| 3 | 11 | 165 | 22 |
| 2 | 7 | 21 | 1 |
| 1 | 1 | 1 | 0 |

Set those eight bit positions to one and all other positions to zero:

$$M_2=2^1+2^7+2^{11}+2^{18}+2^{19}+2^{27}+2^{30}+2^{31}
   =3356231810.$$

All 32 bits are restored exactly. The count and rank describe the placement
of this mask's bits, not 32 expert weights by themselves. The other three
masks and the scale byte remain required to generate the group's weights.

**Initial verification:** the four actual mask roundtrips and the 5249 finite
decoder checks in the saved AX102 result. At that point, count/rank had not
been used as an actual expert-projection replacement. The original four-mask
decoder's model pass in section 10 did not establish this new decoder's model
correctness. The subsequent, separately verified consumer test is recorded
below under "Count/rank consumer model test".

### Account for the complete description

| Candidate format for the group | Total bits including scale | Byte-rounded size |
|---|---:|---:|
| Original four raw masks plus scale | 136 | 17 |
| Count/rank for all four masks plus scale | 142 | 18 |
| Per-plane raw/ranked choice, including four choice bits | 138 | 18 |

The last row uses the shorter payload per plane but still needs one tag bit
per plane to identify that choice. No smaller whole-group representation has
been demonstrated here. A count shared across groups would change the accounting
only if it is genuinely known or stored once and valid for all those groups;
that is not established by this one group.

Larger intermediate representations are allowed in the current mathematical
study. These numbers characterize the candidate; they do not veto the identity
or turn it into a demonstrated compression result.

### Verification state and next gate

Ran the [rank/unrank checker](expert-mask-rank.py) on AX102 against the existing
real group. All four masks reconstructed exactly. An additional 5249 finite
checks cover every 12-bit mask, 32-bit masks with zero/one/two set bits and
their complements, and boundary/midpoint ranks at every population from 0 to 32.
These checks support the implementation; the general rank identity is given
above. The [results](expert-mask-rank-results.json) retain the counts, ranks,
positions and complete format accounting.

The first checker attempt confused the flat report's numeric `group` index
with a nested group object. It was corrected before the successful run; no
model computation was involved in that parsing error.

**Initial status:** exact mask reconstruction checked, not yet a model
replacement. The outstanding gate was to replace plane 2 with its count/rank
decoder at the same expert group and compare exact codes, all 32 weights,
the affected projection and every final norm/logit value. The test below
subsequently passed that gate. It is separate evidence from the rank/unrank
checks, and does not validate other groups or stages.

AX102 checker and result remain in
`/opt/clover-k3/expert-group-equation-20260929-a` as `expert-mask-rank.py`
and `mask-rank-results.json`. This is one sample, not a conclusion about mask
structure across the expert, across layers or across the full checkpoint.

### Count/rank consumer model test

On 2026-09-29 the user authorized the outstanding model test. The
[C decoder](expert-mask-consumer.h) was inserted into a new experimental copy
of the previously tested expert-group helper using this
[helper patch](expert-mask-consumer.patch). The existing group-consumer patch
and all original result checks were retained.

At **layer 1, expert 498, gate/w1 row 0, group 0, position 0, rank 0**, the
candidate performs the following:

1. Derive plane 2's count and combination rank from the loaded group.
2. Pack the fields into bytes and read the fields back from those bytes.
3. Clear plane 2 and regenerate it using only the decoded count and rank.
4. Use the regenerated mask with the other three masks and scale to generate
  the group's 32 weights in the affected projection.
5. Feed the candidate projection result into the original SiTU and remaining
  model. The old value is used for comparison, never as a fallback.

The actual encoding is:

```text
Original plane 2: 3356231810
Count:           8
Rank:            10235410
Packed integer:  (rank << 6) | count
Serialized bytes, little endian: [136, 132, 11, 39]
Regenerated plane 2: 3356231810
```

The six count bits and 24 rank bits occupy 30 meaningful bits in a four-byte
container, with two unused high bits. Width 32 and plane 2 are fixed by this
scoped consumer. The [saved descriptor](expert-mask-count-rank.bin) preserves
those exact bytes; no smaller checkpoint or resident expert is claimed.

**Result: PASS.** The full [model-test evidence](expert-mask-consumer-results.json)
records the following exact comparisons:

| Check | Outcome |
|---|---|
| C rank/unrank decoder controls | 5249 / 5249 pass |
| Invalid count/rank inputs | 34 / 34 rejected |
| Selected regenerated mask | 32 / 32 bits identical |
| Reconstructed group codes | All 32 codes identical; packed bytes unchanged |
| Generated group weights | 32 / 32 bit-identical |
| Affected projection output | 1 / 1 bit-identical |
| Final normalized vector | 7168 / 7168 bit-identical |
| Final vocabulary logits | 163840 / 163840 bit-identical |

The enabled result, fully disabled control, and four-mask-consumer control
with count/rank disabled all produce checksum
`23d162dcefb18211a7540ef12948f1eb`, token 17374. The installed reference run
repeated after the candidate also matches. This is acceptance by full output
equality, not merely agreement on the emitted token.

The [independent checker](expert-mask-consumer-check.py) reuses unchanged
earlier group/output and Python rank checkers, compares against the previous
real group's data, and verifies the serialized fields, reconstructed codes,
weight/projection bits and complete outputs. The C self-check also retains
the original 4096 decode-pair and 512 code/coordinate checks in the group
helper. Neither previous reference tests nor their assertions were weakened.

Artifacts remain under `/opt/clover-k3/expert-mask-consumer-20260929-a` on
AX102: original and candidate sources, original and LF-normalized helper
copies, patched helper, decoder, both patches, checkers, reference-group
data, descriptor, stage reports, outputs, logs and `results.json`. Hashes in
the result cover all included source files, not just the main C file. The
line-ending normalization addressed patch applicability only; the original
helper was preserved. Installed model source, binary, gate and index hashes
remain unchanged.

**Scope:** one plane of one group, one expert and one five-token prefill.
The representation is constructed from the loaded original weight bytes
during the experiment; an alternative checkpoint store was not tested.
The 18-byte whole-group candidate estimates above concern generic format
choices, not a measured rewrite of the checkpoint. No other group, prompt,
decode path or full expert was substituted. Performance remains deferred;
any extension still requires its own same-result gate before advancing.

### Reading the mask change from expert to individual weights

The count/rank step goes one level smaller than the previous expert-group
representation. Its place in the data is:

```text
One routed expert: 17,547,264 packed payload bytes
  Three learned matrices
    Rows of weights
      One group: 32 weights, 17 payload bytes
        Four 32-bit masks: 16 bytes
        One shared scale: 1 byte
```

For the tested group, the existing descriptor is:

```json
{
  "scale": 121,
  "masks": [2964563821, 2011846492, 3356231810, 3458434101]
}
```

Only the third mask, plane 2, changes its description:

```text
BEFORE
  Mask:   3356231810
  Binary: 11001000000011000000100010000010

AFTER
  Count: 8
  Rank:  10235410
    -> positions [1, 7, 11, 18, 19, 27, 30, 31]
    -> mask 3356231810
```

Count says how many bits are set. Rank identifies which arrangement of those
eight set bits is meant. **This is a combination rank, not an expert's routing
rank.** Positions count from zero at the least-significant bit, on the right
of the binary display.

The regenerated mask rejoins the other three masks and scale. The group's
first eight weights are still:

```json
[-0.0078125, 0.03125, -0.0234375, 0.0234375,
 -0.015625, -0.0078125, 0.0234375, 0.03125]
```

These weights feed the projection and the remaining model. The model test
above verified the mask, all 32 weights, affected projection and complete
final norm/logits exactly on AX102.

The new description uses six count bits and 24 rank bits: 30 meaningful bits.
The implementation packs them into four bytes, `[136, 132, 11, 39]`. This is
an exact generating representation, **not a reduction of the entire expert
to those bytes**. The other masks, scale and all other weight groups remain
necessary. The test covers one mask in one group on one five-token prompt.

## 12. A complete group as counts and an arrangement rank

The user authorized the next enclosing unit after the single-mask test: the
entire sequence of 32 four-bit codes, with its shared scale. This study uses
standard enumerative coding, not a new quantization or a change to learned
weights. Only performance is deferred; the replacement must preserve results.

### The generating representation

Let $h_c$ be the count of code $c$, for $c=0,\ldots,15$. The group satisfies

$$\sum_{c=0}^{15}h_c=32.$$

For any remaining histogram $h$, define the number of possible ordered sequences

$$N(h)=\frac{(\sum_c h_c)!}{\prod_c h_c!}.$$

There are $N(h)$ arrangements of the complete group. Its zero-based
lexicographic rank $R$ identifies the exact sequence among them:

$$0\le R<N(h).$$

At position $i$, let $h^{(i)}$ be the counts remaining before selecting the
original code $c_i$. The rank is

$$R=\sum_{i=0}^{31}\ \sum_{\substack{d<c_i\\h_d^{(i)}>0}}
       N\!\left(h^{(i)}-\mathbf e_d\right),$$

where $\mathbf e_d$ subtracts one occurrence of code $d$.
Each smaller possible next code owns a consecutive block of arrangements.
Ranking sums the sizes of blocks preceding the actual sequence. Unranking
visits the possible next codes in ascending order, subtracts each block size
until the rank lies inside one block, emits that code, and repeats with its
count decremented. These disjoint blocks cover every permitted sequence, which
is why the histogram and rank recover the order exactly.

The complete representation is therefore

$$\text{group}=(s,h_0,\ldots,h_{15},R).$$

The histogram alone is insufficient: it tells how often each code appears,
not which coordinate receives it. Arrangement rank is not an expert's router
rank. Scale $s$ retains its existing E8M0 interpretation.

### The real example

The same **layer-1 expert-498 gate/w1 row-0 group-0** has this descriptor:

```json
{
  "scale_byte": 121,
  "counts": [0, 2, 4, 7, 4, 0, 1, 0, 0, 4, 3, 4, 1, 1, 1, 0],
  "rank": "8236746155644726251423865",
  "arrangements": "13113390608825899392000000"
}
```

Counts are listed in code order 0 through 15. For example, code 3 occurs seven
times and code 9 occurs four times. `arrangements` is derived from those counts,
not another stored learned field. Large integers are strings in the JSON report
to avoid losing integer precision in readers that use floating-point numbers.

The decoded sequence is:

```json
[9, 4, 11, 3, 10, 9, 3, 4, 3, 3, 11, 4, 2, 1, 2, 9,
 9, 11, 4, 6, 1, 11, 2, 3, 2, 10, 10, 12, 3, 3, 14, 13]
```

These are the original 32 ordered codes, not a sorted approximation. Their
scale remains 121, so the group's 32 decoded weights are unchanged. The
[full result](expert-group-order-results.json) retains all weights and a
position-by-position arrangement decoding trace.

### Plain-language before and after

Previously the study changed the description of one mask. This step describes
the entire group of 32 codes together. The original sequence shown above
occupies 16 bytes of four-bit codes, plus one scale byte: **17 bytes**.

The new form answers two separate questions:

```text
How often does each code appear?

Code:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Count:  0  2  4  7  4  0  1  0  0  4  3  4  1  1  1  0

Which exact ordering of those codes is this?

Arrangement rank: 8236746155644726251423865
Shared scale:    121
```

For example, seven occurrences of code 3 do not tell us where those seven
values belong. The rank restores that ordering. At each coordinate the
decoder considers possible next codes in ascending order, counts the
arrangements beginning with each, and selects the block containing the rank.
It then continues with one fewer occurrence of the selected code.

```text
Counts + arrangement rank
              |
              v
Original codes: [9, 4, 11, 3, 10, 9, 3, 4, ...]
              |
              v  apply the existing weight-value equation and scale
Original weights:
[-0.0078125, 0.03125, -0.0234375, 0.0234375,
 -0.015625, -0.0078125, 0.0234375, 0.03125, ...]
              |
              v
Expert projection and the remaining model
```

The displayed code and weight lists contain the first eight entries; the
complete 32-code sequence is above and all weights are in the linked result.
All 32 codes, all 32 weights, the affected projection and the complete final
norm/logits matched in the single-group AX102 test. Section 13 records the
subsequent broader checks on 5778 distinct groups from 214 layer/expert pairs.

The current format spends eight bits on scale, 96 on counts and 84 on this
rank: **188 meaningful bits stored in 24 bytes**, versus 17 originally.
This is a verified decomposition into **distribution plus order**, not a
compression result or removal of the learned information. The equation
applies to histograms in its stated domain; model-test evidence remains
limited to the cases actually executed.

### Complete size accounting

The current explicit format uses:

| Field | Bits |
|---|---:|
| Scale | 8 |
| Sixteen counts, six bits each | 96 |
| Arrangement rank for this histogram | 84 |
| Total meaningful payload | 188 |
| Serialized container, including padding | 192 (24 bytes) |

The original group has 136 bits (17 bytes). This candidate is therefore
**24 bytes instead of 17** for the recorded group. The 84-bit rank is a
conditional description given the histogram, not a claim that all code data
has been compressed to 84 bits. The counts must still be supplied.

The [serialized descriptor](expert-group-order.bin) places scale first, then
the 16 six-bit counts, then the rank, with each field least-significant bit
first. The rank width is computed from the counts, so it is not stored as
another field. Group length 32, alphabet size 16, field order and the
group's location are supplied by this experimental format/caller. The
four unused high bits of the final byte are zero.

A larger representation is allowed at this stage. It establishes another
exact decomposition into distribution and arrangement, not a storage,
memory or runtime improvement. No counts have been assumed free or shared
across other groups.

### Actual model consumer and checks

The [C implementation](expert-group-order.h) uses unsigned 128-bit integer
arithmetic for factorials, multinomial counts and ranks. The bound is safe
for length at most 32 because $32!<2^{128}$. No floating-point approximation
is used to choose a code or determine its rank.

The [patch](expert-group-order.patch) changes a copied expert-group helper.
At the selected site, the candidate derives the histogram/rank from the loaded
weights, serializes the descriptor, clears all four masks, reads the scale,
counts and rank back, then regenerates the ordered codes and masks. The existing
group projection consumes those regenerated values. It retains the original
floating-point lane order and supplies the recomputed output to SiTU and the
rest of the model. There is no fallback to the reference projection on a mismatch.

**Result: PASS on AX102**, for the five-token prompt
`1008,10484,318,15383,387`.

| Check | Outcome |
|---|---|
| C sequence decoder controls | 4373 / 4373 pass |
| Invalid histograms, rank or code | 5 / 5 rejected |
| Python arbitrary-integer histogram/rank and serialized decode | Exact |
| Reconstructed group codes and packed bytes | 32 / 32 codes identical |
| Decoded group weights | 32 / 32 bit-identical |
| Affected full-row projection output | 1 / 1 bit-identical |
| Final normalized vector | 7168 / 7168 bit-identical |
| Final vocabulary logits | 163840 / 163840 bit-identical |

The C controls cover every length-six sequence over four codes, all 16 constant
length-32 sequences, all 256 ordered two-code alternating sequences, the
ascending/descending balanced sequences, and first/middle/last ranks of the
balanced length-32 histogram. These are finite implementation checks, not
enumeration of all possible length-32 groups.

Fully disabled, prior four-mask consumer with arrangement coding disabled,
and enabled modes all preserve the original checksum
`23d162dcefb18211a7540ef12948f1eb`, token 17374. The repeated installed
reference matches the initial run. The original 4096 decode-pair and 512
mask-coordinate checks are also retained in the group helper. The
[independent checker](expert-group-order-check.py) uses Python factorials
and arbitrary integers, previous real-group evidence, and the unchanged
earlier full-output comparison function.

### Artifacts and scope

AX102 artifacts are under `/opt/clover-k3/expert-group-order-20260929-a`:
original and candidate sources, preserved and line-normalized base helpers,
patched group helper, arrangement decoder and patch, checkers, stage reports,
all output files, descriptor, decode trace and logs. Source, included headers,
binary and descriptor hashes are recorded in the result. Installed model
source, binary, gate and index hashes remain unchanged.

This pass covers one complete group, one affected row output, one expert and
one five-token prefill. It is not an entire row or expert replacement and
does not test another prompt or decode path. The candidate constructs the
representation from the original loaded data; no alternative checkpoint
store was installed. No earlier mask/rank, router or endpoint experiment was
combined with this build beyond reuse of the group-consumer harness. Future
extensions require their own same-result checks before advancement.

## 13. Broader coverage across selected experts and groups

The user requested tests on other experts' groups. The new coverage harness
reuses the unchanged counts/arrangement codec from section 12 and the prior
group-value helpers. It changes where the representation is applied, not its
mathematics or the reference correctness requirements.

### Sampling plan

The broad mode covers **every expert selected by the five-token input at
layers 1, 48 and 92**. Layers 1 and 48 use KDA attention; layer 92 uses MLA.
For each selected layer/expert pair, it tests all three matrices:

| Matrix | Input width | Output rows | Tested row indices | Tested group indices in each row |
|---|---:|---:|---|---|
| Gate, w1 | 3584 | 3072 | 0, 1536, 3071 | 0, 56, 111 |
| Up, w3 | 3584 | 3072 | 0, 1536, 3071 | 0, 56, 111 |
| Down, w2 | 3072 | 3584 | 0, 1792, 3583 | 0, 48, 95 |

This is 27 distinct weight groups per selected layer/expert pair: three
matrices, three rows and three groups in each row. Each affected row is
recomputed for every position using that expert. Selected groups are decoded
from serialized scale/counts/rank data; all other weights in that row retain
the original decoder. Candidate row values replace the reference values and
feed subsequent model operations only after their bits match. No fallback
substitutes a reference result if the candidate fails.

The unchanged installed source was copied for this experiment; router,
endpoint and single-mask experimental builds were not combined with it.

### Test sequence and observed coverage

First, a disabled control and a one-expert smoke test passed on AX102. The
smoke test used the first expert selected at layer 1 and exercised 27 groups,
864 weights and nine projection outputs across gate/up/down. The broad
France run proceeded only after those checks passed. The second broad input
was verified with the model's tokenizer as `The capital of Japan is`.

| Input | Selected pairs at L1 / L48 / L92 | Total layer/expert pairs | Groups | Decoded weights checked | Projection outputs checked |
|---|---|---:|---:|---:|---:|
| The capital of France is | 74 / 60 / 64 | 198 | 5346 | 171072 | 2160 |
| The capital of Japan is | 74 / 60 / 65 | 199 | 5373 | 171936 | 2160 |

Token IDs are `1008,10484,318,15383,387` for France and
`1008,10484,318,10417,387` for Japan. Both runs use five positions and fresh
prefill, with a separate unchanged reference output and routing trace for
each input.

The runs overlap. Across both, the union is **214 distinct layer/expert pairs,
5778 distinct weight groups and 184896 distinct weights**. Do not add per-run
counts and report repeated groups as new coverage. The smoke sites are a
subset of the broad France coverage and add no new unique groups.

All 16 four-bit codes occur in the samples. The broad runs exercise scale
bytes 115 through 122. They do not establish that these are the only scale
bytes in the checkpoint. Every sampled group within each broad run has a
distinct count histogram in the recorded evidence; this does not claim
histogram uniqueness throughout the model.

### Same-result evidence

**Every sampled code and weight reconstructed exactly.** All 4320 affected
projection outputs across the two broad runs match their reference values
bit-for-bit. The smoke test's additional nine comparisons also pass.

For each input, all expert IDs in the complete routing trace are identical
between reference and candidate, and the entire final normalized vector and
logit vector match:

| Input | Final norm identical | Logits identical | Output checksum | Token |
|---|---|---|---|---:|
| France | 7168 / 7168 | 163840 / 163840 | 23d162dcefb18211a7540ef12948f1eb | 17374 |
| Japan | 7168 / 7168 | 163840 / 163840 | 4b2a7b96fb6323e5639feced66f54a8e | 40484 |

Japan is compared with its own baseline, not the France checksum. An installed
France reference run repeated after the campaign still matches the initial
one. Installed source, binary, gate and model index hashes remain unchanged.

The coverage set is independently determined from each **baseline routing
trace**: expected experts, matrix shapes, rows, groups and position/rank
memberships. The checker requires exact equality of the expected and observed
site sets, rejects duplicate events, checks the terminal summary and verifies
every descriptor with Python arbitrary-precision rank and unrank. Thus a
missing group or a skipped expert cannot count as a passing test merely
because the output stayed unchanged.

The codec's 4373 finite sequence checks and five invalid-input checks, plus
the prior 4096 decode-pair and 512 mask-coordinate checks, run in each enabled
process. They are not new unique samples of checkpoint weights.

### Size, artifacts and limits

Recorded group descriptors range from **22 to 26 bytes**, versus the original
17-byte payload. Across the France samples they total 130712 bytes versus
90882; across Japan, 131377 versus 91341. These per-run figures include
repeated groups across inputs and must not be treated as unique storage totals.
Performance remains deferred; no speed, RAM or compression benefit is claimed.

The [coverage helper](expert-group-coverage.h),
[patch](expert-group-coverage.patch), and
[independent checker](expert-group-coverage-check.py) are retained beside the
[coverage results](expert-group-coverage-results.json). The result includes
every covered expert ID by layer, scale/code counts, descriptor-size totals,
artifact hashes and per-input outcomes. `K3_GROUP_COVERAGE=0` disables the
test, `1` selects the one-expert smoke scope, and `2` enables the three-layer
scope. Enabled runs require `K3_GROUP_COVERAGE_REPORT` for the JSONL trace.

Raw traces and reference routing data remain on AX102 under
`/opt/clover-k3/expert-group-coverage-20260929-a`, with sources, included
codec headers, binaries, checkers, outputs, logs, individual check reports
and aggregate `results.json`. A patch hunk-length typo was corrected during
dry-run, before compilation or model execution; no criterion was relaxed.

**Boundary:** these are sampled groups from selected experts at three layers,
not all groups in those experts or all experts in the checkpoint. Only the two
listed five-token prefills are covered. Longer contexts, other inputs,
generation/decode and alternative checkpoint loading remain untested by this
campaign. This broader pass does not remove learned data or justify skipping
same-result checks for the next extension.

## 14. Next candidate: the histogram as a constrained count vector

Following the updated verification sequence, this step identifies and checks
another data equation without starting a new model campaign. The target is
the 16 counts inside the complete-group representation, not a different
expert, layer or operator.

### The constraint removes independent count fields

The counts obey

$$h_c\ge0,\qquad \sum_{c=0}^{15}h_c=32.$$

The current layout allocates six bits to every count, for 96 bits. But the
counts are not independent: not every combination of those fields is a valid
histogram. The number of valid histograms is exactly

$$N_H=\binom{32+16-1}{16-1}=\binom{47}{15}=751616304549.$$

Thus **40 bits identify any valid histogram** in this domain:

$$\lceil\log_2 N_H\rceil=40.$$

This is a standard stars-and-bars representation. It exploits a known sum
constraint rather than assuming that a particular code distribution repeats.

### Encode and regenerate the counts

Represent the histogram by 32 stars and 15 separators, occupying 47 positions
numbered 0 through 46. For separator $j=1,\ldots,15$, its position is

$$b_j=(j-1)+\sum_{c=0}^{j-1}h_c.$$

These positions are strictly increasing, including when some counts are zero.
Rank their combination with the earlier exact combinatorial equation:

$$R_H=\sum_{j=1}^{15}\binom{b_j}{j},\qquad 0\le R_H<N_H.$$

Unrank $R_H$ to recover the separators, then read the gaps:

$$h_0=b_1,\qquad
h_c=b_{c+1}-b_c-1\ (1\le c\le14),\qquad h_{15}=46-b_{15}.$$

Every valid histogram gives exactly one separator arrangement and vice versa.
Zero counts and counts of 32 are represented exactly; no count is discarded.
Rank values at or above $N_H$ are not valid histograms.

### The same real example, one field simpler

The group used in section 12 has

```text
Counts:
[0, 2, 4, 7, 4, 0, 1, 0, 0, 4, 3, 4, 1, 1, 1, 0]

Separator positions:
[0, 3, 8, 16, 21, 22, 24, 25, 26, 31, 35, 40, 42, 44, 46]

Histogram rank:   658266380967
Arrangement rank: 8236746155644726251423865
Scale:            121
```

The two ranks have different jobs. Histogram rank recovers how many times
each code occurs; arrangement rank recovers the precise ordering of those
codes. The new complete-group representation is

$$\boxed{\text{group}=(s,R_H,R_A).}$$

```text
Histogram rank -> 16 original counts
Counts + arrangement rank -> 32 original ordered codes
Codes + scale -> 32 original weights
```

For this worked group:

| Field | Previous form | Candidate form |
|---|---:|---:|
| Scale | 8 bits | 8 bits |
| Histogram | 96 bits | 40 bits |
| Arrangement rank | 84 bits | 84 bits |
| Total payload | 188 bits | 132 bits |
| Byte-rounded group | 24 bytes | 17 bytes |

The candidate removes 56 count-field bits, or seven bytes, relative to the
previous explicit count layout. The worked example is back to the original
packed group's 17-byte size, not below it in byte-rounded storage.

### Exact reconstruction on saved AX102 groups

The [histogram checker](expert-histogram-rank.py) was run on the two existing
coverage traces from section 13. It ranks/unranks each histogram, serializes
scale plus the 40-bit histogram rank plus the arrangement rank, reloads the
fields, and reconstructs every original ordered code sequence. Rank width
for $R_A$ is derived from the recovered histogram.

**All 5778 unique groups, containing 184896 weights, reconstructed their
histograms and ordered codes exactly.** The checker identified and excluded
4941 repeated group records from the unique counts, while verifying their
original packed data and scale agreed. It also passed 105 finite controls:
all histograms for total six and four codes, every single-code histogram for
the real 32/16 shape, and extreme/middle ranks for the real histogram domain.

The [saved results](expert-histogram-rank-results.json) retain per-group ranks
and bit counts, the real example and source-trace hashes. No weight projection
or full model was rerun. This is integer reconstruction evidence on real model
data, not a new integrated model pass.

### Full sample accounting and limits

| Representation for 5778 unique groups | Group payload bytes |
|---|---:|
| Previous counts + arrangement rank | 141263 |
| Histogram rank + arrangement rank | 100817 |
| Original packed codes + scale | 98226 |

Candidate sizes range from 15 to 19 bytes: 149 groups are smaller than the
original 17 bytes, 2947 are equal and 2682 are larger. The candidate total
remains **2591 bytes larger than the original packed format**, despite being
40446 bytes smaller than the previous explicit-count representation.

The candidate uses a fixed schema: scale is eight bits, histogram rank is
40 bits and arrangement-rank width follows from decoded counts. There is
no per-group mode tag. Group location, a surrounding index and file/container
overhead are outside these payload totals. No alternative checkpoint was
built, no RAM or speed improvement was measured, and compression of the
whole model is not established.

**Initial status:** the candidate's exact integer decoding and serialization
were checked on saved data, but no integrated model run had yet taken place.
The subsequent integration test below separately establishes a model-level
pass for histogram rank plus arrangement rank at the sampled groups. Earlier
partial passes alone did not establish that new field layout's correctness.

AX102 retains `expert-histogram-rank.py` and `histogram-rank-results.json`
under `/opt/clover-k3/expert-group-coverage-20260929-a`, beside the source
traces. The reference implementation, learned weights and gates are unchanged.

### Plain-language histogram walkthrough

This step changes the description of the counts, not the group's learned
weights or their order. Before it, the group carried sixteen explicit counts:

```text
Scale: 121
Counts: [0, 2, 4, 7, 4, 0, 1, 0, 0, 4, 3, 4, 1, 1, 1, 0]
Arrangement rank: 8236746155644726251423865
```

Those counts used 96 bits, but they must sum to 32. Distributing 32 items
among 16 boxes has only 751616304549 possible outcomes. A 40-bit histogram
rank can identify any one of those outcomes exactly.

```text
Histogram rank:   658266380967
Arrangement rank: 8236746155644726251423865
Scale:            121

Histogram rank
  -> original sixteen counts
Counts + arrangement rank
  -> original ordered codes [9, 4, 11, 3, 10, 9, 3, 4, ...]
Codes + scale
  -> original weights [-0.0078125, 0.03125, -0.0234375, 0.0234375, ...]
Weights
  -> expert projection and remaining model
```

**Histogram rank recovers how many. Arrangement rank recovers where.** The
code and weight lists above are excerpts, not the complete 32-value group.

For this example the fields change from $8+96+84=188$ bits to
$8+40+84=132$ bits: 24 bytes becomes 17 after byte rounding. The information
in the counts is preserved; independent count-field capacity that represented
invalid totals is no longer required.

The integrated test recorded below passed on 5778 sampled groups from 214
layer/expert pairs across two prompts. All checked projections, routing and
final norm/logit values matched their references. That is not coverage of
every expert or group. Across this sample the new payload remains slightly
larger than original packed data, even though it is smaller than the previous
explicit-count representation. The original learned tables remain required.

### Integrated histogram and arrangement model run

The user then requested the integrated run. The implemented path is now:

```text
Original loaded group
  -> encode scale + histogram rank + arrangement rank
  -> serialize fields into bytes
  -> decode the 40-bit histogram rank into sixteen counts
  -> decode arrangement rank into the original 32 codes
  -> generate weights and execute the sampled projection rows
  -> continue all remaining model computation and produce logits
```

This integrates the two group-data equations in the real expert consumer,
not just an offline reconstruction. The [C histogram codec](expert-histogram-codec.h)
and [integration patch](expert-histogram-integrated.patch) modify an isolated
copy of the prior coverage helper. Setting `K3_HISTOGRAM_RANK=1` selects
40-bit histogram ranks; `0` retains the earlier sixteen six-bit count fields.
The arrangement decoder, value decoder and floating-point projection order
remain unchanged.

The sampling plan remains section 13's: every selected expert at layers
1, 48 and 92, all three matrices, first/middle/last rows and groups. The
forward calls execute all 93 layers, but only those sampled groups use the
new representation. This is **not integration of all earlier vector/router/
endpoint experiments**, nor replacement of every expert group in the model.

#### Checks and actual results

An unchanged reference, a fully disabled candidate, and a one-expert compact
smoke test passed first. A broad France control using the previous 96-bit
count layout also passed before interpreting the compact result. The
integrated France and Japan tests then produced:

| Input | Layer/expert pairs | Groups | Affected projections | Final norm / logits | Output checksum |
|---|---:|---:|---:|---|---|
| France | 198 | 5346 | 2160 | 7168 / 163840 identical | 23d162dcefb18211a7540ef12948f1eb |
| Japan | 199 | 5373 | 2160 | 7168 / 163840 identical | 4b2a7b96fb6323e5639feced66f54a8e |

**PASS: zero differing bits** in the checked projection outputs and complete
final normalized vectors/logits. Every selected expert ID in each full route
trace also matches that input's reference. The tokens are 17374 and 40484,
respectively; these token matches are not the acceptance criterion.

Across the two inputs, 214 distinct layer/expert pairs and 5778 distinct
groups, containing 184896 weights, were tested. Overlap is counted once.
The compact smoke test covers 27 of those groups and adds no unique groups.
Repeating the installed France reference after the campaign again matched.

Each compact process passed 105 C histogram controls and rejected four
invalid rank/count/domain cases. Existing 4373 arrangement-decoder controls,
five invalid arrangement-input checks, 4096 value decode pairs and 512
mask-coordinate checks were retained. These finite controls do not constitute
enumeration of every possible histogram or model input.

#### Independent validation without altering the prior gates

The [integration checker](expert-histogram-integrated-check.py) validates
the actual compact bytes with the previously checked Python histogram and
arrangement equations. It checks that the recovered counts match the original
32-code histogram and that the recovered ordering matches every original code.

Only after those compact-format checks does it create a separate derived audit
trace, expanding the decoded histogram back into the old count-field layout
for the **unchanged** earlier coverage validator. This is a format adapter,
not a modification of a pass condition. Raw code data, projection comparison
events and model output files are not changed. The old validator still requires
the complete expected site set derived from baseline routes, no duplicate or
missing group/projection events, and identical full outputs. Raw compact
traces and derived audit traces are preserved separately with their hashes.

The [integrated results](expert-histogram-integrated-results.json) record
both kinds of checks, controls, coverage, prompt IDs, compiler flags, source
hashes and per-input descriptor sizes.

#### Payload result and remaining boundary

The integrated consumer generated exactly the compact sizes predicted by
the saved-data study:

| Unique sampled-group payload | Bytes |
|---|---:|
| Histogram rank + arrangement rank | 100817 |
| Original packed codes + scale | 98226 |

Sizes range from 15 to 19 bytes, and total payload remains 2591 bytes larger
than the original. The histogram fields are smaller than the prior explicit
count layout; that is not a claim of smaller model storage or reduced RAM.

Artifacts reside on AX102 at
`/opt/clover-k3/expert-histogram-integrated-20260929-a`: baseline and candidate
sources/binaries, preserved and normalized helper copies, codec and patches,
all checkers, raw traces, derived audit traces, reference/control/candidate
outputs, routing traces and logs. The installed source, binary, gate and index
retain their original hashes. No alternative checkpoint was installed.

The result establishes this integration for the sampled groups in two five-token
prefills. Longer contexts, generation/decode, complete expert replacement and
an evaluator combining every earlier experiment remain outside this test.
Performance remains deferred; no further candidate was explored in this cycle.

## 15. Next candidate: histogram and arrangement as one joint rank

After recording the histogram walkthrough, the next investigation considers
the relationship between the two rank fields. A histogram determines how
many arrangements are possible, so these fields do not form an unconstrained
Cartesian product. This step checks an exact joint representation on saved
AX102 data; it does not run a new model replacement.

### Give each histogram exactly its number of arrangements

For histogram $h$, retain the arrangement count

$$N(h)=\frac{32!}{\prod_{c=0}^{15}h_c!}.$$

Order valid histograms lexicographically by their counts $h_0,\ldots,h_{15}$.
This order is different from section 14's colexicographic separator ranking.
Define the starting position of a histogram's block by

$$O(h)=\sum_{h'<_{\mathrm{lex}}h}N(h'),$$

and place its arrangement rank within that block:

$$\boxed{J=O(h)+R_A.}$$

The ranges $[O(h),O(h)+N(h))$ are disjoint and adjacent. Given $J$, locate
its histogram block, subtract $O(h)$, and recover $R_A$. The earlier arrangement
decoder then restores the original ordered codes. No histogram table needs
to be stored if block sizes are computed from the combinatorial identities.

### Compute block sizes without enumerating every histogram

Suppose counts for earlier code values have been fixed. Let $r$ be the number
of coordinates still unassigned, and let $A$ count the ways to place the already
fixed code occurrences into the original 32 coordinates. Initially $r=32,A=1$.

When considering count $q$ for the next code and leaving $m$ code values
unrestricted afterward, the block size is

$$B(q)=A\binom rq m^{r-q}.$$

Choose which $q$ of the remaining positions receive this code; every remaining
position can receive any of the $m$ later codes. For encoding, add the blocks
for counts below the actual count. Then update

$$A\leftarrow A\binom rq,\qquad r\leftarrow r-q.$$

For decoding, subtract these blocks until $J$ lies inside one, record its count,
and continue. At the final histogram there are exactly $A=N(h)$ states left;
the residual rank is its original arrangement rank. The final count is fixed
by the remaining number of positions.

### The capacity identity

Every sequence of 32 codes from a 16-code alphabet has exactly one histogram.
Partitioning all those sequences by histogram gives

$$\sum_{h:\sum h_c=32}N(h)=16^{32}=2^{128}.$$

Consequently a fixed-width joint rank uses exactly **128 bits** for the codes.
The shared scale remains eight bits: **136 bits or 17 bytes per group**.
Every 128-bit joint-rank value represents one possible code sequence; no field
padding or histogram identifier is required inside that fixed schema.

This is a reordering of the original code space, not a reduction in its
cardinality. Any injective fixed-width encoding capable of representing all
$16^{32}$ ordered sequences requires at least 128 bits for their identity.
That bound does not rule out variable-length compression of a nonuniform
collection, restrictions on reachable sequences, or shared information across
groups. Those would require additional evidence and honest accounting of the
shared information. None is established by the joint-rank result alone.

The bound concerns preservation of the complete code sequence. This study
does not attempt to merge distinct codes that might decode to equal values
under special scales, or change the reference's signed-zero behavior.

### The real example

For layer-1 expert-498 gate row 0, group 0:

```text
Counts: [0, 2, 4, 7, 4, 0, 1, 0, 0, 4, 3, 4, 1, 1, 1, 0]
Arrangement rank:       8236746155644726251423865
Histogram block offset: 26876574836885357711968228245511030048
Joint rank:             26876574836893594458123872971762453913
Scale:                  121
```

The offset plus arrangement rank equals the joint rank exactly. The group
can now be described as `(scale, joint_rank)`:

```text
Joint rank
  -> original histogram and arrangement rank
  -> original 32 ordered codes
Scale + codes
  -> original 32 weights
```

For this particular example the separate ranks needed 132 meaningful bits
including scale; the fixed joint format uses 136. Both occupy 17 bytes.
The joint representation is not shorter for every individual group: it
trades the separate layout's variable lengths for a fixed 17-byte payload.

### Plain-language joint-rank walkthrough

This step combines "how many" and "where" into one number. Before it, the
same group's representation was:

```text
Scale:            121
Histogram rank:   658266380967
Arrangement rank: 8236746155644726251423865
```

After it, the representation is:

```text
Scale:      121
Joint rank: 26876574836893594458123872971762453913
```

Imagine listing every valid histogram and giving each a consecutive block
containing all its possible arrangements. Histograms with more arrangements
receive larger blocks. The joint rank identifies both the block and the
position within it. This is not concatenating the earlier two rank fields.

For this real group:

```text
Histogram block starts at: 26876574836885357711968228245511030048
Position within that block:              8236746155644726251423865
Joint rank:                26876574836893594458123872971762453913
```

The decoder locates the histogram block and subtracts its starting offset
to recover the arrangement rank. It can then regenerate the original data:

```text
Joint rank
  -> original histogram and arrangement rank
  -> original codes [9, 4, 11, 3, 10, 9, 3, 4, ...]
Scale + codes
  -> original weights [-0.0078125, 0.03125, -0.0234375, 0.0234375,
             -0.015625, -0.0078125, 0.0234375, 0.03125, ...]
Weights
  -> expert projection and remaining model
```

The lists above are excerpts. The complete code sequence still has 32
entries. Its joint rank occupies 128 fixed bits; the scale occupies eight:
**17 bytes**, equal to the original packed group. The learned information
is preserved, not eliminated or stored in a hidden universal table.

The model integration below verified 5778 sampled groups from 214 distinct
layer/expert pairs across two prompts. All 4320 affected broad-run projections,
routing selections and complete final norm/logits matched their references.
This is a verified alternative representation, not compression below the
original format or verification of every group in the checkpoint.

### Saved-data checks and actual accounting

Ran the [joint-rank checker](expert-joint-rank.py) on the existing France and
Japan coverage traces. It encoded each unique group, serialized one scale byte
and the 128-bit joint rank, and reconstructed its histogram, arrangement rank
and original ordered codes exactly.

**All 5778 unique groups (184896 weights) reconstructed exactly.** The 4941
repeated records were checked for consistent packed bytes and scale, then
excluded from unique totals. The checker also passed 102 finite controls:
all 81 length-four sequences over three codes, all 16 constant real-width
sequences, three balanced-histogram ranks and the two extremes of the complete
128-bit rank space.

| Format for the 5778 unique groups | Payload bytes |
|---|---:|
| Separate histogram and arrangement ranks | 100817 |
| Joint rank and scale | 98226 |
| Original packed codes and scale | 98226 |

The candidate removes the separate-rank layout's **2591 extra bytes** on this
sample, returning to the original packed payload size. The histogram/order
decomposition remains useful for understanding the data, but it has not
established compression below the original format.

The [results](expert-joint-rank-results.json) retain the exact example, every
sampled group's joint rank, sizes and source-trace hashes. The Python checker
uses arbitrary-precision integers: the domain size $2^{128}$ itself needs
129 bits, even though every rank is representable in 128 bits. A future
fixed-integer implementation must handle that boundary explicitly.

### Verification state

The initial study above checked exact integer representation and serialization
on saved model data. It did not run a new expert projection or full-model pass
with the joint decoder. The earlier histogram/arrangement integration result
could not establish that outcome. The subsequent model integration below
separately verifies the new C joint decoder on the sampled expert groups.

Artifacts remain on AX102 under
`/opt/clover-k3/expert-group-coverage-20260929-a` as `expert-joint-rank.py`
and `joint-rank-results.json`, with local copies linked above. No learned
weights, reference implementation, tests or correctness gates were changed.
The 17-byte accounting assumes a shared fixed schema and externally supplied
group location; it does not include a model-store index or container overhead.

### Joint-rank model integration runs

The user authorized model integration on 2026-09-29. The
[C joint codec](expert-joint-codec.h) and
[integration patch](expert-joint-integrated.patch) were applied to an isolated
copy of the coverage helper. Original arrangement/value decoders, projection
reduction order, reference outputs and correctness gates were retained.

The enabled path in each sampled group now performs:

```text
Loaded original codes and scale
  -> histogram and arrangement rank
  -> 128-bit joint rank
  -> serialize scale byte + 16 joint-rank bytes
  -> read scale and joint rank from the serialized payload
  -> regenerate histogram and arrangement rank
  -> regenerate the original 32 ordered codes and weights
  -> replace the sampled projection results and continue the model
```

Decoded counts and arrangement are obtained from the joint payload, not
substituted from reference arrays. The original data remains for comparison;
a difference fails the run rather than causing a fallback to reference
results. This still constructs descriptors from already-loaded checkpoint
weights, not from a newly installed persistent weight store.

#### The 128-bit boundary

Each joint rank fits in an unsigned 128-bit integer, but the number of ranks,
$2^{128}$, does not. The C decoder treats every 128-bit value as valid for
the fixed length-32, alphabet-16 domain instead of evaluating that out-of-range
domain size. Smaller test domains have explicit range checks.

The algorithm calculates individual histogram-prefix block sizes, each smaller
than the complete domain. Checked additions and multiplications reject integer
overflow rather than wrapping it. The 102 finite decoder controls include
the all-zero and all-one 128-bit rank patterns, constant-code groups, balanced
histogram boundaries and the complete small-domain sequence check. Five
invalid smaller-domain, shape or arrangement cases are rejected. These are
implementation controls, not enumeration of all $2^{128}$ possible inputs.

The scale and rank are serialized as separate fields into a byte buffer:
one eight-bit scale followed by the 128-bit rank, least-significant bit first.
The code does not try to combine their 136 bits into one 128-bit integer.

#### Controls and coverage

The fully disabled candidate and one-expert smoke test passed before the
broad runs. The smoke test covers 27 groups and nine affected projections.
A France control using the older explicit 96-bit count fields also passed.
Broad coverage uses the same predetermined rows and groups as section 13,
across gate, up and down matrices for all selected experts at layers 1, 48
and 92.

| Input | Layer/expert pairs | Groups | Affected projections | Final norm / logits | Checksum |
|---|---:|---:|---:|---|---|
| France | 198 | 5346 | 2160 | 7168 / 163840 identical | 23d162dcefb18211a7540ef12948f1eb |
| Japan | 199 | 5373 | 2160 | 7168 / 163840 identical | 4b2a7b96fb6323e5639feced66f54a8e |

**PASS:** all checked codes, weights and projection outputs match, as do all
routing IDs and every final norm/logit value for each prompt's own reference.
Tokens are 17374 and 40484, respectively. Token agreement alone is not the
gate. An installed France reference repeated after the campaign also matches.

The union is **214 distinct layer/expert pairs, 5778 distinct groups and
184896 weights**. The 4320 affected projections are comparisons across the
two broad executions, not a count of distinct weights. Shared groups are
counted once in the union; the smoke sites add no extra unique groups.

#### Independent checks and payload result

The [integration checker](expert-joint-integrated-check.py) parses each actual
17-byte payload, decodes and re-encodes its joint rank with Python arbitrary
integers, compares the rank with the earlier saved-group study, and verifies
the reconstructed histogram, arrangement and ordered codes against the actual
packed group.

After validating those raw joint bytes, it writes a **separate derived audit
trace** in the former explicit-count format for the unchanged coverage
validator. Original codes, projection events and model output files are not
changed. That validator still derives expected sites from baseline routing,
rejects omissions/duplicates, and requires bit-identical projections, routes
and full outputs. Original raw traces and derived audit traces are retained
with separate hashes. Existing arrangement/value decoder controls are unchanged.

All joint descriptors are exactly 17 bytes. The 5778 unique groups total
**98226 bytes**, equal to the original packed codes and scales. This confirms
the saved-data prediction and removes the separate-rank layout's extra bytes;
it does not demonstrate compression below the original format. No speed or
memory saving is claimed.

#### Artifacts and limits

The [integration results](expert-joint-integrated-results.json) include
controls, prompt IDs, code/scale observations, covered expert lists, per-run
values, source/binary hashes and the raw/audit trace identities. AX102 retains
the full experiment under `/opt/clover-k3/expert-joint-integrated-20260929-a`,
including sources, preserved and line-normalized base helpers, patch files,
codec headers, binaries, outputs, routes, checkers and logs. Two patch-format
errors were repaired during dry-run before compilation; no test was weakened.

The installed source, binary, gate and model index retain their original
hashes. The runs execute all 93 layers but replace only the sampled groups
at three layers. They do not replace every expert group, combine the earlier
vector/router/endpoint experiments, test generation/decode or other context
lengths, or install an alternative checkpoint. Performance remains deferred.
No further equation candidate was investigated during these integration runs.

## 16. Next candidate: a row's scale bytes as base plus offsets

After recording the joint-rank walkthrough, the next investigation targets
the other field in `(scale, joint_rank)`: the scale bytes across neighboring
groups. Joint ranking reached the original fixed-width code capacity; this
candidate instead looks for a measured relationship among multiple scales.
The 128-bit code representation and all learned code values remain intact.

### The scale sequence and equation

Each gate/up row has 3584 weights and 112 scale bytes, one per 32-weight
group. A down row has 3072 weights and 96 scales. The scale data is a real
contiguous row of the checkpoint's scale tensor, not the three sparse groups
per row used in the earlier model sampling.

For a row of $G$ scale bytes, define

$$b=\min_g s_g,\qquad \delta_g=s_g-b,$$

$$d=\left\lceil\log_2(1+\max_g s_g-\min_g s_g)\right\rceil.$$

All original scale bytes are regenerated exactly by

$$\boxed{s_g=b+\delta_g.}$$

If every scale is equal, $d=0$ and there are no offset bits. Otherwise each
offset occupies $d$ unsigned bits, where $0\le d\le8$. The explicit row format
uses an eight-bit base, four-bit width field, and $Gd$ ordered offset bits:

$$\text{row payload bytes}=\left\lceil\frac{12+Gd}{8}\right\rceil.$$

This is lossless base-plus-offset encoding, not a new quantization. It always
represents the scale sequence, but may grow rather than shrink if its range
requires eight offset bits. No narrow range is assumed for unexamined rows.

For ordinary scale bytes, substitution into the weight equation gives

$$w_{g,i}=R_{32}\!\left(\operatorname{E2M1}(c_{g,i})\,2^{b+\delta_g-127}\right).$$

A future same-result consumer should recover the original byte $s_g$ and
use its already-established decode semantics, rather than introduce a new
association of floating-point multiplies. The byte-255 signed-zero special
case must remain unchanged. The stored offset is not a learned approximation
to the original exponent.

### The actual first gate row

Fresh read-only AX102 checkpoint reads identified **layer 1, expert 498,
gate/w1 row 0**. All its 112 scale bytes are either 120 or 121. The first
sixteen are:

```text
Original scales:
[121, 121, 121, 121, 120, 120, 121, 121, 121, 121, 120, 121, 120, 120, 121, 121]

Base: 120
Offset width: 1 bit
First sixteen offsets:
[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
```

An offset of zero recovers scale byte 120, whose ordinary multiplier is
$2^{-7}=1/128$; an offset of one recovers byte 121 and multiplier $1/64$.
The first group remains scale 121, as in the earlier expert-498 example.

This row's 112 scale bytes become $8+4+112=124$ bits, stored in **16 bytes**.
The code payload for its 112 groups is still 1792 bytes, so the complete row's
code-plus-scale payload becomes **1808 bytes instead of 1904**. There is no
claim that the entire row or expert is now only 16 bytes.

### Nine complete scale rows checked

The [checker](expert-scale-row.py) parses the actual binary model index using
its documented record layouts, confirms the selected tensors' dimensions and
byte offsets against their safetensors headers, and reads the first, middle
and last scale rows of all three matrices in expert 498. It cross-checks
27 previously observed scale bytes against the earlier verified France trace.

| Matrix | Row | Scales | Observed range | Offset bits | Original scale bytes | Candidate scale bytes |
|---|---:|---:|---|---:|---:|---:|
| Gate, w1 | 0 | 112 | 120-121 | 1 | 112 | 16 |
| Gate, w1 | 1536 | 112 | 120-121 | 1 | 112 | 16 |
| Gate, w1 | 3071 | 112 | 120-122 | 2 | 112 | 30 |
| Up, w3 | 0 | 112 | 120-122 | 2 | 112 | 30 |
| Up, w3 | 1536 | 112 | 120-122 | 2 | 112 | 30 |
| Up, w3 | 3071 | 112 | 120-122 | 2 | 112 | 30 |
| Down, w2 | 0 | 96 | 120-121 | 1 | 96 | 14 |
| Down, w2 | 1792 | 96 | 120-122 | 2 | 96 | 26 |
| Down, w2 | 3583 | 96 | 120-121 | 1 | 96 | 14 |

**All 960 original scale bytes reconstructed exactly** from the encoded row
payloads. Twelve finite codec controls cover offset widths zero through eight
and constant rows including byte 255. The real selected rows contain no 255
bytes. These controls are not a claim of exhaustive parser validation.

| Payload across these nine rows | Original bytes | Candidate bytes |
|---|---:|---:|
| Scale data, including new row headers/padding | 960 | 206 |
| Unchanged code data, 128 bits per group | 15360 | 15360 |
| Code and scale together | 16320 | 15566 |

The saving is **754 payload bytes on the sampled rows**. This is not a
whole-expert measurement. The format relies on known row length and order
from tensor metadata. A persistent store's row-offset index, container
overhead and random-access requirements are not included in these totals.
Scale data is only one byte of the original 17-byte group, so its percentage
reduction must not be reported as the percentage reduction of all weights.

### Status and next verification

The [results](expert-scale-row-results.json) preserve all original row bytes,
base/width fields, serialized descriptors, tensor offsets, index/trace hashes
and per-row sizes. The exact-read source matches the model index and earlier
captured scales. No model weights or installed files were changed.

**Initial candidate status:** exact scale-byte reconstruction was verified
from fresh checkpoint reads, but no scale-row consumer replacement had run.
The earlier joint-rank pass did not automatically cover this change. The
integration below separately tests recovered scales in actual projections
while retaining the original weight and accumulation semantics.

Artifacts remain on AX102 under
`/opt/clover-k3/expert-group-coverage-20260929-a` as `expert-scale-row.py`
and `scale-row-results.json`, with local copies linked above. Only this
expert's nine rows were examined; no universal scale-range restriction,
other-expert result, runtime improvement or whole-checkpoint reduction is
claimed. No further candidate was explored.

### Scale-row model integration

The user authorized the integrated model run. The
[C scale-row codec](expert-scale-codec.h) and
[integration patch](expert-scale-integrated.patch) modify an isolated copy
of the passing joint-rank coverage helper, not the installed model.

For every tested row, the candidate encodes all original scales as base,
width and offsets, decodes those serialized bytes into a new scale row, and
checks every scale and every weight decoded with it against the original.
The reconstructed scales then replace the row-scale pointer for both the
sampled joint-rank groups and the remaining packed-code weights in the row.
Recomputed row projections feed the remaining model. Original data is used
for comparison, never as a fallback after a mismatch.

#### Coverage and controls

The test covers first/middle/last rows of all three matrices for every expert
selected at layers 1, 48 and 92. Unlike earlier code sampling, **every scale
group in each tested row** is represented: 112 per gate/up row or 96 per
down row, totaling 960 scales in nine rows per selected layer/expert pair.
Joint-rank code decoding still covers only three groups per row. These are
different coverage counts and must not be merged.

A fully disabled control passed. The one-expert smoke test then passed all
nine previously measured scale rows, 960 scale bytes, 30720 decoded weights
and nine projections, with the predicted 206-byte scale payload. A broad
joint-rank control with scale encoding disabled also passed before the new
scale-row integration was evaluated.

| Input | Layer/expert pairs | Complete scale rows | Scales checked | Weights using decoded scales | Projections |
|---|---:|---:|---:|---:|---:|
| France | 198 | 1782 | 190080 | 6082560 | 2160 |
| Japan | 199 | 1791 | 191040 | 6113280 | 2160 |

**PASS: all checked scales, weights and projection values are bit-identical.**
All expert IDs in each full routing trace match, and both complete outputs
match their respective unchanged references:

| Input | Final norm identical | Logits identical | Checksum | Token |
|---|---|---|---|---:|
| France | 7168 / 7168 | 163840 / 163840 | 23d162dcefb18211a7540ef12948f1eb | 17374 |
| Japan | 7168 / 7168 | 163840 / 163840 | 4b2a7b96fb6323e5639feced66f54a8e | 40484 |

The repeated installed France reference matches the initial run. The union
across both inputs is **214 layer/expert pairs, 1926 scale rows, 205440 scale
groups and 6574080 weights**. Joint ranks cover 5778 unique code groups within
these rows. Shared rows and smoke sites are counted once; the 4320 projection
comparisons are across the two broad executions, not extra unique weights.

Each scale-enabled process passes twelve codec controls covering widths zero
through eight and constant rows including byte 255. Six malformed cases are
rejected: short descriptor, invalid width, nonzero padding, overflowing scale
sum, zero count and excessive row length. Prior joint-rank, arrangement and
value-decoder checks remain unchanged.

#### Independent checks

The [integration checker](expert-scale-integrated-check.py) decodes every actual
scale descriptor with the earlier Python codec, re-encodes it, and requires
identical original scales and payload bytes. The original nine rows also match
the earlier checkpoint-read evidence. Expected row/projection sites and position
membership come from baseline routing; missing or duplicate rows fail the check.

The earlier joint-payload and coverage/output checkers are reused unchanged
on separately generated joint-only and expanded audit traces. Scale records
are independently checked before exclusion from the joint-only view. Raw
scale traces, projection comparisons and output files are preserved; per-group
scale fields must agree with their independently decoded row values.

#### Payload accounting and limits

Across the unique tested rows:

| Payload | Original bytes | Candidate bytes |
|---|---:|---:|
| Scales | 205440 | 39914 |
| Unchanged code information | 3287040 | 3287040 |
| Code and scale together | 3492480 | 3326954 |

The saving is **165526 scale-payload bytes**. Offset widths are one bit in
1148 rows and two bits in 778 rows. This is a measured sample property,
not an imposed restriction on other rows.

Base/width headers and byte padding are included; a persistent row-offset
index and file/container overhead are not. The harness retains original
arrays, reference computation and diagnostic per-group descriptors, including
their scales. Therefore these totals are **representation accounting, not
measured checkpoint size or process RAM after deployment**. Performance remains
deferred; no whole-expert or whole-model saving is inferred.

The [complete results](expert-scale-integrated-results.json) preserve per-run
counts, controls, expert lists, checksums, widths, totals and source hashes.
AX102 retains raw, joint-only and audit traces, source copies, codec headers,
patches, binaries, checkers, original row evidence, outputs and logs under
`/opt/clover-k3/expert-scale-integrated-20260929-a`. Installed model source,
binary, gate and index hashes are unchanged.

These are complete model forwards for two five-token prompts, with replacements
in sampled rows at three layers. No entire matrix/expert, other context length,
generation/decode path, alternative weight store, or combination of all earlier
vector/router/endpoint experiments is covered. No later candidate was explored.

### Plain-language scale-row walkthrough

This step changes the scales shared by neighboring weight groups, not their
learned values. One gate row contains 3584 weights arranged into 112 groups
of 32. Each group has one scale byte, so the original row has 112 scale bytes.

For expert 498's first gate row, the first sixteen scale bytes are:

```text
[121, 121, 121, 121, 120, 120, 121, 121,
 121, 121, 120, 121, 120, 120, 121, 121]
```

All 112 scales in that measured row are either 120 or 121. Keep the smallest
one once, and describe each value by its difference from that base:

```text
Base: 120
Offset width: 1 bit
First sixteen offsets:
[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]

120 + 1 -> original scale byte 121 -> multiplier 1/64
120 + 0 -> original scale byte 120 -> multiplier 1/128
```

The generating equation is $s_g=b+\delta_g$. Wider actual ranges use more
offset bits; the 120/121 range is not imposed on other rows. Recovering the
same scale bytes preserves the weight-decoder inputs exactly.

For this row the eight-bit base, four-bit width field and 112 one-bit offsets
total **124 bits, stored in 16 bytes**:

```text
Before: 1792 code bytes + 112 scale bytes = 1904 bytes
After:  1792 code bytes +  16 scale bytes = 1808 bytes
```

That is 96 fewer payload bytes in this row. Across the integrated sample's
1926 rows from 214 layer/expert pairs, scale payload changed from 205440 to
39914 bytes. Including unchanged codes, payload changed from 3492480 to
3326954 bytes, saving 165526 bytes. Checked scales, decoded weights,
projections, routing and complete final norm/logits matched exactly on both
five-token prompts.

These are payload totals, not measured checkpoint or RAM reductions. The
row headers and padding are counted, but a persistent store's index/container
overhead is not. Original data remains in the diagnostic harness. The results
do not establish the same scale ranges or savings for unexamined rows.

## 17. Next candidate: usual scale plus exceptional positions

The next candidate keeps the validated row base and examines the offset
sequence. One-bit offsets need not occur equally often. Rather than always
store one or two bits at every group, describe the most common offset once
and identify the positions and values of exceptions. This is exact data
representation, not thresholding or dropping unusual weights.

### The equation

For $G$ groups, retain $b=\min_g s_g$ and width $d$ from section 16.
Let $u$ be the most frequent offset $s_g-b$, choosing the smaller offset on
a frequency tie. Define the exception set

$$P=\{g:s_g-b\ne u\},\qquad k=|P|.$$

Then

$$s_g=b+\begin{cases}
u,&g\notin P,\\
v_j,&g=p_j\in P,
\end{cases}$$

where $p_1<\cdots<p_k$ are the exception positions and $v_j$ their original
offsets. The positions have an exact combination rank

$$R_P=\sum_{j=1}^{k}\binom{p_j}{j},\qquad 0\le R_P<\binom Gk.$$

The count $k$ is part of the description. Rank/unrank regenerates exactly the
same positions, using the same combinatorial identity as the earlier mask
study, now at row length 96 or 112 instead of 32.

Exception values cannot be omitted in general. Of the $2^d$ possible offsets,
one is the usual value, leaving $Q=2^d-1$ alternatives. Map each exception to
a digit $z_j=v_j$ if $v_j<u$, or $z_j=v_j-1$ otherwise. Encode their values in
position order as

$$R_V=\sum_{j=1}^{k}z_jQ^{j-1},\qquad 0\le R_V<Q^k.$$

Repeated integer remainder/division recovers those digits; skip over $u$
to recover the original offsets. If $d=1$, there is only one non-default
value, so $Q=1$ and no value-rank bits are needed. If $d=0$, there are no
exceptions and both position/value fields are empty. These cases do not
discard information: the missing values are determined by the other fields.

### Complete field costs and an adaptive choice

For exceptions, the payload before byte rounding is

$$B_E=12+d+\lceil\log_2(G+1)\rceil
       +\lceil\log_2\binom Gk\rceil
       +\lceil\log_2 Q^k\rceil.$$

The 12 bits are the eight-bit base and four-bit width; $d$ bits retain $u$.
The count, position rank and value rank are all explicitly charged. A field
with only one possible value uses zero bits. The $d=0,k=0$ case has no value
field rather than requiring evaluation of a logarithm at zero.

The existing fixed-offset payload is $B_F=12+Gd$. A tagged adaptive row stores
one choice bit followed by the shorter candidate payload:

$$B_{\mathrm{adaptive}}=1+\min(B_F,B_E).$$

Ties select fixed offsets. The format keeps both choices so it does not assume
every row has sparse exceptions. Each row is padded to a byte boundary; known
row length and ordering are supplied by tensor metadata. There is no hidden
per-row default, count, position table or exception-value table.

### A concrete row, one step further

For the same layer-1 expert-498 gate row 0:

```text
112 original scales
  95 values are 121
  17 values are 120

Base: 120
Width: 1 bit
Usual offset: 1 (scale 121)
Exception count: 17
Exception positions:
[4, 5, 10, 12, 13, 17, 19, 29, 61, 64, 71, 82, 85, 92, 104, 105, 106]
Position rank: 23143654914149171162
Exception value rank: 0 (no stored bits; every exception is offset 0)
```

Start every coordinate with scale 121. Decode the 17 positions and set each
to scale 120. This recovers all original bytes, including the first-sixteen
example in the preceding walkthrough.

| Field | Bits |
|---|---:|
| Adaptive choice tag | 1 |
| Base and width | 12 |
| Usual offset | 1 |
| Exception count | 7 |
| Exception-position rank | 66 |
| Exception values | 0 |
| Total | 87, stored in 11 bytes |

The scale row progresses from **112 original bytes to 16 fixed-offset bytes
to 11 adaptive exception bytes**. With the row's unchanged 1792 code bytes,
the candidate row payload is 1803 bytes rather than 1808 for fixed offsets
or 1904 originally. This does not mean the complete row fits in 11 bytes.

### Saved-data results

Ran the [exception checker](expert-scale-exceptions.py) on the complete scale
rows captured by the previous model integration. It first checks the original
fixed-offset descriptor with the unchanged Python codec, then encodes,
serializes and decodes every row under all three candidate formats.

**All 1926 unique rows, containing 205440 scale bytes, reconstruct byte-for-byte.**
The 1647 repeated row records across the two prompts are checked for consistent
original data and excluded from the unique totals. Another 804 finite roundtrip
checks cover all eight-position binary sequences, all offset widths zero through
eight, and constant rows including byte 255, under all three formats.

| Scale-row representation | Bytes for the unique sampled rows |
|---|---:|
| Original scale bytes | 205440 |
| Validated fixed-offset format | 39914 |
| Exceptions-only candidate | 23585 |
| Tagged adaptive candidate | 23830 |

The adaptive candidate chooses exceptions in 1897 rows and fixed offsets in
29. After byte rounding it is smaller than the existing fixed-offset form
in 1879 rows and equal in 47; no row grows in this sample. The exceptions-only
total is slightly smaller overall because it avoids choice tags, even though
it can lose against fixed offsets on individual rows. The adaptive format
explicitly pays for its ability to choose; these are distinct candidate layouts.

Keeping the same 3287040 code bytes, adaptive code-plus-scale payload becomes
**3310870 bytes**, compared with 3326954 using fixed offsets: 16084 additional
payload bytes saved on the sampled rows. Index/container overhead is excluded;
this is not a measured checkpoint or RAM reduction.

### Verification status

The [saved results](expert-scale-exceptions-results.json) include row identities,
exception counts/ranks, all three format costs, trace hashes and the complete
worked-row scales and positions. Raw source traces are from the earlier passing
France/Japan scale integration. At this initial stage the exception decoder
had not run inside the model, so these reconstruction results alone did not
establish a new projection or final norm/logit pass.

To accept a consumer replacement, an integration must regenerate
the original scale bytes from the new descriptor, feed them into the same
weight/projection path, and compare the complete model outputs. Prior passing
formats do not automatically verify new code. The subsequently authorized
integration below separately passed those checks. Performance remains deferred.

AX102 retains `expert-scale-exceptions.py` and `scale-exceptions-results.json`
under `/opt/clover-k3/expert-scale-integrated-20260929-a`. The installed model,
learned values and correctness gates are unchanged. The evidence covers the
sampled rows only, not an entire expert/checkpoint or a universal distribution
of scale values. No further candidate was investigated.

### Adaptive exception model integration

The user authorized the model integration. The
[C codec](expert-scale-exception-codec.h) and
[integration patch](expert-scale-exception-integrated.patch) were added to
an isolated copy of the verified scale-row plus joint-rank test path.
`K3_SCALE_EXCEPTIONS=1` uses the adaptive format; `0` retains fixed offsets.

Each enabled row is encoded to the format described above, including the
choice tag. The consumer reads the emitted bytes, regenerates exception
positions and values (or reads fixed offsets when that mode was chosen),
and recovers every original scale byte. Those reconstructed scales feed
the actual row's weight decoding and projection. A mismatch terminates the
test; choosing fixed-offset mode for a row is an encoded format decision,
not a fallback after a failed correctness check.

#### Exact integers without a hidden width restriction

Wide rows can require more than 128 bits for an exception-value rank. The
implementation uses GMP exact integers so that it implements the same
equation for the full supported domain, not only the observed one/two-bit
offset rows. It supports lengths 1 through 112 and widths zero through eight.

GMP development headers were not installed on AX102. The Ubuntu
`libgmp-dev` package, version `2:6.3.0+dfsg-2ubuntu6.1`, was downloaded and
unpacked under the experiment's `deps/` directory; its static archive was
linked into the candidate. **No system package was installed.** Package,
header and static-library hashes are included in the result alongside code
and binary hashes.

The C roundtrip controls cover the earlier 804 checks plus three checks of
a wide, 112-value row under fixed, exceptions-only and adaptive formats,
for **807 successful checks**. The wide case tests the large-rank path even
when adaptive selection would choose fixed offsets. Five malformed input
cases are rejected. Original scale, joint-rank, arrangement and value-decoder
checks remain enabled and unchanged; no correctness criterion was weakened.

#### Model results

A disabled control and a one-expert smoke run passed first. For the original
nine expert-498 rows, adaptive scales occupy 123 bytes instead of 206 fixed-
offset bytes, with identical scales, decoded weights and projections. A broad
France fixed-offset control then passed before evaluating adaptive rows.

| Input | Layer/expert pairs | Scale rows | Scales checked | Weights checked | Projection outputs | Adaptive scale bytes |
|---|---:|---:|---:|---:|---:|---:|
| France | 198 | 1782 | 190080 | 6082560 | 2160 | 22054 |
| Japan | 199 | 1791 | 191040 | 6113280 | 2160 | 22136 |

**PASS:** all checked scale bytes, decoded weights and projection outputs
match exactly. Complete route traces are identical, and both model outputs
match their own unchanged references:

| Input | Final norm identical | Logits identical | Checksum | Token |
|---|---|---|---|---:|
| France | 7168 / 7168 | 163840 / 163840 | 23d162dcefb18211a7540ef12948f1eb | 17374 |
| Japan | 7168 / 7168 | 163840 / 163840 | 4b2a7b96fb6323e5639feced66f54a8e | 40484 |

An installed France reference repeated after the campaign still matches.
The union is 214 layer/expert pairs, 1926 rows, 205440 scale groups and
6574080 weights. Joint-code decoding remains sampled at 5778 groups, while
all scales in each tested row use the new representation. The broad runs
compare 4320 projection outputs; repeated rows and the smoke sites are not
counted again as unique coverage.

#### Independent verification

The [integration checker](expert-scale-exception-integrated-check.py) decodes
and re-encodes every actual adaptive payload using the earlier Python codec
and compares its size, bit count and mode choice with the saved per-row
prediction. This validates the bytes emitted by the model run, not merely
an in-memory array that bypasses the new format.

After this check, the harness creates separate fixed-offset audit records
with the same decoded scales and correctly recomputed representation byte
totals. The unchanged scale-row, joint-rank and coverage/output validators
then check those derived views. Original scale values, weight/projection
results, model outputs and raw traces are not rewritten. Expected row sites
and position memberships come from the reference routing trace; missing
rows cannot silently pass. Raw and derived artifacts remain separate.

#### Payload outcome and scope

Across the unique tested rows:

| Payload | Original | Fixed offsets | Adaptive exceptions |
|---|---:|---:|---:|
| Scale bytes | 205440 | 39914 | 23830 |
| Unchanged code bytes | 3287040 | 3287040 | 3287040 |
| Combined payload bytes | 3492480 | 3326954 | 3310870 |

This is **16084 fewer payload bytes than fixed offsets**, and 181610 fewer
than the original code-plus-scale representation on the same sample.
The adaptive codec chose exception coding for 1897 unique rows and fixed
offsets for 29. The total exactly matches the earlier saved-data prediction.
Tags, headers and padding are included; persistent indexing and container
overhead are not. The harness retains original data and diagnostic records,
so these are not measurements of process RAM or deployed checkpoint size.

The [complete integration results](expert-scale-exception-integrated-results.json)
retain counts, checks, per-input outcomes, mode choices, payload sizes,
dependencies and source/binary hashes. AX102 artifacts are under
`/opt/clover-k3/expert-scale-exception-integrated-20260929-a`, including
raw, fixed-offset, joint-only and audit traces, code copies, private GMP
files, controls, outputs and logs. Installed model source, binary, gate and
index hashes remain unchanged.

The passes cover two five-token prefills with selected rows from all three
expert matrices at layers 1, 48 and 92. They do not cover every row/expert,
new context lengths, generation/decode, an alternative persistent store or
all earlier vector/router/endpoint experiments combined. Performance remains
deferred. No further equation candidate was explored during this integration.

## 18. Reverse-order cumulative integration

The user directed that all thirteen inventory items be combined in reverse
order, one at a time, without skipping ahead to an all-at-once build.
The [ordered checklist and evidence log](reverse-integration.md) records that
sequence. **All thirteen cumulative steps passed**, each before the next item
was introduced. No new equation candidate was substituted for an inventory item.

Step13 established adaptive scales with packed-code consumers, without silently
starting from the previous joint-rank consumer. Step12 added explicit base-offset
reconstruction; steps11 through7 then added joint ranks, histogram ranks,
count/arrangement descriptors, mask count/rank and four-mask value decoding.
These overlapping representations are explicit serial reconstruction stages,
not additive compression savings.

Step6 added the tail source-expression consumer; step5 added entry row references;
step4 added the rounded router source expression. Steps3,2,1 then introduced
the previously data-only vector codecs into the generated L84/pos0 vector that
feeds router normalization/projection: both zlib layouts, both shared-exponent
block sizes, and the signed odd-integer component equation. The component
serialization has an explicit zero flag to preserve signed zero; its descriptor
size is not the earlier nonzero-only analytical estimate.

The [campaign report](reverse-integration-results.json) records13passing builds
and26broad model runs. Every final norm/logit and route trace matches its own
France/Japan reference. The final sampled expert scope remains1926rows from
214layer/expert pairs,5778code groups and205440scales. Vector codecs cover one
complete7168-value generated vector per prompt; endpoint and router stage
checks remain active. This does not generalize to all rows or context lengths.

The [final tested source](reverse-integration/step1/candidate.c),
[final step result](reverse-integration/step1/results.json), and all prior
source/result snapshots are preserved locally. Protected source/binary/gate/
index hashes remain unchanged, and a reference repeated after the campaign
still matches. Private GMP/zlib dependencies were unpacked inside the experiment,
not system-installed. No persistent store, checkpoint/RAM saving, generation
coverage or performance improvement is claimed by this correctness integration.