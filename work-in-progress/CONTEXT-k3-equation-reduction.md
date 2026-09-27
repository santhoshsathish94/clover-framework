# Context — K3 equation reduction

Working context file for the reduction exploration in `k3-equation-reduction.md`.
Written during the work, not after. Each stage is recorded when it completes,
including the ones that establish nothing.

**Reference that must not move:** `k3-model-equation.md`, the bit-exact equation.
79,742,816 / 79,742,816 prefill, 96,587,584 / 96,587,584 with one decode step,
emitted token 17374 then 20829.

---

## Goal

Find result-preserving transformations that make computation genuinely disappear,
in the sense of

$$1+2+\cdots+n \;\longrightarrow\; \frac{n(n+1)}{2}$$

not in the sense of a faster implementation of the same work.

**Success criterion, fixed in advance.** A candidate reduction must either

1. reproduce the engine bit-exactly, or
2. be explicitly labeled as exact-in-real-arithmetic-only, with the deviation measured.

A reduction that changes the result is not a reduction. A reduction that removes
computation only by relocating it is not a reduction either.

---

## Established

| | |
|---|---|
| The equation is executable and exact | Python and C both reproduce the engine bit for bit |
| C evaluation costs 50.70 s | against the engine's 56.14 s, warm page cache, same token, logits identical |
| `-ffp-contract=off` is required | otherwise the compiler fuses `a*b+c` where the engine rounds twice |
| KDA state is exactly rank $t$ | measured, eight-order gap between $s_t$ and the noise floor |
| The factored KDA state is a valid sufficient statistic | in real arithmetic; reproduces `kda.o` to 1.9e-08 |
| Snapshot normalization is cacheable bit-exactly | 4,048 of 4,974 removed, output byte-identical |
| The SiTU caps never bind anywhere measured | max $|g|/b_1 = 0.56$, max $|u|/b_2 = 0.04$ |
| Gate and up rows are strongly aligned per channel | mean $\|\cos\|$ 0.622 matched against 0.011 shuffled, 55x, with control |
| Backward query propagation is sound, once corrected | float32 epsilon; the document's own form errs by 28% |
| The final norm scalar cannot change the argmax | confirmed; top-1000 ordering preserved, full ordering not |
| 11.5% of expert weights are exactly zero | stable across experts, std 0.026%; 5.19% of the whole pass |
| The depth trajectory has a period-12 cycle | autocorr 0.656 at lag 12; push-layer turns 2.40x the rest |
| The cycle is architectural, not input-dependent | holds on all 5 prefill positions and the decode position |
| Eleven functionals are phase-locked beyond chance | permutation null, 2000 shuffles, strongest z = -15.35 |
| Residual components are roughly Gaussian at most layers | L1/L2 peaks at sqrt(2n/pi) in every trajectory |
| The model is I/O bound, not compute bound | 16 to 8 threads costs 10%; 144.72 GB at 2.69 GB/s is the whole runtime |
| Every run fetches ~100 to 157 GB from the device | 1.45 TB of experts against 124 GB of RAM; there is no warm cache |

---

## Ruled out

| candidate | how it failed | evidence |
|---|---|---|
| Lossy $\phi$ on the whole KDA state space | $o = S^\top q$ with $q$ unconstrained makes $S \mapsto (q \mapsto S^\top q)$ injective; no lossy map can exist | algebra, not measurement |
| Factored KDA state as a bit-exact substitute | reassociates the arithmetic; 7,463 of 61,440 values identical | step below |
| Factored KDA state for long sequences | cost $2 \cdot 128 \cdot t$ beats $128^2$ only while $t < 64$ | arithmetic |
| Snapshot caching as a speedup | exact, but 0.0108% of the pass, 1 part in 9,219 | timings below |
| Single-snapshot ablation as a redundancy test | individual removability does not compose | step below |
| Exact repetition among $(g,u)$ pairs | zero duplicates at all three SiTU sites | step below |
| Exact per-channel collapse of the $(g,u)$ pair | no row pair is parallel; closest $\vert\cos\vert = 0.9982$ | step 3 below |
| Affine recurrence along the channel index | $R^2 \approx 0$, control scores higher in 3 of 6 | step 4 below |
| Periodicity in the channel sequence | max autocorrelation 0.019 to 0.056, no peak | step 4 below |
| Generating structure in weight row order | adjacent rows 0.01154 against shuffled 0.01184 | step 4 below |
| Block recurrence in the channel dimension | at or below chance at 128, 512 and 2048 | step 4 below |
| Cauchy-Schwarz bounds on the argmax | prunes 0 of 163,840 rows; overshoots 6x at the winner | step 27 below |
| Rank deficiency in the MLP projections | full rank by LU and by Gram spectrum | step 5 below |
| Dropping the final norm scalar as a speedup | exact for the argmax, but 0.0000040% of the pass | step 28 below |
| The transition matrix used in Steps 13 to 26 | $D_\alpha - \beta kk^\top$ gives 28% error; the truth is $(I-\beta kk^\top)D_\alpha$ | step 14/18 below |
| A shared low-dimensional subspace across experts | two experts already span all of $\mathbb{R}^{3584}$ | experts below |
| Duplicate or reused experts | 896 distinct of 896; differ across layers too | experts below |
| Skipping the 11.5% exactly-zero expert weights | perfectly scattered; 0 all-zero blocks of any size | experts below |
| A conserved quantity around the 12-layer cycle | every candidate at or above the random control's CV | cycle E below |
| Replacing any invariant with a numeric constant | best candidate ranges 16.19 to 67.87 against a fixed 67.55 | cycle F below |
| Arithmetic reduction as a way to make K3 faster | operations are not the binding resource; bytes are | cycle G below |

---

## Remaining

Open items from the Step-41 program in `k3-equation-reduction.md`:

- rank structure of the $(g,u)$ channel data
- recurrence candidates among channels
- shared generating relation across channels
- whether any candidate survives K3's arithmetic
- whether any candidate removes work rather than moving it

Open items from elsewhere in that document, not yet tested:

- Step 14/18, backward query propagation $y_{i-1} = D_{\alpha_i}y_i - \beta_i k_i(k_i^\top y_i)$
- Step 28, the final RMSNorm scalar being irrelevant to the argmax
- Step 27/34/37, branch and bound on the argmax through exact bounds

---

## Evidence log

### Cycle A — KDA minimal sufficient statistic

**Direction.** Is the $128\times128$ per-head state minimal?

**Correction found first.** The document's Step 13 gives $S' = (D_\alpha - \beta kk^\top)S + \beta kv^\top$.
The code reads $u$ from the *decayed* state, so the correct form is

$$S' = (I - \beta kk^\top)D_\alpha S + \beta kv^\top$$

The two differ by $\beta kk^\top(D_\alpha - I)S$, nonzero whenever $\alpha \neq 1$.

**Measured.** Singular spectrum of $S$, head 0, layer 0:

```
pos  s0          s1          s2          s3          s4          s5          rank
0    2.585e-01   3.926e-09                                                   1
1    3.811e-01   1.096e-01   6.379e-09                                       2
2    5.024e-01   1.746e-01   5.586e-02   8.961e-09                           3
3    4.428e-01   2.543e-01   6.834e-02   3.387e-02   1.095e-08               4
4    4.824e-01   1.780e-01   1.053e-01   2.945e-02   2.234e-02   1.258e-08   5
```

Rank is exactly $t$. The factored recurrence

$$U' = [\,D_\alpha U \mid \beta k\,], \qquad V' = [\,V \mid v - V(U^\top k)\,], \qquad o = V'(U'^\top q)$$

never materializes $S$ and reproduces `kda.o` to 1.86e-08 absolute, 2.4e-07 relative,
with 7,463 of 61,440 values bit-identical. Cost 76.8x fewer multiply-accumulates at
$t = 0$, 15.4x at $t = 4$, break-even at $t = 64$.

**Outcome.** A genuine sufficient statistic. Not bit-exact. Bounded to short sequences.

**Instrument error, sixth of the project.** The first rank measurement reported 94, 127, 128.
That used a float64 tolerance on a float32-rounded matrix and was counting rounding noise
as rank. A rank test inherits the precision of the data, not of the library.

### Cycle B — snapshot normalization caching

**Direction.** The eight snapshots are frozen when pushed, so $pr_i = v_i \cdot \mathrm{inv}_i$
never changes. Caching it involves no reassociation, so it should stay bit-exact.

**Measured.**

```
AR normalizations : 926 computed, 4048 reused     81% removed
cached vs ENGINE, final.norm :   7168 / 7168
cached vs ENGINE, logits     : 163840 / 163840    max abs diff 0
baseline 47.54 s, 51.68 s      cached 49.83 s, 49.68 s
ops eliminated 5.8e+07, which is 0.0108% of the pass, 1 part in 9,219
```

**Outcome.** The first elimination that is both genuine and bit-exact. Worth nothing
measurable: the saving is ~0.005 s against run-to-run noise of about 4 s.

### Cycle C — snapshot ablation

**Direction.** Test whether individual snapshots are informationally necessary.

**Measured.** Baseline token 17374, margin 3.678 over token 384.

Single removals:

```
drop 0 (layer  0) : 163589   margin 0.580     token changed
drop 1 (layer 12) : 220      margin 1.367     token changed
drop 2 (layer 24) : 17374    margin 3.641
drop 3 (layer 36) : 17374    margin 3.530
drop 4 (layer 48) : 17374    margin 4.008
drop 5 (layer 60) : 17374    margin 4.801
drop 6 (layer 72) : 17374    margin 3.884
drop 7 (layer 84) : 17374    margin 3.712
```

Subsets:

```
{7}        17374   3.404        {4,5,6,7}   276     token changed
{6,7}      17374   3.404        {2,3,4,5}   276     token changed
{5,6,7}    17374   1.025        {2..7}      13      token changed
{2,3}      17374   3.569        {0,1}       220     token changed
{4,5}      17374   3.185        all         13      token changed
{3,5,7}    17374   3.726
{2,4,6}    17374   0.013
```

**Outcome.** Six of eight snapshots are individually removable without changing the token,
and four of those six together are not. Individual removability does not compose.

Two further facts that matter more than the token counts. `{2,4,6}` survives with a margin
of 0.013, so the token is preserved by a hair rather than by invariance. And `{3,5,7}` has
margin 3.726, *higher* than keeping everything, so removal is not simply loss of support.
The winning logit falls from 18.113 to between 12.9 and 17.9 in every surviving case and
the runner-up changes identity nearly every time.

**Outcome, stated precisely.** These runs measure argmax robustness, not sufficiency.
Under the document's own criterion — $\phi(S_1)=\phi(S_2) \Rightarrow G(S_1,q)=G(S_2,q)$
for all $q$ — dropping a snapshot fails immediately, because the logit vector changes.
Argmax preservation discards the evidence needed to judge sufficiency.

### Step 1 — capture the real $(g,u)$ pairs

**Note on the document.** Step 39 writes the pairs as $(g_0,u_0),\dots,(g_{7167},u_{7167})$.
The dense MLP has $W_{\text{gate}} \in \mathbb{R}^{33792\times7168}$, so there are 33,792
pairs, not 7,168. SiTU also appears at two other widths. All three were captured.

**Measured.** Capture verified by rebuilding the engine's own outputs from it:

```
dense MLP  layer 0 : (5, 33792)   ffn.out    35840 / 35840   VERIFIED
shared exp layer 1 : (5,  6144)   shared_out  7168 / 7168    VERIFIED
routed exp layer 1 : (16, 3072)   experts 853, 543, 695, 500, ...

248,832 pairs in total
```

### Step 2 — exact equality and repetition

**Measured.** Exact float32 bit patterns, no tolerance.

```
                        pairs    distinct pairs   dup    g==u   exact zeros
dense  layer 0         168960    168960            0      0      0
shared layer 1          30720     30720            0      0      0
routed layer 1          49152     49152            0      0      0

cross-position repeats  dense 0 of 337920, shared 0 of 61440
cross-expert repeats    routed 0 of 368640

repeats within g alone  126 of 168960, 6 of 30720, 13 of 49152
```

**Outcome.** No exact repetition anywhere. The handful of repeats in $g$ alone never
coincide with a repeat in the matching $u$, and at these counts they are consistent with
float32 collisions among values drawn from a narrow range. The "repeated pairs" candidate
from Step 39 is ruled out at all three SiTU sites.

The caps do not bind: max $|g|/b_1$ is 0.082, 0.560, 0.297 and max $|u|/b_2$ is
0.012, 0.015, 0.040 at the three sites.

### Step 3 — rank structure

**(a) Spectra of the captured activations.** Nothing is rank-deficient.

```
dense  g  (5, 33792)   rank 5 of 5    6.017e+00 4.273e+00 3.818e+00 3.628e+00 3.378e+00
dense  u  (5, 33792)   rank 5 of 5    5.896e+00 4.245e+00 3.737e+00 3.638e+00 3.384e+00
shared g  (5,  6144)   rank 5 of 5    7.538e+00 4.352e+00 3.854e+00 3.567e+00 3.320e+00
routed g  (16, 3072)   rank 16 of 16  9.266e+00 9.211e+00 9.096e+00 8.991e+00 8.921e+00
```

**(b) Per channel, is $u_i$ proportional to $g_i$?** The 2-by-$n$ matrix $[g_i; u_i]$ has
rank 1 exactly when $|\cos| = 1$.

```
dense   33792 channels   |cos| min 0.000009  median 0.817263  max 0.999981577
shared   6144 channels   |cos| min 0.000061  median 0.390219  max 0.991706059
routed   3072 channels   |cos| min 0.000000  median 0.229412  max 0.870805692

channels with 1-|cos| < 1e-3 : dense 185, shared 0, routed 0
channels with 1-|cos| < 1e-5 : dense   0, shared 0, routed 0
```

**No channel is exactly proportional at any site.** Exact per-channel collapse of the
pair is ruled out.

**(c) The weight rows behind it.** Proportionality of the activations for *every* input,
rather than just this prompt, would require $W_{\text{gate}}[i] \parallel W_{\text{up}}[i]$.

```
cos(W_gate[i], W_up[i]) : min -0.995756  median -0.006592  max 0.998230
rows with |cos| > 0.99  : 24 of 33792
rows with |cos| > 0.50  : 22790 of 33792
rows exactly parallel   : 0
distinct rows, exact bytes : 33792 of 33792 in both matrices
```

**Control, because the number above is surprising enough to be an artifact.**

```
gate[i] vs up[i]   MATCHED   mean|cos| 0.62226  frac>0.5 0.6747  max 0.9963
gate[i] vs up[j]   shuffled  mean|cos| 0.01130  frac>0.5 0.0000  max 0.1994
gate[i] vs gate[j] shuffled  mean|cos| 0.01136  frac>0.5 0.0000  max 0.1903
random 7168-dim expectation  0.00942
```

Both shuffled controls sit on the random expectation. The matched pairing is 55 times
higher. The alignment is real and is a property of the *pairing*, not of the matrices.

**Outcome.** Two findings that must not be merged.

1. **Exact proportionality is ruled out.** No row pair is parallel, the closest is
   $|\cos| = 0.9982$, and the orthogonal remainder there is still 6% of the norm. Under
   the success criterion fixed above, this is not a reduction.
2. **A strong approximate structure exists and is measured.** For two thirds of dense
   channels the gate and up projections point along nearly the same axis, up to sign.
   That is a real fact about the learned weights, established with a control, and it is
   the first structural regularity this exploration has found that is not an artifact of
   the architecture's declared shapes.

It is recorded as a fact, not as a reduction. Turning it into one would require accepting
approximation, which the criterion for this work excludes.

### Step 4 — recurrence candidates

**Not attempted, and why.** A position recurrence $x_{t+1} = F(x_t)$ cannot be tested at
this sample size. Five positions give four transitions; a general map on $\mathbb{R}^{7168}$
has 51.4 million parameters. Any fit would be exact by construction and would mean nothing.
Testing it would have produced a number, and the number would have been worthless.

**(a) Affine recurrence along the channel index**, $g_{i+1} = a g_i + b$, fit over all
channels. $R^2 = 1$ would mean the sequence is generated by iterating an affine map.

```
                measured        shuffled control
dense  g        +0.000000       +0.000006
dense  u        +0.000007       +0.000010
shared g        +0.000678       +0.000001
shared u        +0.000091       +0.000251
routed g        +0.000003       +0.000103
routed u        +0.000066       +0.000489
```

Indistinguishable from the shuffled control, and in three of six cases the control scores
*higher*. No affine recurrence.

**(b) Consecutive weight rows**, $W_{\text{gate}}[i]$ against $W_{\text{gate}}[i+1]$.

```
adjacent rows   mean|cos| 0.01154   std 0.01640
shuffled rows   mean|cos| 0.01184   std 0.02203
norm ratio ||W[i+1]|| / ||W[i]||    mean 1.01139   std 0.13300
```

Adjacent rows are no more related than randomly paired ones, and the norm ratio is not
fixed. Row order carries no generating structure.

*Control artifact, noted so it is not misread later:* the shuffled comparison reported
`max |cos| = 1.00000`. A random permutation of 4,000 indices has about one fixed point,
and a fixed point compares a row with itself. That 1.0 is the permutation, not a finding.

**(c) Periodicity by autocorrelation**, lags 1 to 2048.

```
dense   max |autocorr| 0.01948 at lag 1498
shared  max |autocorr| 0.04582 at lag 71
routed  max |autocorr| 0.05579 at lag 129
```

An exact period would give 1.0. Nothing here is above noise.

**(d) Block recurrence**, does channel block $j+1$ follow from block $j$.

```
block  128 : 264 blocks, mean|cos| 0.06886     chance 1/sqrt(128)  = 0.088
block  512 :  66 blocks, mean|cos| 0.03317     chance 1/sqrt(512)  = 0.044
block 2048 :  16 blocks, mean|cos| 0.01674     chance 1/sqrt(2048) = 0.022
```

Every value sits at or below the chance level for vectors of that dimension.

**Outcome.** No recurrence of any tested kind, at any of the three SiTU sites. The
cycle hypothesis of Step 8 finds no support in the channel dimension. This is a clean
negative, and combined with Step 3 it means the only structure found so far in the
$(g,u)$ data is the gate-up row alignment, which is approximate.

### Step 27 viability — can bounds decide the argmax early

Run out of order, because it is independent of the Step-41 program and the data was
already on disk.

The proposal is exact branch and bound: settle the argmax without evaluating all
163,840 logits. The natural cheap bound is Cauchy-Schwarz, since row norms can be
precomputed once and reused for every token ever generated.

```
||x||             47.735310
||W_j||           0.710300 .. 3.580537,  mean 2.277318
bound ||W_j|| ||x||   33.906 .. 170.918
true |logit|           0.000 ..  18.113

bound / |logit| at the winner : 5.960
bound / |logit| at the median : 37.285

rows pruned : 0 of 163840   (0.00%)
```

**Ruled out.** The bound assumes $\cos = 1$. The winner sits at $\cos \approx 0.168$ and
the median row at $\approx 0.027$, so in 7168 dimensions the norm product overshoots by
6x even in the best case and nothing is eliminated.

Two things worth keeping separate from that negative. It is a negative for *this bound*,
not for the principle — the document itself warns at Step 38 that a valid bound need not
be a reduction, and this is that case, now measured. And the target geometry is extremely
favorable: exactly **one** row lies within 1.0 of the winner and the top-1 is isolated by
3.678, so a tight enough bound would prune essentially the whole vocabulary. What fails
is the looseness, not the idea.

### Environment finding — no optimized BLAS

Discovered while Step 5 was running far slower than estimated.

```
numpy 1.26.4, linked against numpy/linalg/lapack_lite       (no external BLAS)
3000^3 float64 GEMM : 12.43 s = 4.3 GFLOPS
2000^3 with OPENBLAS_NUM_THREADS=1  : 5.1 GFLOPS
2000^3 with OPENBLAS_NUM_THREADS=32 : 4.7 GFLOPS
no openblas, atlas or mkl in ldconfig
```

Threading environment variables do nothing because there is no threaded library to
configure. Every numpy linear-algebra number in this project is single-core reference
BLAS.

**Deliberately not fixed.** Reinstalling numpy to get a bundled OpenBLAS would change the
interpreter that produced the bit-exact verified results. The speed of the analysis
scripts is not worth risking the reference.

**Correction this forces to an earlier claim.** The bit-exactness tax for the Q operator
was measured at 12x, lane-emulated against "plain BLAS matmul". That comparison was
against this unoptimized reference BLAS at roughly 4 GFLOPS. Against a threaded BLAS the
ratio would be far larger. The 12x is a lower bound on the tax, not an estimate of it.

### Step 5 — shared generating relation across channels

$W_{\text{gate}}$ is 33792 x 7168, so at least 26,624 rows are combinations of the others.
That is forced by the shape and is not a finding. The question is whether the rank is
*below* 7168, which would mean the projection could be computed in a smaller basis.

**Two independent instruments, because the first was slow enough to invite a shortcut.**

LU on square submatrices. If any 7168 x 7168 submatrix is nonsingular, the rank is full
and no spectrum is needed.

```
first 7168 rows    slogdet sign +1   log|det| 52378.8   solve residual 3.545e-11
last  7168 rows    slogdet sign +1   log|det| 52363.9   solve residual 1.611e-11
random 7168 rows   slogdet sign -1   log|det| 52369.7   solve residual 7.195e-11
```

Gram spectrum, the slower route, with a random int8 matrix of the same shape as control.

```
W_gate   rank 7168 of 7168    sv 2.773e+04 .. 1.867e+03    sv[-1]/sv[0] 6.731e-02
W_up     rank 7168 of 7168    sv 2.737e+04 .. 1.866e+03    sv[-1]/sv[0] 6.819e-02
W_sh1    rank 6144 of 6144    (Gram is 7168 wide, so 1024 zero eigenvalues are the shape)
random   rank 7168 of 7168    sv 1.975e+04 .. 7.307e+03    sv[-1]/sv[0] 3.699e-01
```

**Outcome: ruled out.** Full rank by both instruments. No group of channels shares an
exact generating relation beyond what the shape already forces.

One secondary fact worth keeping: $W_{\text{gate}}$ is markedly more anisotropic than
random, 0.067 against 0.370, so the learned matrix is far from isotropic. It is still
nowhere near singular, and anisotropy is not rank deficiency.

**A self-check that caught a repeat of the earlier tolerance error.** The W_sh1 line
reported "rank 6658 at float64 tolerance" for a matrix with 6144 rows. A rank above the
row count is impossible, so that number is the float64 tolerance counting Gram noise. The
float32 tolerance gives the correct 6144. The error was caught by an arithmetic
impossibility rather than by judgment, which is the only reason it did not survive.

### Where the Step-41 program stands

Steps 1 to 5 are complete. Step 6 is conditional, "derive a candidate reduced equation if
supported", and nothing supports one:

| step | question | result |
|---|---|---|
| 1 | capture the real pairs | done, verified against the engine |
| 2 | exact repetition | none, 0 duplicates in 248,832 pairs |
| 3 | rank structure | full; exact per-channel collapse ruled out |
| 4 | recurrence | none of four kinds, each with a control |
| 5 | shared generating relation | none; full rank by two instruments |

**This line closes out negative.** The SiTU information boundary of Step 35 stands: the
$(g,u)$ pairs carry independent information, and the document's own Step 39 says that in
that case the nonlinear channels "should not be artificially reduced".

The one structural regularity found along the way, the gate-up row alignment, is
approximate and therefore outside the criterion fixed at the top of this file.

### Step 14/18 — backward query propagation

The document proposes computing $o = S^\top q$ without ever forming $S$, by propagating
the query backward: $y_t = q_t$, $y_{i-1} = A_i^\top y_i$, $o = \sum_i b_i^\top y_i$.

Tested in two versions, because the transition matrix in the document is not the one the
code implements.

```
                            pos 0       pos 1       pos 2       pos 3       pos 4
document as written      3.7e-09     4.9e-03     1.2e-02     2.0e-02     1.9e-02
corrected                3.7e-09     1.1e-08     1.1e-08     1.5e-08     1.5e-08
```

**The document's form reaches 28% relative error.** That is a wrong formula, not a
rounding difference. Position 0 agrees for both versions because $S_0 = 0$ makes the
decay term unreachable there, which doubles as a check that the harness is right.

The corrected form, $A_i^\top = D_\alpha - \beta (D_\alpha k) k^\top$, sits at float32
epsilon throughout. The dual-space identity itself is sound; only the matrix was wrong.

**Scope of the error in `k3-equation-reduction.md`.** Step 13 defines
$A = D_\alpha - \beta kk^\top$ and Steps 14, 16, 18, 19, 20, 24, 25 and 26 build on it.
All of them inherit it. What survives unaffected is the structural claim: the true
transition $(I - \beta kk^\top)D_\alpha = D_\alpha - \beta k(D_\alpha k)^\top$ is still
diagonal plus rank one, so the composition algebra of Step 24 and the closure argument of
Step 25 stand. Only the specific formula changes.

**Cost.** Per head, per query, the backward form is 96x cheaper at position 0 and 19x at
position 4. But it answers exactly one query, where the forward state answers every
future one too, so they are not interchangeable unless the full history is retained at
$3 \times 128 \times t$ per head.

### Step 28 — the final norm scalar is irrelevant to the argmax

$x'_i = (w_i x_i)\rho$ with $\rho > 0$, and the lm_head is linear, so the scalar cannot
change the selected token. Tested by dropping it from the C implementation.

```
with scalar     top1 17374 (18.112919)  top2 384 (14.434750)  margin 3.678169
without scalar  top1 17374 (14.180687)  top2 384 (11.301032)  margin 2.879655

implied scalar  1.276985 .. 1.278524        (constant in exact arithmetic)
full ranking identical : 163475 / 163840    first difference at rank 4323
top 1 / 10 / 100 / 1000 ordering identical : yes
top 10000 ordering identical               : no
bit-identical logits                       : 0 / 163840
```

**Confirmed, with a boundary.** The emitted token is unchanged and the top 1,000 ordering
is preserved exactly. The *complete* ordering is not: 365 positions reorder from rank
4323 down, where logits are small and tightly spaced, and the implied scalar varies in
the fourth decimal for the same reason. So the invariance is exact over the reals and
holds for greedy decoding, but it is not safe for sampling or for top-k with large k.

**Value.** 21,504 operations, which is 0.0000040% of the forward pass.

### The routed experts — 45% of the pass

The tail and the aggregation are small denominators. The expert matmuls are not, so this
is where a reduction would have to land.

**Four structural questions, all negative.**

```
1. byte-identical experts, layer 1   : 896 distinct of 896, 0 duplicates
2. one expert's W1, 3072 x 3584      : slogdet nonzero, rank = 3072, full
3. two experts stacked, 6144 x 3584  : slogdet nonzero, they already span all of R^3584
4. expert 0 at layer 1 vs layer 2    : different
```

Question 3 was the one that mattered. If the 896 experts shared a low-dimensional row
space, the latent could be projected once and every expert would become cheap. Two
experts already span the whole space, so no such subspace exists.

**But the decode revealed something with a real denominator.**

```
zeros in decoded expert weights : 11.5327% (w1), 11.5373% (w3), 11.9909% (w2)
source                          : E2M1 code 0 or 8. dead scale bytes: 0 of 344,064
stability across 64 experts     : 11.4338% .. 11.5609%, std 0.0259%
```

Multiplying by zero and adding zero to a finite accumulator are exact no-ops, so this is
**5.19% of the whole forward pass in provably redundant arithmetic** — three orders of
magnitude larger than any previous exact candidate.

**It fails on structure.**

```
fully zero 16-element blocks : 0 of 688,128
fully zero 32-element groups : 0 of 344,064
fully zero rows              : 0 of 3,072
fully zero columns           : 0 of 3,584
zeros per row                : min 346, median 412, max 497, of width 3584

control: random placement at the same rate predicts 9.8e-16 zero 16-blocks,
i.e. 0.0 expected. observed 0.
```

Every row carries its share, no unit the kernel actually loads is all zero, and the
placement is statistically indistinguishable from random. At 88.5% density a sparse
format costs more in indexing than the 11.5% it saves.

**Outcome.** The largest exact redundancy found in this work, and still not a reduction.
It is worth recording that this is a *third* failure mode, distinct from the previous
two: not inexact, not too small, but structurally unexploitable.

### The denominator, measured instead of estimated

Every "worth X%" figure above had been computed from my own FLOP arithmetic. The C
implementation was instrumented to measure it instead.

```
where the time actually goes, 56.15 s total, 16 threads

   MoE routed experts                       42.14 s   75.05%
   attention, KDA and MLA                    8.35 s   14.87%
   MoE shared expert                         3.00 s    5.34%
   tail: aggregate, norm, lm_head            0.94 s    1.68%
   MoE latent down-projection                0.64 s    1.14%
   unaccounted: weight load, glue            0.58 s    1.02%
   MoE router                                0.31 s    0.56%
   dense MLP, layer 0 only                   0.16 s    0.29%
   snapshot aggregation + pre-MLP norm       0.02 s    0.04%
   pre-attention norm                        0.00 s    0.01%
```

**The estimate was wrong by a large factor.** I had put the experts at 45% and attention
at 38%. Measured, they are 75.05% and 14.87%.

The gap is the MXFP4 decode. A FLOP count sees one multiply-accumulate per weight, but
every expert weight must first be nibble-extracted, looked up in the E2M1 table and
multiplied by its group scale. That is roughly four operations before the one the FLOP
count counts, and it happens on every use with no reuse to amortize it: across five
positions the layer draws 80 experts of which 74 are distinct.

**Consequences for figures recorded above.**

- The expert zeros are worth **8.6%** of the pass, not the 5.19% recorded earlier.
  The finding is unchanged; only the denominator was wrong.
- The snapshot caching figure stands. Aggregation is 0.04% of measured time and the
  caching removes 81% of part of it, so it remains far below the noise.
- Attention being 14.87% rather than 38% makes the KDA reductions of Cycle A and
  Step 14/18 less valuable than they looked, not more.

**What this changes about method.** Six earlier entries in this file quote a percentage
of the forward pass derived from the same FLOP estimate. They are directionally right
and quantitatively unreliable. A FLOP model of a kernel that decodes its own weights
measures the wrong thing.

### Cycle D — trajectory geometry, and a real period-12 structure

**Direction, from the human:** the model normalizes repeatedly, so its states live on
spheres and the natural coordinates are angles. If there is circular structure, observation
should show it.

**Instrument.** No new run needed. `layer.out`, site 8, is tapped at all 93 layers, so the
trajectory is already on disk. Position 4, normalized directions.

**Not a circle.**

```
PCA rank of the 93 normalized directions : 92 of 93
first 2 components explain               : 22.25%
angle from the layer-0 direction         : rises to ~87 deg and stays, no return
residual norm                            : 0.46 to 133.41, ratio 289
```

A circle would be rank 2 with a constant turn angle. It is neither.

**But emphatically not random.** Against a matched random-direction control:

```
                          measured     random control
turn angle, mean          38.05 deg    90.03 deg
turn angle, std           15.88 deg     0.68 deg
first 2 components        22.25%        2.64%
first 10 components       61.97%       12.89%
```

**And genuinely periodic, at exactly the snapshot period.**

```
autocorrelation of the turn-angle sequence
   lag 12  +0.6562      lag 24  +0.5188      lag 36  +0.4042
   troughs between, around -0.20

turn entering a push layer, L mod 12 == 11 : n=7,  mean 82.66 deg
all other steps                            : n=85, mean 34.38 deg   ratio 2.40x

phase profile, mean turn by position in the cycle
   0: 49.3   1: 43.4   2: 37.3   3: 32.2   4: 31.6   5: 33.9
   6: 28.6   7: 29.7   8: 27.8   9: 30.4  10: 32.4  11: 82.7
```

**Mechanism, identified rather than inferred.** Stage 14 of `k3-stages.md`: at a push
layer `have_prefix` is cleared and the residual is *replaced* by the attention output
instead of added. That discontinuity is the 82.66 degree turn. The period is the
residual-replacement schedule made visible in the geometry, not a hidden invariant.

**Outcome.** A genuine cyclic structure in the depth trajectory, measured, controlled and
mechanistically explained. It is the first positive structural result in this exploration
that is neither approximate nor forced by the tensor shapes.

Whether it enables any reduction is a **separate and untested question**. A sawtooth in
turn angle says the trajectory reorients hard every twelve layers and settles in between.
It does not by itself say any computation can be skipped.

**Method note.** This came from a direction supplied by the human and from running the
measurement. Four prior cycles of reasoning about where structure ought to live produced
nothing; one change of observable produced this. The direction was not derivable from the
evidence already collected.

### Cycle E — the cycle is architectural, and carries no invariant

Two steps, run in order, with the second not conditional on the first.

**Step 1: is the cycle a property of the model or of position 4?**

All five prefill positions, plus the decode position, which comes from a separate forward
call with carried state.

```
source              mean    push   other   ratio   ac@12
prefill position 0  33.30   83.88  29.14   2.88x   0.6267
prefill position 1  37.36   86.39  33.32   2.59x   0.6214
prefill position 2  35.90   79.46  32.31   2.46x   0.6721
prefill position 3  37.78   84.41  33.94   2.49x   0.6762
prefill position 4  38.05   82.66  34.38   2.40x   0.6562
decode position     37.30   83.33  33.51   2.49x   0.6382
random control      90.03   90.24  90.01   1.00x   0.1612

push ratio : 2.40 to 2.88, std 0.157     ac@12 : 0.621 to 0.676, std 0.021
```

**The cycle belongs to the architecture.** Phase 11 lands between 79.5 and 86.4 degrees in
all six trajectories. Position 0 is the only one with a different phase shape, which is
explicable: with causal attention it has no context to attend to.

**Step 2: is anything conserved around the cycle?**

```
                                  mean      std       CV
total turn per cycle            437.59    79.86   0.1825
net turn per cycle               79.44     4.19   0.0527
norm ratio per cycle              4.22     4.67   1.1072
log norm ratio                    0.58     1.50   2.5986
consecutive snapshot angle       80.35     3.66   0.0456

[control] random consecutive      90.06     0.29   0.0032
[control] random total turn     1035.3    118.2   0.1142
```

**Nothing is conserved.** Every candidate is either more variable than the random control
or indistinguishable from it. Total turn is *more* variable than random, 0.183 against
0.114. Norm ratios run from 0.119 to 12.96 across the eight cycles.

**The control is what makes this readable, and it prevented a false positive.** In 7168
dimensions random unit vectors sit at 90 degrees with very small spread, so the random
consecutive angle has CV 0.0032, lower than anything measured in the model. Low variation
is the null hypothesis here, not the signal. Without that baseline the consecutive
snapshot angle at CV 0.0456 would have been reported as nearly conserved. It is six times
*less* constant than chance.

Secondary facts. The eight snapshots sit 74.6 to 87.1 degrees apart, slightly closer than
the 90 of random. Their first two principal components explain 36.63%, against about 29%
for eight random directions, so there is mild structure. A circle through eight points
would be rank 2 at 100%. Snapshot norms are 0.79, 3.89, 0.46, 0.58, 0.52, 6.75, 3.20,
7.36, with no visible pattern.

**Outcome.** The period-12 cycle is real, robust across positions and across a separate
forward call, and mechanistically explained. It carries no conserved quantity. A
repeating pattern without an invariant is not a closed form, and the invariant is exactly
the part that would have made one possible.

### Cycle F — a systematic invariant sweep, and the closest thing to a constant

Cycle E tested four quantities chosen by me, which is guessing. This sweeps a family and
lets a permutation test decide.

**Null, chosen so it cannot be gamed.** Shuffle the *layer order*. That preserves every
value the model produced and destroys only the cyclic arrangement. Statistic: mean
within-phase coefficient of variation, over 2,000 permutations.

```
functional              dim    observed  null mean       z   p(lower)
norm growth             yes      0.1955     0.3325  -15.35     0.0000
turn cosine             yes      0.1608     0.2379   -9.96     0.0000
cos to cycle snapshot   yes      0.4391     0.6388   -9.49     0.0000
std of components        no      0.5954     0.8904   -8.28     0.0000
norm                     no      0.5954     0.8898   -8.11     0.0000
||r|| / cycle snapshot  yes      0.6120     1.2685   -6.95     0.0000
participation ratio     yes      0.6172     0.6464   -2.83     0.0140
spectral entropy        yes      0.0511     0.0581   -2.64     0.0105
L1 / L2                 yes      0.0497     0.0553   -2.43     0.0190
max|r| / ||r||          yes      0.4814     0.5506   -2.23     0.0210
kurtosis                yes      1.2885     1.4498   -1.25     0.1050
frac |r| > mean|r|      yes      0.0194     0.0203   -0.95     0.1115
cos to layer 0          yes      3.9378     4.4972   -0.03     0.8200
```

**Eleven functionals are phase-locked beyond chance**, far more than Cycle E's four
candidates suggested. But phase-locked is not constant: norm growth still varies by 20%
at fixed phase.

**The candidate for a numeric constant.** Three functionals have small variation and only
weak phase-locking, meaning near-constancy across all layers rather than within a phase.
The strongest is $\|r\|_1/\|r\|_2$, whose Gaussian closed form contains pi:

$$\frac{\|x\|_1}{\|x\|_2} \to \sqrt{\frac{2n}{\pi}} = 67.5521 \quad \text{for } n = 7168$$

```
                       K3, 93 layers        Gaussian sample, same shape
L1 / L2                64.9487 +- 5.8277    67.5665 +- 0.2244
frac |x| > E|x|         0.4195 +- 0.0140     0.4252 +- 0.0031
participation ratio     1173.0 +- 771.5      2389.4 +- 47.5
kurtosis                  46.79 +- 156.84      3.0011 +- 0.0600

maximum L1/L2 per trajectory : 67.64 67.48 67.59 67.65 67.87 67.82
```

Every trajectory's maximum lands on $\sqrt{2n/\pi}$ to within 0.5%.

**But it is not a constant.** The per-layer profile:

```
within 0.5% of sqrt(2n/pi) : 14 of 93
within 2%                  : 53 of 93
within 5%                  : 76 of 93
median 66.5619, which is 1.47% below
range 16.19 to 67.87
```

Forty of 93 layers depart by more than 2%, the final cycle drifts monotonically down to
49.34, and L12 collapses to 16.19. Substituting a constant would introduce errors up to
76%. The excursions are not explained by the norm, correlation -0.077, and push layers
are not specially implicated: only 3 of the 8 appear among the 17 largest departures.

**What the near-match actually means.** $\|x\|_1/\|x\|_2 \approx \sqrt{2n/\pi}$ holds for
any vector whose components look Gaussian. Finding it here says the residual's component
distribution is roughly Gaussian at most layers. That is a real measured property, and it
is a property of high-dimensional activations generally, not something specific to K3 and
not an invariant of the computation. The pi is Gaussian bookkeeping, not structure.

**A circular statistic I produced and then had to retract.** The same run reported "the 53
layers within 2% have CV 0.00533". That conditions on closeness and then measures
closeness. It is a tautology and would have read as a strong invariant. Selecting a subset
by the property you then measure is a way of guaranteeing the answer.

**Outcome.** No constant. The sweep found real phase structure, the strongest candidate
for a numeric invariant sits near a pi-containing Gaussian value, and it varies far too
much to be replaced by one.

### Cycle G — the binding constraint is not arithmetic

**Why this was asked.** Six cycles of candidates all failed, and six of the seven failed
on something that is not mathematics: float32 reassociation, denominator size, memory
layout. That pattern raises a prior question. If the time is not spent on arithmetic,
then removing arithmetic cannot help, and the whole program was aimed at the wrong
quantity.

**First measurement, and it corrects an earlier caveat.** Disk reads were instrumented
with `read_bytes` from `/proc/self/io`, which counts mmap page faults reaching the device
where `rchar` counts only `read()` syscalls.

```
rchar        0.00 GB
read_bytes 101.72 GB      on a run I had been calling "warm page cache"
```

There is no warm cache. The expert pool is 1.45 TB against 124 GB of RAM, so every run
fetches roughly 100 GB from the device. The "warm page cache" qualifier attached to the
earlier 50.70 s comparison against the engine was wrong.

**Thread scaling, the decisive test.**

```
threads   total     experts   read_bytes   effective rate
   16     53.74 s   40.28 s   144.72 GB    2.69 GB/s
    8     56.84 s   44.41 s   157.34 GB    2.77 GB/s
    4     75.08 s   62.16 s   118.97 GB    1.58 GB/s
```

Halving the threads from 16 to 8 costs **10%**. Compute-bound work would take twice as
long. The threads are waiting on the device, not computing. Only below 8 threads does
compute begin to bind, 1.40x from 8 to 4.

The arithmetic closes it: 144.72 GB at 2.69 GB/s is 53.8 s, which is the entire runtime.
Device transfer alone accounts for all of it.

`read_bytes` varies between 119 and 157 GB across runs of an identical computation. That
is page-cache retention varying run to run, and it is measurement noise, not a difference
in work done.

**Outcome. The model as implemented is I/O bound, not compute bound.**

**What this does to the preceding six cycles.** It does not invalidate any measurement,
but it re-prices all of them. Every candidate was scored as a fraction of *operations*.
Operations are not the binding resource. A reduction that removed 100% of the expert
multiply-accumulates would still have to fetch the same 145 GB and would save close to
nothing.

**It also finishes the expert-zeros result.** Cycle "experts" recorded 8.6% of the pass as
multiplication by exact zero, unexploitable because the zeros are scattered across SIMD
blocks. The I/O measurement makes that stronger and simpler: the zeros are 4-bit codes
inside packed bytes that must be fetched regardless. They are unavoidable at the device,
not merely awkward in the register file. No arrangement of the kernel could recover them.

**The question this leaves.** The reduction program asked how to make computation
disappear. On this machine the computation is largely free and the bytes are expensive.
The corresponding question for bytes would be a different investigation with a different
success criterion, and it has not been started here.

**Caveat, stated because it bounds the claim.** This is measured on one machine, with one
implementation that has no expert cache. The production engine holds a 30 GB expert cache
and reported a 98.5% hit rate, yet still read 99.72 GB for the same prompt, so it appears
to sit in the same regime. That is an inference from its log, not a measurement I made.

---

## What none of this covers

- One prompt, five tokens, `The capital of France is`. Every number above is n=1.
- Cycle C's margins are sensitive enough that a different prompt could reorder the table.
- Steps 1 and 2 sampled layer 0 and layer 1. The other 91 layers are not captured.
- The routed-expert capture is one position, because the trace taps only the last one.
