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

---

## What none of this covers

- One prompt, five tokens, `The capital of France is`. Every number above is n=1.
- Cycle C's margins are sensitive enough that a different prompt could reorder the table.
- Steps 1 and 2 sampled layer 0 and layer 1. The other 91 layers are not captured.
- The routed-expert capture is one position, because the trace taps only the last one.
