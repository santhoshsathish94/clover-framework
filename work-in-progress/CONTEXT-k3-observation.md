# CONTEXT — observing K3

Read this first. Written during the work, not after it.

## Goal

Understand how Kimi K3 actually behaves by watching real tokens move through it, across
several kinds of input. Not to reach a particular result — the reality decides what the
result is. Context stage. **No Direction has been set, and none should be inferred from
this file.**

## Why there is no Direction yet

An earlier attempt added a `--direction` boundary to the engine that refused to generate
without a human-owned Direction file. That was Direction work done before Context work,
and it was wrong in two ways: it presumed the Direction before the system had been
observed, and it broke the chat option contract test by refusing before argument
validation. **Removed completely** — the branch is now upstream `a2ad8e5` plus the trace
instrumentation only (`clover/observe`, commit `f97eaaa`). With it gone the full suite
reports ALL WEIGHTLESS TESTS PASSED.

## The instrument

`K3_TRACE=<path>` makes every stage of the forward pass append one JSON record naming
the tensor that passed through it: element count, L2, min, max, mean, count of
non-finite values, and an FNV1a-64 of the raw bytes. Statistics show behavior; the hash
establishes identity, so "the tensor leaving A is the one entering B" is checkable.
Records are stamped with generation step, position and layer.

Taps: `forward.T`, `forward.cached`, `embed`; per layer `layer.in`, `stack.depth`,
`attn_res.pre_attn`, `snapshot.pushed`, `norm.pre_attn`, `attn.is_kda`, `attn.out`,
`resid.post_attn`, `attn_res.pre_mlp`, `norm.pre_mlp`, `ffn.out`, `layer.out`; inside the
router `router.pick`; then `final.aggregate`, `final.norm`, `logits`, `argmax`.

Verified not to disturb the arithmetic: with tracing compiled in, the 13-layer fixture
still reports ENGINE MATCHES THE REFERENCE EXACTLY on all four gates.

## Held constant across every variation

So that differences between runs mean something:

```
machine   AX102-3, 37.27.114.74
branch    clover/observe @ f97eaaa
model     /root/k3model     trunk /root/k3trunk_i8 (int8)
flags     --incremental --trunk-gb 60 --cache-gb 30
```

Each run is a fresh process, so no expert cache state carries between variations.

## The variations

One at a time. Context updated after each before the next starts.

| id | what it probes | prompt | gen | state |
|---|---|---|---|---|
| v1 | baseline | "The capital of France is" | 1 | **done** — `" Paris"`, 32.1 s |
| v2 | decode vs prefill | same as v1 | 8 | **done** — 89.1 s, cache warms |
| v3 | repetition | a repeated token | 4 | **done** — routing collapses |
| v4 | another domain | code | 4 | **done** — massive activation at L91/92 |
| v5 | same fact, other language | French form of v1 | 4 | **done** — 46% expert overlap with EN |
| v6 | prefill at scale | long passage | 4 | **done** — 831 GB prefill, diversity 14.3% |
| v7 | out of distribution | nonsense ids | 4 | **done** — 2-cycle, confidence halves |
| v8 | clean register control | short EN prose, other topic | 4 | **done** — overturns v6's reading |

Per-variation findings go in `k3-flow/<id>-<name>.md`.

## The instrument agrees with the engine

Worth recording once: for v1 the trace yields 5683 distinct (layer, expert) pairs and
the engine's own cache statistics independently report `requests 5683`. Two unrelated
mechanisms counting the same thing and agreeing is the strongest available evidence that
the taps measure what they claim to.

## Established (observed, on the real checkpoint)

From the first traced run — "The capital of France is", 5 positions, 1 token, int8
trunk, 93 layers, 39.1 s, 54.47 GB trunk + 99.72 GB experts read. **Emitted token 17374
= `" Paris"`.**

- The layer chain is bit-identical end to end: `layer.out[L]` = `layer.in[L+1]`, 92/92.
- Snapshots fire at layers 0, 12, 24, 36, 48, 60, 72, 84 — spacing exactly 12, eight of
  them. At each, the residual is **replaced** by the attention output, not added to.
- The residual stream is therefore a **sawtooth**: L11 274.96 → L12 12.34; L23 356.22 →
  L24 1.38; L83 138.83 → L84 17.10. Mean L2 at snapshot layers 7.14 against 82.56
  overall, maximum 482.16.
- Trained norm gains do real work: outputs 15.2–126.6 and 1.7–46.5 where √n is 189.3.
- MLA output is *larger* than KDA on real weights: mean L2 19.49 (24 layers) vs 15.55
  (69 layers).
- Routing renormalizes to exactly 1.00000; one expert's share ranges 0.003–0.645.
- Routing weights are never descending, in 460 of 460 decisions, because selection uses
  the biased score and the stored weight is the unbiased one. The bias pushes the
  top-weighted expert out of first place in 20% of decisions.
- Zero non-finite values at any stage, 837 tensors.

From v2 — same prompt, eight tokens:

- **Prefill and decode are the same code but different regimes.** Prefill costs 30.4 s
  and 99.72 GB of expert reads; decode settles at 7.1–7.7 s and 12–18 GB.
- **Mean adjacent-step expert reuse is 42.4%** (range 23.4–54.1%). That is the figure
  that predicts read volume. Only 54 of 9991 (layer, expert) pairs — 0.5% — are used by
  every step, so the all-step intersection says nothing useful about the cache.
- Each token needs ~1472 experts; the cache holds 1708 slots, about one token's worth.
- Decode positions carry a smaller residual than prefill ones: 23.9–28.5 per position
  against 36.92.
- The sawtooth, the snapshot layers, and the replace-not-add rule are identical in both
  regimes.

From v3 — twelve copies of one token:

- **Repetitive input routes narrowly.** In-layer diversity 37.0% against v1's 77.2%;
  17664 expert slots need only 6541 distinct (layer, expert) pairs. Positions do not
  route independently.
- Mean adjacent-step reuse 65.5% against v2's 42.4%; pairs used by every step 9.3%
  against 0.5%. Reads settle at 11.3 GB per step against v2's 12.3 GB.
- So repetitive input is cheap **because it routes narrowly**, not because the engine
  treats it specially.
- The model continues the repetition exactly: every generated token is ` the`.

From v4 — Python source:

- **The two consecutive MLA layers at the end (91, 92) can amplify enormously.** On the
  code prefill, layer 91's attention output is 19.1× layer 90's residual; every other run
  and step measured sits at 0.1–0.3×, including v4's own decode steps.
- The *input* to layer 91 is ordinary (L2 129.9, max 4.37). The output's maximum element
  is 3457.8 against 1.34 on the factual prompt — **2585×** — with the mean still ~0. A few
  extreme elements, not a shifted distribution.
- The residual reaches 20494 and the model is unaffected, because RMSNorm is
  scale-invariant: logits come out at max 23.5, in line with every other run, and the
  continuation is correct.
- Activations are fp32 so this is harmless here. It is the documented massive-activation
  behavior, observed directly.
- Prefill read 210 GB of experts, more than double v1's, because 18 positions route more
  widely. In-layer diversity 45.2%; mean adjacent-step reuse 48.0%.

From v5 — the same fact in French:

- **The French prompt produced the identical first four tokens as the English one**
  (17374, 20829, 10, 427), despite sharing essentially no input tokens.
- Prefill expert-set Jaccard: EN vs FR **46.0%**, against 13.9% with code and 20.8% with
  repeated text. On the shared decode step, 45.5% (920 of 1472 pairs).
- No final-layer amplification (0.14×), so v4's spike is not a long-prompt effect: v5 has
  7 positions, v1 has 5, v4 has 18, and only v4 spikes.

### Prediction on the record, before v6 runs

v1 and v5 are both prose; v4 is code and v3 is degenerate. The 46% may be measuring
register rather than meaning. v6 is long English prose on an unrelated topic and settles
it: **near 46% means the result is about register; near 20% means it is about meaning.**

### How that prediction turned out: the test was invalid

Observed Jaccard was **8.5%**, below both thresholds — because Jaccard is not robust to
an 8× difference in set size. v6 holds 47,362 (layer, expert) pairs against v1's 5,683,
so even total containment of v1 in v6 could not exceed 12.0%. The measure could not have
returned 46% whatever the truth was. A badly designed test, not a surprising result.

Redone as **lift over chance** (space = 92 MoE layers × 896 = 82,432 pairs):

| overlap with v1 | coverage | observed | chance | lift |
|---|---:|---:|---:|---:|
| v5 French, same fact | 7.4% | 65.1% | 7.4% | **8.85×** |
| v3 repetition | 7.9% | 37.0% | 7.9% | 4.66× |
| v4 code | 14.5% | 38.0% | 14.5% | 2.61× |
| v6 English prose, other topic | 57.5% | 73.3% | 57.5% | **1.28×** |

**Superseded by v8 — see below.** v6 covers 57.5% of the whole space, so its lift is
compressed by saturation and it cannot carry this argument. A size-matched control was
needed, and it reverses the answer.

### v8, the size-matched control: register dominates, meaning adds a smaller increment

"The chemical symbol for iron is" — 6,009 pairs against v5's 6,062, so lift is directly
comparable between them for the first time.

| overlap with v1 | set size | observed | **lift** |
|---|---:|---:|---:|
| v5 French, **same fact** | 6,062 | 65.1% | **8.85×** |
| v8 English, **other fact, same register** | 6,009 | 50.6% | **6.94×** |
| v3 repetition | 6,541 | 37.0% | 4.66× |
| v7 nonsense ids | 10,626 | 38.7% | 3.00× |
| v4 code | 11,971 | 38.0% | 2.61× |
| v6 long prose (saturated) | 47,362 | 73.3% | 1.28× |

Short English factual prose shares v1's experts at 6.94× chance **whatever it is about**.
Making it the same fact in another language adds a further 1.28× — 824 more shared pairs
out of 5,683, which is far above noise but a minority of the enrichment. Everything that
is not short natural prose sits at 2.6–4.7×, clearly separated from both prose runs.

**Four measures were tried before one could answer the question**: Jaccard (not
size-robust), containment (rewards larger comparison sets), lift against a saturated
comparison set (valid statistic, unsound comparison), and finally lift against a
size-matched control. The fourth reversed the answer the third gave.


From v6 — 225 positions:

- Prefill read **831 GB** of experts and took 443 s; decode steps were normal (9.5–11 s).
- **In-layer diversity falls as positions rise**: 77.2% at 5, 58.8% at 7, 45.2% at 18,
  **14.3% at 225**. Long context saturates, which is what the expert cache lives on.
- Largest single expert share 0.8515 — one expert taking 85% of a token's routed
  contribution.
- First adjacent-step reuse 96.5%: the token after a long prefill reuses almost
  everything it touched.

### Correction to v4

v4's file first read the L91/92 amplification as code-specific. **v6 also spikes**, at
7.30× with max element 3319.9. Across five prefills the spike appears at T = 18 and
T = 225 but not at T ≤ 12, and the max element lands near 3300–3500 in both spiking runs.
Content and length vary together across these runs, so neither is isolated. The v4 file
has been corrected.

From v7 — twelve out-of-distribution token ids:

- The model produces a **2-cycle**, ` ` and `的` alternating. Degenerate, not text.
- **Confidence collapses and is directly visible**: peak logit 8.385, against 18.1–23.5
  for every run on real text; logits L2 812 against 1159–1398.
- **The cleanest diversity comparison available**, because v3 and v7 both have exactly 12
  positions: repetition 37.0% in-layer diversity and 115 GB read, nonsense 60.2% and
  186 GB. Routing *breadth*, not sequence length, is what expert reads track.
- Routing mechanics are untouched — weights still sum to 1, largest share 0.56, top weight
  not first 22.7%. Only the distribution moves.
- Lift over chance vs v1: 3.00×.

## What holds across all eight variations

Eight runs, 24 forward passes, every input kind tried. These never varied:

- Snapshot layers **0, 12, 24, 36, 48, 60, 72, 84** — spacing exactly 12, in every run
  and every step.
- At a snapshot layer the residual is **replaced** by the attention output; everywhere
  else it is **added to**. No exception in 2,000+ layer observations.
- `layer.out[L]` is **bit-identical** to `layer.in[L+1]`, always.
- Routed experts within a decision are always distinct; weights always sum to exactly
  1.00000.
- The routing bias pushes the top-weighted expert out of first place in **20–28%** of
  decisions, in every run — stable across prose, code, repetition and nonsense.
- No non-finite value ever appeared, including in the run whose residual reached 20,494.

### What varied, and what drives it

| | v1 EN | v8 EN | v5 FR | v3 repeat | v7 nonsense | v4 code | v6 prose |
|---|---:|---:|---:|---:|---:|---:|---:|
| positions | 5 | 6 | 7 | 12 | 12 | 18 | 225 |
| in-layer diversity | 77.2% | — | 58.8% | **37.0%** | **60.2%** | 45.2% | **14.3%** |
| prefill expert GB | 99.7 | 105.4 | 106.4 | **114.8** | **186.5** | 210.1 | 831.1 |
| max logit | 18.1 | — | 18.7 | 19.0 | **8.4** | 23.5 | 18.1 |
| L91 amplification | 0.14× | — | 0.14× | 0.35× | 0.11× | **19.1×** | **7.30×** |
| lift vs v1 | — | 6.94× | 8.85× | 4.66× | 3.00× | 2.61× | 1.28× |

- **Routing breadth, not sequence length, drives the expert reads.** v3 and v7 both have
  12 positions; repetition costs 115 GB and nonsense 186 GB.
- **More positions means more agreement between them**, monotonically: 77.2% at 5 down to
  14.3% at 225. This saturation is what makes long context affordable.
- **Confidence is legible in the output layer.** Real text gives a peak logit of 18.1–23.5;
  nonsense gives 8.4.
- **Decode steps are remarkably uniform** across every input kind: 7–11 s and 12–26 GB,
  whatever the prefill did.

## The next cycle

The one experiment this set owes: **the same text truncated to 5, 12, 18 and 40
positions**, so length moves and content does not. Across these eight runs length and
content vary together, so the L91/92 amplification threshold between 12 and 18 positions
is suggestive and not established.

Also still open: a per-position tap, to say *which* position carries the extreme
activations.

## Ruled out

- **"RMSNorm pins the L2 to √n."** True of the random-init fixture, false of the real
  checkpoint. It was measuring the fixture.
- **"MLA contributes ~12× less than KDA."** Same — a fixture artifact; reversed on real
  weights.
- **Tapping routing at the call site in the layer.** The batched prefill path never
  writes the caller's `idx`/`wt`, so the tap reported sixteen zeros. Reading it as real
  would have produced the claim "K3's routing weights do not sum to 1". The tap belongs
  inside `k3_router`.
- **"In-layer expert diversity" as a cross-variation measure.** It is degenerate at
  T = 1, where it reads a meaningless 100%. Only v1 and v6 can be read for it.
- **Counting distinct expert ids across layers.** Each layer has its own pool of 896, so
  the identity is (layer, expert), not the bare id.
- **Raw L2 as a cross-run comparison.** It scales with the number of positions in the
  buffer; it has to be normalized per position before a prefill and a decode step can be
  put side by side.
- **Jaccard as a cross-run expert-overlap measure.** Not robust to set-size differences,
  which here run to 8×. Use lift over chance.

## Open

- KDA's nine internal steps and MLA's cache are still black boxes; only what enters and
  leaves them is observed.
- **Which position carries v4's extreme activations.** The taps hash the whole
  `T × hidden` buffer, so "one token dominates" is a hypothesis, not a finding.
  Confirming it needs a per-position tap.
- Whether the L91/92 amplification is triggered by code specifically, by prompt length,
  or by particular characters. v4 is one prompt.
- **Length against content for the amplification.** The one experiment this set owes:
  the same text truncated to 5, 12, 18 and 40 positions, so length moves and content does
  not.
- Whether routing diversity depends on depth. The first run showed ~75–79% distinct and
  flat, but over only five positions; that is a different quantity from the cache hit
  rate measured earlier over a long run. v6 is the one that can speak to this.
- Whether the sawtooth amplitude depends on input kind.

## Findings not raised to anyone

- Two block comments in `src/core/k3_ops.c` (lines 254, 1101) describe a four-accumulator
  multiply-then-add reduction; the code uses sixteen accumulators and explicit fused
  products. Not a correctness bug — all paths agree — but the comments invite a change
  that would silently break bit-identity with the fixtures.
- The MXFP4 trunk will not bind on upstream `a2ad8e5`:
  `self_attention_res_proj.weight has 1904 elements, engine expects 7168`, and
  1904 × 2 = exactly 7168 × 0.53125. The packer quantized a vector that must stay full
  precision. Our MXFP4 measurements came from the archived `mxfp4-trunk` branch, which
  carries the matching binder.
