# cycle.md — the cycles observed inside one Clover-K3 inference

This records what the engine actually did, observed on AX102 on 2026-10-03. Nothing
here is inferred from documentation or from how such engines usually work. Where a
thing could not be observed, it says so.

## How it was observed

Three instruments, all read-only. None of them changed the engine.

1. **`bin/profile-stages`** — the engine's own stage profiler, built from
   `profile-stages.c`, which compiles `clover-one.c` with `CLOVER_PROFILE_STAGES`.
   It emitted 13,560 `STAGE_JSON` rows across 3 positions and asserted its own
   correctness: `PASS: profiled all 93 layers from token IDs, prior output
   418/276 unchanged`. Raw output kept at `cycle-stages-20261003.jsonl`.
2. **An observation harness** at `/tmp/observe-cycle.c` on the host, outside the
   repository. It implements the same three `stage-profile.h` callbacks and, at
   every stage boundary, reports the engine's live state: `nsnap`, `have_prefix`,
   `NPOS`, cumulative expert projections, the L2 norm of the residual, and the
   attention cache length. 4,182 rows. It reproduced 418 and 276.
3. **A routing harness** at `/tmp/observe-route.c`, using the `CLOVER_CHECK_EXPERT`
   hook the engine already provides for test harnesses, recording which experts
   each layer selected. 276 rows. It also reproduced 418 and 276.
4. **A multi-prompt harness** at `/tmp/observe-multi.c`, combining both hooks and
   able to feed its own output back as the next input. Nine runs, **61 positions,
   5,612 routing rows**, recording for every position the snapshot layers, the
   layers where the residual was replaced, the residual norm at all 93 layers, and
   the 16 experts chosen at each of the 92 routed layers.

The nine runs, chosen to vary length and kind, with one holding the input token
constant and one repeating an earlier run in a separate process. Every prompt was
round-tripped through the tokenizer (`decode(encode(text)) == text`) before use.

| run | kind | prompt | tokens | positions |
|---|---|---|---|---|
| A | prose | `Rain falls` | 2 | 4 (2 generated) |
| B | prose | `The capital of France is` | 5 | 6 (1 generated) |
| C | prose | `In 1969, humans first walked on` | 9 | 9 |
| D | prose | ` the the the the` — token 276 four times | 4 | 4 |
| E | prose | `Water boils at` | 3 | 4 (1 generated) |
| Arepeat | prose | `Rain falls` again, separate process | 2 | 4 (2 generated) |
| Hsql | code | `SELECT name, email FROM users WHERE` | 7 | 7 |
| Kjson | code | `{"name": "Ada", "role":` | 9 | 9 |
| Jnum | numeric | `1, 2, 3, 4, 5,` | 14 | 14 |

Prompt throughout: token ids 91019, 25528 — `Rain falls`. Position 0 and 1 consume
the prompt, position 2 consumes the engine's own first output, 418.

One boundary was found and respected rather than worked around. `result_options()`
in `clover-one.c` refuses to run if `K3_LSTAT`, `K3_DUMPSEL`, `K3_DUMPROUTE`,
`K3_PROV`, `K3_DUMPRES` or twelve other dump variables are set — it calls
`result_miss("unsupported persistence or raw-reader option")`. Those dumps were not
enabled. The harnesses observe through the sanctioned compile-time hooks instead.

## The shape that was found

Observed, by counting the stages the profiler recorded:

| | observed |
|---|---|
| layers per position | 93, numbered 0 to 92 |
| dense MLP layers | 1 — layer 0 only (`MGATE`, `MUP`, `MDOWN`) |
| mixture-of-experts layers | 92 — layers 1 to 92 |
| KDA attention layers | 69 — those carrying `Q K V B FA FB` |
| MLA attention layers | 24 — those carrying `QA QB KA KB` |
| stages inside one MoE layer | 15, in a fixed order |
| experts selected per layer | 16, out of 896 scored |
| projections per expert | 3 |
| expert projections per layer | 48 |
| expert projections per position | 4,416 |

Observed stage order inside a layer, identical at layers 90, 91 and 92:

```
binding-and-stored-folds
pre-attention-aggregation
snapshot-push
pre-attention-normalization
attention
attention-residual
pre-mlp-aggregation-and-normalization
moe-scratch-and-parameter-bind
router-and-top16
prefetch-submit-and-latent-projection
result-lookup-and-first-expert-submit
shared-expert-during-read
experts-mix-normalize-up
mlp-merge-and-cleanup
mlp-residual-cache-save-and-unbind
```

After the last layer, only when a token is to be produced:

```
final-aggregation-and-normalization -> head-fill-and-projection -> argmax-and-cleanup
```

Position 0 did not run those three. Positions 1 and 2 did. The engine only projects
to the vocabulary at the position being asked for.

---

## Cycle 1 — the projection

**What repeats.** One weight block applied to one vector: read the block, multiply,
accumulate, write one output vector. Every named weight stage is one of these —
`Q`, `K`, `V`, `B`, `FA`, `FB`, `G`, `O`, `QA`, `QB`, `KA`, `KB`, `MGATE`, `MUP`,
`MDOWN`, `EDOWN`, `EUP`, `SH1`, `SH2`, `SH3`.

**What carries in.** One input vector and one fixed weight block.
**What carries out.** One output vector. The weights are unchanged.

**Observed.** The profiler attributes time to each weight by name. At position 1
the two projection operators account for almost all the work: `op:X mxfp4 expert
proj` 4,883.1 ms and `op:Q int8 projection` 1,608.9 ms, against 58.2 ms for the
router and 3.6 ms for aggregation.

This is the only cycle in the engine that touches a stored weight. Every larger
cycle below is a pattern of these.

---

## Cycle 2 — the expert

**What repeats.** Four stages, which the profiler names directly:
`detail:expert-gate`, `detail:expert-up`, `detail:expert-activation`,
`detail:expert-down`. Two projections from the 3,584-wide latent into 3,072, a
gating combination of the two, then one projection back down to 3,584.

**What carries in.** The layer's latent vector and one expert's weights.
**What carries out.** One 3,584-wide vector. Nothing is written back to the expert.

**Observed.** Exactly 3 projections per expert, 48 per layer, 4,416 per position —
and the engine checks this itself: `resident_expert_calls + resident_result_hits * 3
!= 92 * 16 * 3 * NPOS` is a fatal error. The three detail stages cost 1,659.2,
1,574.8 and 1,632.4 ms at position 1, close to equal, which matches three
projections of similar size.

---

## Cycle 3 — the routing decision

**What repeats.** Score all 896 experts against the layer's post-norm vector, add a
per-expert bias, keep the highest 16, then normalise those 16 scores so they sum to
one. Stage `router-and-top16`.

**What carries in.** The vector produced by this layer's `pre-mlp-aggregation-and-normalization`.
**What carries out.** 16 expert indices and 16 weights. No weight is modified.

**Observed, across 31 positions and 5 prompts. This is the richest result of the
exercise, and the single-prompt version of it was too narrow.**

*It is deterministic.* Run A and Arepeat are the same prompt in two separate
processes. At all four positions, **92 of 92 layers chose a byte-identical list of
16 experts**. Nothing about the selection is random or run-dependent.

*The input token barely matters; the context does.* Run D feeds token 276 four
times. The input is identical at every position, yet **no layer, at any pair of
positions, chose the same 16 experts** — 0 of 92 every time. Shared experts per
layer went 2.86 (pos0 vs 1), 4.52 (1 vs 2), 9.92 (2 vs 3), and only 1.24 between
positions 0 and 3. The same token, read with more history behind it, is routed
somewhere else.

*With no history, different prompts route almost alike — and the kind of text
barely matters.* At position 0 there is no attention history. Across all 28
cross-run pairs the mean overlap was **11.76 of 16**, and SQL, JSON and a list of
numbers sat alongside English prose:

| pair kind | pairs | mean experts shared per layer |
|---|---|---|
| prose vs prose | 14 | 12.33 |
| numeric vs prose | 6 | 11.95 |
| code vs code | 1 | 11.28 |
| code vs numeric | 2 | 11.24 |
| code vs prose | 12 | 11.16 |

Code routes slightly further from prose than prose does from prose, but the gap
(11.16 against 12.33) is small next to the 11.76 they all share.

*The divergence happens in two positions, then stops.* Mean overlap between
different runs, by position index:

| position | pairs | mean | range |
|---|---|---|---|
| 0 | 28 | 11.76 | 10.98 – 13.09 |
| 1 | 28 | 4.80 | 3.16 – 7.25 |
| 2 | 28 | 1.32 | 0.63 – 3.91 |
| 3 | 28 | 1.16 | 0.60 – 2.73 |
| 4 | 10 | 1.29 | 0.84 – 1.76 |
| 5 | 10 | 1.31 | 0.63 – 2.42 |
| 6 | 6 | 1.31 | 0.36 – 2.87 |

It does not keep falling. It drops from 11.76 to 1.32 over two positions and then
flattens at about 1.3 of 16 — close to what 16 draws from 896 would share by
chance, though that comparison was not computed against the actual gate
distribution. Different sequences separate almost immediately and then stay about
equally far apart.

*Within one run, overlap grows as context accumulates — for every kind of text.*
Consecutive positions:

```
A  [prose]   1.78 4.99 8.87
B  [prose]   2.04 3.68 4.30 5.71 6.03
C  [prose]   2.79 3.55 5.76 5.97 2.77 7.00 5.83 7.64
D  [prose]   2.86 4.52 9.92
E  [prose]   1.68 5.15 7.43
Hsql [code]  1.98 5.82 6.25 5.80 6.35 6.54
Kjson[code]  2.79 5.20 8.11 3.65 5.14 7.42 6.13 8.64
Jnum [num]   2.72 5.39 6.72 8.78 7.75 6.54 8.23 5.67 4.32 6.92 7.29 6.08 7.05
```

The trend rises in every run but is not monotonic: C drops from 5.97 to 2.77
between positions 4 and 5, Kjson from 8.11 to 3.65, Jnum from 8.23 to 4.32. Those
drops were recorded, not explained.

Taken together: the selection is a deterministic function of the vector arriving at
the layer, that vector is dominated by accumulated history rather than by the
current token, two sequences that start apart separate within two positions, and
consecutive positions of the same sequence grow more alike as they share more
history.

---

## Cycle 4 — the expert mixture

**What repeats.** For each of the 16 selected experts: look for a cached result,
run Cycle 2 if absent, then multiply by the expert's normalised weight and
accumulate. Then normalise the mixture and project it back to the 7,168-wide
model width. Stages `prefetch-submit-and-latent-projection`,
`result-lookup-and-first-expert-submit`, `experts-mix-normalize-up`.

**What carries in.** 16 indices, 16 weights, one latent vector.
**What carries out.** One 7,168-wide vector, added to the shared-expert output.

**Observed.** `experts-mix-normalize-up` is the single most expensive stage of a
position: 4,983.5 ms against 6,696.8 ms for all 93 layers summed, and 7,018.5 ms
for the whole position — **about 71% of a position**. The expert projection
operator alone, `op:X`, was 4,883.1 ms, so nearly all of that stage is the 1,472
expert projections and almost nothing else.

*An earlier draft of this file reported 36.9% here. That was wrong.* It divided by
the sum of every boundary stage, which double-counts the `detail:` stages nested
inside them. The correct denominator is the sum of `total:layer`, or the position
total. Corrected on the same data, no re-run needed.

Inside this cycle sits a read-ahead cycle — submit the next expert's read, compute
the current one, release. It is visible in the stage order: the shared expert is
deliberately computed at `shared-expert-during-read` to overlap a read. **It does
not currently fire.** The engine only stages reads when
`resident_pipeline.initialized && !resident_roots[L]->direct`, and with
`experts.direct` mapped the second condition is false. A live request reports
`cross_layer_submitted` 0, hits 0, misses 0.

What it costs now that there is nothing to overlap, measured at position 1 against
6,696.8 ms of layer time:

| stage | ms | share |
|---|---|---|
| `detail:cross-layer-predict-and-submit` | 0.0 | 0.00% |
| `detail:read-ahead-wait` | 0.0 | 0.00% |
| `result-lookup-and-first-expert-submit` | 1.6 | 0.02% |
| `shared-expert-during-read` | 351.1 | 5.24% |

The first three are free because they return immediately. The fourth is not
overhead at all — it is the shared expert itself, `SH1`+`SH2`+`SH3` = 349.8 ms,
which still has to be computed. It is only its *placement* that was chosen to hide
a read that no longer happens.

---

## Cycle 5 — the attention step

**What repeats.** Project the normalised input into queries and keys, write this
position's keys and values into the layer's cache, then score the query against
every cached position from 0 to the current one, softmax those scores, and take
the weighted sum of the cached values. Stage `attention`.

**What carries in.** This position's vector, plus every earlier position's keys and
values from this layer's cache.
**What carries out.** One vector, and one more entry appended to the cache.

**Observed.** Two different mechanisms, counted from which weights each layer used:
69 layers run the KDA form (`Q K V B FA FB`, carrying a recurrent state and a short
convolution history rather than per-position keys) and 24 run the MLA form
(`QA QB KA KB`, latent keys and values written per position). The cache length
observed at layer entry was 0 at position 0, 1 at position 1, 2 at position 2 — one
entry added per position, at every one of the 93 layers. `attention` cost 1,171.4 ms
at position 1, 8.7%.

---

## Cycle 6 — the layer

**What repeats.** The 15 stages listed above, 93 times per position, in the same
order every time.

**What carries in.** The residual vector from the previous layer, and the snapshot
stack built so far.
**What carries out.** A modified residual, an appended attention cache entry, and
for the KDA layers an updated recurrent state.

**Observed.** Per-layer total at position 1: median 73.7 ms, slowest L69 at 85.0 ms,
fastest L0 at 42.2 ms — layer 0 is the only dense layer and runs no experts. The
range across the 92 expert layers is narrow, 46.5 to 85.0 ms, so no layer dominates.

---

## Cycle 7 — the aggregation

**What repeats.** `AR()` takes every snapshot plus the current residual, normalises
each one, scores each against a stored fold vector, softmaxes those scores, and
blends the sources by the resulting weights. It runs twice per layer —
`pre-attention-aggregation` and `pre-mlp-aggregation-and-normalization` — and once
more at the very end.

**What carries in.** `nsnap` snapshots and the current residual, so `nsnap + 1`
sources.
**What carries out.** One blended vector. The snapshots are not modified.

**Observed.** Source count grows through the layers as snapshots accumulate: 1
source for layers 0 to 11, rising to 9 for layers 84 to 92. Total cost is small —
`op:AR snapshot aggregate` was 3.6 ms at position 1, about 0.03% — but it is the
mechanism by which a layer deep in the stack can weigh an early state against the
current one rather than only seeing what the previous layer handed it.

---

## Cycle 8 — the snapshot band

**What repeats.** Every twelfth layer, the current residual is pushed onto the
snapshot stack and a flag is cleared, so that at that layer the attention output
**replaces** the residual instead of being added to it. For the next eleven layers
the residual accumulates normally.

**Observed, directly, not derived.** Snapshots were pushed at layers
**0, 12, 24, 36, 48, 60, 72, 84** — 8 of them, `nsnap` going 1 to 8. At exactly
those 8 layers `have_prefix` was 0 at the `attention` boundary; at the other 85
layers it was 1.

Across all **61 positions and 9 runs** — English prose, SQL, JSON, a numeric
sequence and a repeated single token — there was exactly **one** distinct set of
snapshot layers and **one** distinct set of replaced layers, both the same eight.
The band structure is a property of the architecture, not of the input.

The residual norm shows the same 8 bands without being told about them:

```
L0=0.8    L1=2.1    L2=7.5    L3=7.5    L4=10.6   L5=15.9
L6=51.7   L7=42.4   L8=53.4   L9=60.4   L10=191.9 L11=129.6
L12=9.7   L13=11.5  L14=47.4  L15=47.8  L16=48.5  L17=49.5
L18=99.3  L19=99.2  L20=99.3  L21=169.8 L22=238.6 L23=324.1
L24=1.0   L25=1.5   L26=13.9  L27=16.2  L28=17.3  L29=21.2
...
L84=7.1   L85=9.9   L86=15.7  L87=16.7  L88=19.3  L89=25.0
L90=61.3  L91=66.4  L92=74.9
```

The norm collapses to around 1 at each band start and climbs through the band, as
high as 399.8 at L59. The state is not one smooth accumulation across 93 layers; it
is eight restarts, each keeping the previous states available through Cycle 7
rather than carrying them in the residual itself.

Tested across every position observed: in **488 of 488** cases (61 positions × 8
bands) the norm at the band start was the smallest value in that band. Band-start
norms ranged 0.39 to 9.7; the largest norm seen anywhere was 399.9.

### An observation that was not expected

The residual-norm profile over the 93 layers is nearly the same for completely
different prompts, and diverges only as context accumulates. Mean relative
difference across the 93 norms, between pairs of unrelated prompts:

| position | mean relative difference between prompts |
|---|---|
| 0 | 0.010 – 0.041 |
| 1 | 0.027 – 0.102 |
| 2 | 0.051 – 0.153 |

At position 0 the first twelve layers produce almost the same magnitudes whatever
the token:

```
A in=91019    0.8   2.1   7.5   7.5  10.6  15.9  51.7  42.4  53.4  60.4 191.9 129.6
B in=1008     1.0   2.3   8.6   8.9  11.8  16.4  53.0  43.0  51.9  59.7 205.0 137.8
C in=767      0.9   2.3   8.5   8.7  11.6  16.4  52.4  42.8  51.9  59.8 202.7 136.8
D in=276      1.3   2.6   9.1   9.7  12.8  17.0  54.9  44.9  53.1  61.0 211.3 142.3
E in=29349    0.9   2.2   8.0   8.0  10.9  16.2  52.7  43.0  52.5  59.8 197.8 132.3
```

**This is a statement about magnitude only.** Two vectors of 7,168 numbers can have
nearly equal norms and point in completely different directions, and the norm is
all that was recorded. What can be said is that the size of the state follows a
fixed shape set by the architecture, and that the shape is perturbed, not replaced,
by which token arrived. It sits beside the position-0 routing result in Cycle 3,
where unrelated prompts shared 11.5 to 13.1 of 16 experts, but neither observation
establishes that the states themselves are similar.

---

## Cycle 9 — the position

**What repeats.** 93 layers, then — only if this position is the one being asked
for — a final aggregation over all 8 snapshots plus the residual, a normalisation,
a projection to all 163,840 vocabulary scores, and an argmax.

**What carries in.** One token id, the position number, and the attention caches
from all earlier positions.
**What carries out.** One token id, 93 appended cache entries, and the updated
recurrent states.

**Observed.** `nsnap` was 0 at `request-reset` for all three positions and 8 at the
end of each — the snapshot stack is rebuilt from scratch every position and is not
carried forward. What is carried forward is the attention cache and the recurrent
state. The tail cost 319.1 ms at position 1 and 50.2 ms at position 2; the first
projection pays to fill the head cache, the second does not.

---

## Cycle 10 — the generation loop

**What repeats.** The token produced by Cycle 9 becomes the input token of the next
Cycle 9.

**Observed.** Position 0 consumed 91019 and produced nothing. Position 1 consumed
25528 and produced 418. Position 2 consumed **418** — the engine's own output — and
produced 276. The ids decode as `Rain falls` → ` on the`.

Run A and Arepeat, the same prompt in two separate processes, produced the same
chain: 23549, 418, 276, 13159 both times, with byte-identical expert selections at
all 92 routed layers of all 4 positions. The feedback loop is reproducible.

Run D shows the loop reaching a fixed point: fed token 276 four times, it predicted
220, then 137121, then **276, then 276**. Once enough history had accumulated, the
next token it proposed was the token it was being given.

This is the only cycle in the engine whose input is something the engine itself
produced. Every other cycle consumes either stored weights or state computed within
the current position.

---

## Cycle 11 — the request

**What repeats.** A request runs the prompt positions, then the generated ones,
until a stop condition.

**Observed.** Prompt positions that do not project are now buffered and evaluated
together in one pass over the weights; the projecting position is evaluated with
them. Measured warm on a two-token prompt: 4.81 / 4.86 s one position at a time
against 3.65 / 3.68 / 3.66 s batched. Decode positions cannot be batched this way
because each one needs the previous one's output.

---

## The frame this was asked under

The human stated the mapping before the observation: initial vectors are Context,
trunk and experts are Direction, moving through the stages is Execution, the
residue and snapshots are Outcomes, and those Outcomes return to layer 0 with more
Context to repeat the process.

Set beside what was observed, without extending it:

| the human's term | what was observed carrying it |
|---|---|
| Context | the embedding at position 0; the residual arriving at each layer; the attention cache of every earlier position; the snapshot stack |
| Direction | the fixed weights — trunk projections, 896 experts per layer, router bias, fold vectors — none of which changed during any run |
| Execution | the 15 stages per layer, 93 layers, 4,416 expert projections per position |
| Outcome | the residual leaving each layer, the 8 snapshots, the appended cache entry, and at the end one token id |
| the return to layer 0 | observed at Cycle 10 — position 2's input was position 1's output |

Two observations sit against the mapping rather than inside it, and are recorded as
observations only.

The return is not only at layer 0. Cycle 8 restarts the residual eight times within
a single position, and Cycle 7 lets any later layer reach back to those earlier
states. Within one position there are nine places where accumulated state is set
down and something begins again, not one.

Direction is fixed but not uniformly applied. The weights never changed, yet Cycle
3 selected different experts at every position of every prompt — and run D showed
that even an identical input token is routed somewhere else once the history behind
it differs, with 0 of 92 layers agreeing at any pair of positions. The same stored
Direction, read through a different Context, produced a different path.

Context appears to dominate the token itself. At position 0, where there is no
history, 28 pairs of unrelated runs — English, SQL, JSON, a numeric sequence —
shared a mean of 11.76 of their 16 experts per layer. By position 2 that had fallen
to 1.32, where it stayed for every later position observed. Whatever is selecting
the path, it is driven far more by what has accumulated than by what just arrived,
and the separation happens within two positions.

## What this does not cover

- Nine runs, 61 positions, one machine, one day. The longest context observed was
  14 positions. Nothing here says what happens at hundreds or thousands.
- Prompt kinds covered: short English statements, SQL, JSON, a Python-style
  numeric sequence, and one degenerate repeated token. No other natural language,
  no long document, no dialogue. The tokenizer used is ASCII-restricted, so
  non-ASCII prompts were not attempted at all.
- The plateau at about 1.3 shared experts was not compared against a chance
  baseline drawn from the actual gate distribution, so "close to chance" is a
  guess and is not claimed.
- Timings come from several runs at different cache states. The profiler run was
  colder (7.57 / 7.02 / 7.12 s per position); the first observation harness ran
  warmer (2.39 / 2.35 s); the multi-prompt runs ranged 2.24 to 12.97 s. The
  proportions between stages were stable; the absolute totals were not.
- The observation harnesses add work inside the stage boundaries, so their
  absolute timings are not clean measurements. Stage proportions were taken from
  `profile-stages`, which was built for timing.
- The residual was recorded as a norm, never as a direction. Everything said about
  states being "similar" across prompts is about magnitude only, and does not
  establish that the vectors are alike.
- Run C's drop in routing overlap from 5.97 to 2.77 between positions 4 and 5 was
  recorded, not explained, and the same kind of drop appears in Kjson (8.11 to
  3.65) and Jnum (8.23 to 4.32). Nor was it established why the norm reaches 399.8
  in one band and 25 in another.
- Expert selection was observed, but not the weights attached to the 16 choices.
  A shared expert at rank 1 and at rank 16 counted the same in every overlap figure
  above.
- `expert_result_hits` was 0 throughout, so every number is for fully computed
  experts with no cache reuse.
