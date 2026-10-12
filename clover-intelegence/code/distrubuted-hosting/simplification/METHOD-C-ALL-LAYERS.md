# Method C: constant substitution, every KDA layer

**Status: every layer individually holds. All 68 substituted at once destroys the model.
The result that motivated Method B turns out not to be about pod 12 at all.**

## Why this was run

Method B replaced pod 12's attention with a cached vector and nothing moved. Pod 12 had
been chosen because its attention was measured to be near-constant, the only layer of 92
above 0.99 for cos(p2, p8).

But the single-token prompt in that run substituted a vector 46 percent too large in
magnitude and changed the output by 0.000001. That does not fit "the attention is
constant, so a constant is a fair substitute". It fits something weaker and broader, and
the way to find out is to do it everywhere rather than argue about it.

`port-const-attn-layered.py` makes the hook per layer: `CLOVER_CONST_ATTN` names a
prefix and pod N loads `<prefix>.N.f32`, so any subset is enabled by choosing which
files exist. All 68 KDA pods patched and rebuilt; six sampled layers re-verified with
the hook off.

Each layer's cached vector is its own post-norm attention from **genesis at position 4**,
and the sweep prompt is **france** — a different prompt.

## All 68 at once

Every one of the twelve prompts collapsed to ` the`. Complete failure.

## One at a time

| | |
|---|---|
| layers tested | 68 |
| tokens unchanged | **68 of 68** |
| cosine at pod 92 | mean 0.99305, min 0.95238, max 0.99917 |

| depth band | mean cosine |
|---|---|
| pods 1 to 23 | 0.99501 |
| pods 24 to 46 | 0.99694 |
| pods 47 to 69 | 0.99374 |
| pods 70 to 92 | **0.98586** |

Six most affected: pods 90 (0.9524), 88 (0.9607), 89 (0.9641), 84 (0.9689), 6 (0.9823),
60 (0.9858).

## What this says about Method B

**Pod 12 is not distinctive here.** Its cosine is 0.999143; pod 17 is higher at 0.999171
and pod 26 is comparable at 0.999079. Thirty-odd layers sit above 0.996.

So Method B's success was not caused by the property that motivated it. Any single KDA
layer's attention can be replaced by a constant taken from a different prompt, and the
answer does not move. The constancy of pod 12 is real and was measured carefully, but it
is not why the substitution worked.

That is worth stating plainly: **a correct measurement motivated an experiment whose
success had a different cause.** Had the sweep not been run, Method B would read as
"the near-constant layer tolerates a constant", which is true and misleading.

## What it says about the network

A single layer's attention contribution, out of 68 KDA layers, is individually
dispensable on this prompt — and the depth ordering is the opposite of what the
alignment work suggested. The last block is the most sensitive, not the first.

The contrast with the all-at-once collapse means the tolerance is not additive. Where it
breaks between one and sixty-eight has not been measured, and that is the next thing to
run.

## Cumulative: it breaks between 4 and 8, or holds to 32, depending on the order

Growing the substituted set, with two orderings. Shallowest-first is the obvious one.
Most-tolerant-first takes its order from the per-layer sweep's own cosines, so the
selection is measured rather than guessed.

| n | shallow first | | tolerance ordered | |
|---|---|---|---|---|
| | token | cos | token | cos |
| 2 | ` Paris` | 0.9937 | ` Paris` | 0.9963 |
| 4 | ` Paris` | **0.7149** | ` Paris` | 0.9975 |
| 8 | ` France` | 0.3474 | ` Paris` | 0.9967 |
| 16 | ` the` | 0.2653 | ` Paris` | 0.9897 |
| 24 | ` France` | 0.2811 | ` Paris` | 0.9452 |
| 32 | ` not` | 0.3010 | **` Paris`** | **0.8896** |
| 48 | ` the` | 0.3229 | ` the` | 0.5759 |
| 68 | ` the` | 0.5699 | ` the` | 0.5699 |

Shallowest-first breaks at eight. Choosing by measured tolerance holds the answer to
**thirty-two of sixty-eight** and breaks between 32 and 48. Which layers are chosen
matters more than how many.

The representation degrades long before the token does: at 32 the answer is still
` Paris` while the final vector has fallen to cosine 0.8896, and in the shallow arm at
n=4 the token survived a drop to 0.7149. **Token identity lags the damage.** A
conservative cut by the continuous measure would be nearer sixteen layers, at 0.9897,
than thirty-two.

## Limits

One prompt for the sweep, france, judged by one token and one cosine. The all-at-once
arm used the full twelve. A layer that survives alone on one prompt may not survive on
another, and nothing here measures the cumulative case, which is the one that would
matter for an actual saving. No timing was measured; the substitution happens after the
work runs.
