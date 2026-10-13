# Method D: fewer routed experts

**Status: K=12 holds cleanly. K=8 keeps every token while destroying one prompt's
representation. K=4 breaks.**

## What motivated it, and what it predicted

The routed path is where per-token work happens: no expert is chosen by every token and
two tokens share about one pick in eight. But that does not mean all sixteen picks carry
equal weight, so the weights were measured first.

| pod | top weight | top-4 share | effective experts | needed for 90% |
|---|---|---|---|---|
| 12 | 0.0771 | 0.2718 | 15.70 | 14.93 |
| 24 | 0.0799 | 0.2959 | 15.73 | 15.00 |
| 48 | 0.1047 | 0.3554 | 14.64 | 13.97 |
| 84 | 0.1973 | 0.5095 | 10.47 | 12.82 |
| 23 (MLA) | 0.1919 | 0.4822 | 10.86 | 12.82 |

Uniform would be 0.0625 and 16. The distribution behaves like 10.5 to 15.7 equally
weighted experts and needs 13 to 15 of them to reach 90 percent of the weight. There is
no few-dominate structure.

**That predicted the cut would be expensive.** It was run anyway, because a prediction
is not an observation.

## What was run

`port-topk.py` zeroes the weights of ranks at or beyond K and renormalises the rest, so
the layer behaves as though it routed to fewer experts. All 92 pods; four sampled layers
re-verified with `CLOVER_TOPK` unset. `mix_experts` still evaluates all sixteen, so this
measures fidelity only and claims no saving.

## Result

| K | tokens identical | worst cosine at pod 92 |
|---|---|---|
| 12 | **12 of 12** | **0.9867** |
| 8 | 12 of 12 | **0.2170** |
| 4 | 9 of 12 | 0.1581 |

**K=12 holds.** Dropping ranks 13 to 16, about ten percent of the router weight, changes
no token and the worst final representation is still at 0.9867.

**K=8 is the trap.** Every token matches, and eleven of the twelve prompts sit between
0.94 and 0.99. One does not: `thethe` falls to **0.2170** while still emitting ` the`.

| prompt | cosine at K=8 |
|---|---|
| hello | 0.9926 |
| genesis | 0.9804 |
| germany | 0.9407 |
| **thethe** | **0.2170** |

`The the the the the` answers ` the` almost regardless of what happens inside it, so its
token is the least informative in the set and it is exactly the prompt whose damage the
token measure hides. A harness that reported only token identity would have passed K=8
as clean.

**K=4 breaks**: french, n2 and question all change, and `thethe` reaches 0.1581 while
still matching.

## The prediction was wrong

The weight concentration said 13 to 15 experts were needed for 90 percent of the weight,
so K=12 should have hurt and K=8 should have been severe. K=12 is clean and K=8 leaves
eleven of twelve prompts above 0.94. **Share of router weight over-predicts the damage**,
which is the third measure in this programme to look reasonable and fail to predict an
outcome, after constancy and injected-error size.

## What this licenses

Routing to twelve experts rather than sixteen, on this evidence. That is a quarter of
the routed-expert work at each of 92 layers, and the routed path is the expensive one.

It does not license K=8 on the strength of its token score.

## Limits

Twelve short prompts, greedy top-1 and one cosine, no timing. The saving is only real
once `mix_experts` skips the zeroed ranks rather than computing and discarding them,
which has not been done. Whether K=12 survives long generation or an actual evaluation
is unmeasured, and `thethe` at K=8 is a standing warning about how little the token
measure sees.
