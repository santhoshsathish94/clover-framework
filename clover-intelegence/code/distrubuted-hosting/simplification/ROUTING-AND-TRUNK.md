# The trunk and the routed experts

Not a simplification method. A correction to how the earlier methods were reasoned
about, and the measurement that settles it.

## The correction

After Methods B and C I was decomposing each layer's output by position, into an
across-position mean and a deviation, and asking which layers were "constant". The
creator stopped that:

> the constant part of every layer is the trunk and the variance part is the selected
> routed experts — if you just look at the position you will not understand the reality,
> even that's not the complete reality

The report says the same thing in §2.3: shared experts "retain a full-width path for
common transformations, whereas specialized routed experts operate in a compact latent
space". Two paths, one fixed and one selected, and I had been measuring their sum.

## First attempt, and why it measured the wrong thing

`port-moe-split.py` records the two paths separately at stage 116, where the engine does
`residual += routed + shared_output` and they are still distinguishable. All 92 pods
patched, rebuilt, eight sampled layers re-verified with tracing off.

Decomposing each path into mean and deviation across pooled positions and prompts:

| pod | routed mean-frac | shared mean-frac |
|---|---|---|
| 12 | 0.069 | 0.709 |
| 24 | 0.133 | 0.291 |
| 48 | 0.166 | 0.264 |
| 72 | **0.606** | 0.255 |
| 84 | **0.465** | 0.296 |
| 90 | 0.218 | 0.117 |
| 23 (MLA) | 0.150 | 0.088 |
| 91 (MLA) | **0.644** | 0.311 |

On this measure the shared path is **not** constant, 0.088 to 0.311 nearly everywhere,
and at pods 72, 84 and 91 the routed path is the *more* constant of the two.

That reads as a refutation and is not one. **Output variance is the wrong quantity.**
The shared path applies the same weights to every token, but its output moves because
its input moves. "Constant" is a property of which weights are used, and summing over
coordinates cannot see that. The measurement was correct and the question was wrong.

## What the selection actually does

`port-route-ids.py` records `sequence->selected`, the sixteen chosen expert ids, after
stage 24 — the choice itself rather than what it produced. Forty tokens across three
unrelated prompts: genesis, French, and `aaaa bbbb cccc`.

| pod | kind | always chosen | distinct experts | mean overlap of 16 | never used |
|---|---|---|---|---|---|
| 1 | KDA | 0 | 363 | 0.72 | 533 |
| 2 | KDA | 0 | 297 | 1.28 | 599 |
| 12 | KDA | 0 | 215 | 2.26 | 681 |
| 17 | KDA | 0 | 260 | 1.43 | 636 |
| 24 | KDA | **1** | 253 | 2.33 | 643 |
| 48 | KDA | 0 | 188 | 2.25 | 708 |
| 72 | KDA | 0 | 208 | 2.03 | 688 |
| 90 | KDA | 0 | 189 | 3.02 | 707 |
| 7 | MLA | 0 | 277 | 1.30 | 619 |
| 47 | MLA | 0 | 189 | 3.11 | 707 |
| 91 | MLA | 0 | 208 | 2.55 | 688 |

**No expert is chosen by every token**, at fifteen of sixteen layers measured; pod 24 has
exactly one. Two tokens share between 0.72 and 3.11 of their sixteen picks, so typically
around one in eight. A single layer draws on 188 to 363 distinct experts over forty
tokens.

There is no quiet subset of always-on experts behaving as a second trunk. The routed
path is per-token in the strong sense.

## What this changes about the methods

Methods B and C replaced the **attention** output with a constant and found it tolerated
at every layer individually. That now reads differently. The attention is not the part
of the layer carrying per-token selection; the routed experts are. Substituting the
attention leaves the mechanism that distinguishes one token from another entirely
intact, which is a more likely reason the substitution survived than anything about
constancy.

It also predicts the failures rather than explaining them away. Method A replaced
snapshots and collapsed everything; all-layer attention substitution collapsed
everything; both cuts broke on French. A simplification that touches the selected path,
or that accumulates enough error to change which experts get selected, should break, and
those did.

## What is still not measured

Whether the substitutions change **which** experts get chosen downstream. That is now
directly observable, since the selection is recorded, and it has not been run. If a
tolerated substitution leaves the routing unchanged and an intolerable one scrambles it,
that would connect the two halves of this.

## It does, and that connects them

Three runs of the same prompt: no substitution, pod 12 alone which held 12 of 12, and
all 68 which collapsed every prompt. Comparing the sixteen chosen experts per position
against the baseline.

| pod | one layer, overlap of 16 | top kept | identical | all 68, overlap | top kept |
|---|---|---|---|---|---|
| 1 | **16.00** | 16/16 | **16/16** | 9.75 | 15 |
| 6 | **16.00** | 16/16 | **16/16** | 6.75 | 12 |
| 11 | **16.00** | 16/16 | **16/16** | 4.56 | 3 |
| **12** | **8.75** | 10 | 1 | 3.88 | 4 |
| 13 | 14.88 | 16 | 1 | 4.25 | 6 |
| 24 | 15.56 | 16 | 2 | 5.00 | 9 |
| 48 | 15.56 | 16 | 0 | 5.88 | 8 |
| 72 | 15.56 | 16 | 2 | 7.38 | 4 |
| 92 | 15.94 | 16 | 7 | 5.44 | 5 |

Three things fall out.

**The experiment checks itself.** Pods 1, 6 and 11 sit before pod 12 and their selection
is bit-for-bit identical, 16 of 16 at every position. A substitution at layer 12 cannot
reach layer 11, and it does not.

**The tolerated substitution does disturb routing, locally.** At pod 12 itself the
overlap is 8.75 of 16: about half the chosen experts change. Attention is stage 9 and
routing is stage 24, so replacing the attention output changes that layer's own
selection. The network absorbs it — every layer downstream keeps a mean 15.53 of 16,
about 97 percent, and the top choice is unchanged at 16 of 16 positions nearly
everywhere.

**The intolerable one does not recover.** All 68 substituted drops the mean overlap to
about 5.66 of 16, 35 percent, with the top choice kept at only 3 to 15 positions and no
position retaining its full ordered list.

Put against the earlier figure that two genuinely different tokens share 0.72 to 3.11
experts of 16, the collapsed run's routing is nearer to *a different token's* than to
its own.

So the measure that tracks whether a simplification survives is not how constant the
substituted quantity was, nor where the layer sits, nor the size of the injected error —
all three of which failed to predict it. It is whether the routing downstream still
selects the same experts.

## Still not measured

The router weights against the selection, the two shared experts separately, and the
attention as a third path beside these two. The creator's "even that's not the complete
reality" has not been exhausted.

