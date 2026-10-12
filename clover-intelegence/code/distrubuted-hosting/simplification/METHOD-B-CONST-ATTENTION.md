# Method B: replace pod 12's attention with one cached vector

**Status: held on this prompt set. 12 of 12 tokens identical, worst pod-92 cosine
0.996102.**

## What motivated it

Pod 12, model layer 13, is the one layer of ninety-two whose attention output is close
to a fixed direction. Measured before this experiment was designed:

- cosine 0.9964 to 0.9998 between positions within a prompt, in all five prompts tested,
  out to position 15, with norms clamped between 3.81 and 3.90
- cosine 0.9949 to 0.9998 across prompts as unlike as biblical English, French and
  `aaaa bbbb cccc`, from inputs that disagree at cosine 0.14
- of all 92 pods, the only one above 0.99 for cos(p2, p8) of its attention output; the
  next are 0.970 and 0.968

If it is really a constant, computing it is wasted work.

## What was run

`port-const-attn.py` adds, behind `CLOVER_CONST_ATTN`, a substitution of a cached vector
for `sequence->attention` immediately after its per-head normalisation. The computation
still runs; only its result is replaced. **Fidelity is tested before any compute is
removed**, so a failure is a failure of the premise rather than of a rewrite.

The cached vector is pod 12's post-norm attention from **genesis at position 4** — a
different prompt from eleven of the twelve tested, and a different position from most.

Gate check first: with `CLOVER_CONST_ATTN` unset the layer still prints
`VERIFIED 12 france 5 2`, so the patch changes nothing when off.

## Result

| prompt | baseline | method B | cos at pod 92 | same |
|---|---|---|---|---|
| france | ` Paris` | ` Paris` | 0.999143 | yes |
| french | `.",\n` | `.",\n` | 0.996839 | yes |
| genesis | ` and` | ` and` | 0.999499 | yes |
| germany | ` Berlin` | ` Berlin` | 0.999223 | yes |
| hello | `,` | `,` | 0.999999 | yes |
| japan | ` Tokyo` | ` Tokyo` | 0.996102 | yes |
| n2 | ` of` | ` of` | 0.999996 | yes |
| nonsense | ` f` | ` f` | 0.996591 | yes |
| once | ` was` | ` was` | 0.998903 | yes |
| question | `?\n\n` | `?\n\n` | 0.999467 | yes |
| thethe | ` the` | ` the` | 0.996935 | yes |
| water | ` Celsius` | ` Celsius` | 0.998385 | yes |

Every token identical. The final representation moves by at most 0.0039 in cosine.

## The part that was not predicted

`hello` is a single token, so pod 12 runs only at position 0, where the real attention
norm is about 2.64 rather than the 3.85 of the cached vector. Substituting a vector
46 percent too large changed the output by 0.000001. That is a stronger statement than
"the attention is constant": at this layer the attention's **magnitude** barely matters
either, which says its contribution is small rather than merely fixed.

## What this does and does not license

It licenses skipping, at pod 12 only, the qkv projection, the decay, and the 96 x 128 x
128 recurrent state update, and dropping that layer's 6.29 MB of per-request state.

It does not license calling that a speed-up. **No timing was measured here**: the
substitution happens after the work, so this run is strictly a fidelity test. One
layer's attention out of ninety-two is at most about one percent of the layer budget,
so the value is the finding rather than the saving.

And it is twelve short prompts judged by greedy top-1 identity. The margin between the
top two candidates was measured earlier at 0.34 at position 0 against 3.68 at position
4, so short contexts are exactly where a token could flip. Nothing here says the method
survives long generation, other domains, or an actual evaluation.

## Worth testing next

Whether other layers pass the same substitution despite not looking constant — the
`hello` result suggests magnitude insensitivity may be broader than the constancy that
motivated this. That is the cheap, obvious follow-up and it has not been run.
