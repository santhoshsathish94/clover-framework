# Kimi K3, one stage at a time

The model is not 93 things happening in a row. It is a long sequence of individual
transformations, per position, with a recurrent aggregation that ties groups of layers back
together rather than passing state strictly forward. This document walks those stages one at
a time and proves each one separately against the running engine.

Each entry has the same four parts:

1. **The equation** — what the stage is, mathematically.
2. **What this stage exactly does** — in words, including what it does *not* do.
3. **The real data** — the actual captured numbers from a real execution.
4. **What the equation gave** — my computed result against the engine's, and the verdict.

Evidence throughout comes from one traced execution of the released checkpoint on the pure-C
engine: prompt `The capital of France is`, 5 tokens, raw per-stage vectors captured to
`e1.bin`. Weights are read from the checkpoint, never from the engine's own buffers.

---

## Stage 1 — the embedding lookup

### 1. The equation

$$h^{(0)}_p \;=\; W_{\text{embed}}\big[\,t_p\,\big], \qquad
W_{\text{embed}} \in \mathrm{BF16}^{\,163840 \times 7168}$$

For each position $p$, take token id $t_p$, select that row of the embedding table, and widen
each BF16 value to float32. There is no sum, no product, and no nonlinearity.

### 2. What this stage exactly does

It turns five integers into five vectors of 7,168 floats.

- **Input**: the token ids, one per position. Nothing else. No weights are combined, no
  context is consulted, position $p$ does not see position $q$.
- **Operation**: a row selection followed by a format widening. BF16 to F32 is exact by
  construction — BF16 *is* the top 16 bits of an F32, so widening is a shift and a zero-fill,
  not a conversion that can round.
- **Output**: $h^{(0)}_p$, the residual stream at its starting value, 7,168 floats per
  position, 35,840 floats in total for this prompt.
- **What it does not do**: no positional information is added here. No normalization. This is
  the one stage in the model that performs no arithmetic at all, which is why it is the
  natural zero point — anything that fails to match here is a reading error, not a rounding
  difference.

### 3. The real data

Source: engine tap site 31 (`embed`), five records of 7,168 float32.
Table: `language_model.model.embed_tokens.weight`, BF16, `[163840, 7168]`, in
`model-00094-of-000096.safetensors`, memory-mapped directly from the checkpoint.

**The output is a pure widening.** Every captured float has its low 16 bits zero. A float32
produced by any computation essentially never does.

```
pos 0..4:  0 of 7168 floats have nonzero low 16 bits   => pure widening: True
```

**The token ids were not taken from the log.** To keep the check independent, each position's
engine vector was searched against all 163,840 rows of the table. Exactly one row matched in
every case:

```
pos 0  token id 1008    candidates sharing component 0: 117
pos 1  token id 10484   candidates sharing component 0: 299
pos 2  token id 318     candidates sharing component 0: 219
pos 3  token id 15383   candidates sharing component 0:  92
pos 4  token id 387     candidates sharing component 0:  52
```

**The recovered ids decode back to the prompt**, which is what makes the match above
non-circular — a wrong row would not spell anything:

```
1008 b'The'   10484 b' capital'   318 b' of'   15383 b' France'   387 b' is'
joined: b'The capital of France is'   (24 bytes)
```

The engine log independently records `tokenized: 24 bytes -> 5 ids`. Byte count and id count
both agree with the recovered rows.

### 4. What the equation gave

Bit-for-bit against the engine, all five positions:

| position | token id | identical floats | max ulp |
|---|---|---|---|
| 0 | 1008 | 7168 / 7168 | 0 |
| 1 | 10484 | 7168 / 7168 | 0 |
| 2 | 318 | 7168 / 7168 | 0 |
| 3 | 15383 | 7168 / 7168 | 0 |
| 4 | 387 | 7168 / 7168 | 0 |

**35,840 of 35,840 floats identical. Max ulp 0.**

The first six components of position 0, with the XOR of my float against the engine's:

| i | bf16 bits | equation | engine | xor |
|---|---|---|---|---|
| 0 | 0x3D05 | 0.032470703 | 0.032470703 | 0x00000000 |
| 1 | 0xBB21 | -0.002456665 | -0.002456665 | 0x00000000 |
| 2 | 0xBD14 | -0.036132812 | -0.036132812 | 0x00000000 |
| 3 | 0xB8AA | -8.1062317e-05 | -8.1062317e-05 | 0x00000000 |
| 4 | 0x3C14 | 0.0090332031 | 0.0090332031 | 0x00000000 |
| 5 | 0x3CBF | 0.02331543 | 0.02331543 | 0x00000000 |

**Verdict: identical, not approximate.** The equation for stage 1 is exactly the engine's
behavior, and the agreement is at the level of raw bits rather than a tolerance.

One caveat worth stating plainly, because it bounds what this proves: stage 1 matches trivially
*because* it contains no summation. It establishes that the capture is being read correctly,
that the weight file is being addressed correctly, and that the comparison method is sound. It
says nothing yet about stages that accumulate.

Reproduce with `stage1.py` on the trace host.

---

## Stage 2 — the pre-attention aggregation, which does not run here

### 1. The equation

$$h \;\leftarrow\; h \qquad \text{when } d = 0$$

Identity. Not because an aggregation computed something that happened to equal its input, but
because the aggregation is never entered.

### 2. What this stage exactly does

Nothing, at this point in the model.

The engine carries a stack of snapshots of the residual stream and, before attention, blends
the current residual with every stored snapshot. That blend is the mechanism that stops the
model being a straight chain. But it is guarded:

```c
/* aggregation before attention, only when snapshots already exist */
if (*n_blocks > 0) {
    ...
    k3_attn_res(h + t*E, src, foldA, *n_blocks + 1, E, c->rms_eps);
}
K3_TRACE_VEC("attn_res.pre_attn", h, T*E);
```

At this stage `*n_blocks == 0` — no snapshot has been pushed yet, because the first push
happens *after* this point in the same layer. The guard is false, the kernel is not called,
and the trace is emitted on an untouched buffer.

So there is no summation, no weight, no `eps` and no `expf` in this stage. None of it
executes. The residual arrives and leaves unchanged.

### 3. The real data

| evidence | value |
|---|---|
| `stack.depth` | 0 |
| guard `*n_blocks > 0` | false, kernel not entered |
| `embed == layer.in` | 7168 / 7168 floats, all 5 positions |
| `embed == attn_res.pre_attn` | 7168 / 7168 floats, all 5 positions |
| L2 across the three sites | 4.13786926, unchanged |
| FNV across the three sites | `f45927e03fd96f75`, unchanged |

The hash is byte-identical at `embed`, `layer.in` and `attn_res.pre_attn`. That is what a
pass-through looks like, and what a computation essentially never produces.

### 4. What the equation gave

Difference from the engine: **exactly zero**, for the same reason as stage 1 — nothing was
computed. 35,840 of 35,840 floats identical.

### How this was nearly recorded wrong

The first attempt at this stage worked from the captured numbers alone. It proposed candidate
operations, scored each against the data, and the winner was a softmax over RMS-normalized
projections — which, with a single source, gives weight exactly 1.0 and returns the input
untouched. Every number agreed. The conclusion drawn was that stage 2 is that formula
degenerating to the identity.

That is wrong. The kernel is not invoked at all.

Both accounts predict bit-identical output, so no amount of comparing vectors could separate
them. Only reading the code that runs could. The lesson is narrow and worth keeping: a
hypothesis that reproduces the observations to the last bit can still misdescribe the machine,
and at a control-flow boundary it usually will.

### Noted for later stages, not acted on

Reading `k3_attn_res` to establish the above also settled two things that belong further
along, recorded here so they are not rediscovered:

- the source stack is ordered **snapshots first, current residual last**, with
  `nsrc = n_blocks + 1`
- the weighted sum accumulates in **float32**, source-major — earlier reconstructions of this
  kernel accumulated in float64, which is the likely cause of the partial bit-agreement seen
  in those checks
