# Understanding Kimi K3

What this model is and how it works, written from reading the source rather than papers
or summaries. Built in order: **the C code first, then the weights, then everything
else.** Each part says plainly what has been read and what has not.

The point of this document is that hours of reading should become a page someone can
understand. If something here is vague, it means I did not understand it well enough yet.

---

# Part 1 — The C code

**Read so far:** `include/k3/k3.h` in full (572 lines); in `src/core/k3_ops.c` the
function map plus `k3_kda_step`, `k3_moe`, `k3_decoder_layer`, and the matmul commentary.
**Not yet read:** the MXFP4 matmul body (~400 lines), `k3_mla_cached`, `k3_kda_layer`,
`k3_bind.c`, `k3_trunk.c`, `k3_cache.c`, the tokenizer, the chat layer.

## The shape of the thing

93 layers. Layer 0 is an ordinary dense feed-forward layer. The other 92 use a mixture
of experts. Every layer also has an attention mechanism, and there are two different
kinds: **69 layers use Kimi Delta Attention, 24 use Gated MLA**.

Hidden width is 7168. Vocabulary is 163,840.

The MLA layers sit at positions 3, 7, 11 and so on every four layers, but the pattern
breaks at the end: **91 and 92 are both MLA**. So the model finishes with two quadratic
attention layers back to back.

## How a token moves through one layer

Most transformers keep one running sum — the residual stream — and each layer adds to
it. K3 does something else. It keeps a **stack of snapshots**, and every layer looks
back over that whole stack.

```
running = h
if the stack is not empty:
    h = attend over [stack…, running]        # replaces h
if this layer is a multiple of 12:
    push running onto the stack
    running = nothing                        # and clear it
h = norm(h);  h = attention(h)
running = (running is nothing) ? h : running + h
h = attend over [stack…, running]            # always, no emptiness check
h = norm(h);  h = experts(h) or dense(h)
running = (running is nothing) ? h : running + h
```

"Attend over" here means: normalise each source, score each one against a single learned
vector, softmax the scores, then mix the **unnormalised** sources by those weights. The
scores come from normalised keys; the output is built from the raw values.

The catch is the clearing. On every twelfth layer the running sum is pushed onto the
stack and then wiped, so at that layer it does **not** also appear as a separate source.
Everywhere else it does. Get this wrong and the model still runs and still produces
fluent text.

## Kimi Delta Attention — the 69 layers

This is the part I found most interesting, because it is a learning rule running inside
the forward pass.

Each head carries a state matrix `S`, 128 by 128. For every token:

1. decay: each row of `S` is multiplied by its own forget factor. Not one number for the
   head — one per key channel.
2. read: `u = Sᵀk`. This is what the state currently predicts for this key.
3. **write: `S += k · β(v − u)ᵀ`**
4. output: `o = Sᵀq`, read from the state **after** the write.

Step 3 is the whole idea. `(v − u)` is the difference between what actually arrived and
what the state predicted. The state is corrected by its own error, scaled by `β`, which
is a single number per head produced by a small projection of the input.

So the model is not just retrieving from memory. It is running a tiny online
error-correcting update, once per head per token, 6,624 times per token at K3's scale.

Around that recurrence sit: a causal depthwise convolution of width 4 with SiLU fused
in; L2 normalisation on `q` and `k` but never on `v`; a head-wise RMSNorm on the output;
and finally a sigmoid gate. The order matters — **normalise first, then gate**.

## Gated MLA — the 24 layers

Ordinary attention, but compressed. Keys and values are projected down to a 512-wide
latent and back up, so the cache stores less.

Two oddities:

- **No position rotation at all.** The 64 "rope" dimensions still exist, are still glued
  onto every query and key, and are still cached — they are simply never rotated. The
  softmax scale is over the full 192 width, not the 128 that carry content. Deleting the
  unused 64 changes the model.
- The output gate multiplies **before** the output projection with no normalisation —
  the opposite order to KDA. The engine says plainly that sharing one code path between
  the two is wrong.

## The experts — 92 layers, 896 experts each

The routing is not a softmax. Each expert gets an **independent sigmoid score**, so the
scores do not compete and do not sum to one. A frozen bias is added only to **choose**
the top 16; the weights used for mixing come from the scores **without** that bias. Then
they are renormalised.

The structural surprise is where the experts live:

```
x (7168)
  ├─ router reads the FULL 7168 width
  ├─ down-project to latent 3584
  │     each of 16 chosen experts: 3584 → 3072 → SiTU-GLU → 3584
  │     weighted sum of the 16, in latent space
  │     RMSNorm the SUM (not each expert)
  │     up-project 3584 → 7168
  └─ shared expert reads the ORIGINAL 7168: → 6144 → 7168, added UNWEIGHTED
```

The routed experts never see the full hidden width. They work in a narrower 3584 space.
Only the router and the two shared experts touch the full 7168. Every token pays for the
shared experts at full strength, with no weight at all.

## Numbers and formats

The activation is SiTU-GLU: `a = 4·tanh(gate/4)·sigmoid(gate)`, `u = 25·tanh(up/25)`,
output `a·u`. Both halves are bounded, so the output cannot exceed 100 in magnitude. The
sigmoid sees the *uncapped* gate.

The always-active weights ship as **bf16**. Widening bf16 to fp32 is a pure 16-bit left
shift — no rounding — so reading bf16 and widening inside the matmul computes exactly
what an fp32 copy would, at half the bytes.

The routed experts ship as **MXFP4** and are multiplied straight out of that form. Four
bits per weight plus one shared 8-bit exponent per 32 weights: 0.53125 bytes per
parameter. One expert is 33,030,144 parameters and exactly 17,547,264 bytes. Unpacked it
would be 132 MB, and a token touches 1,472 of them — 194 GB per token if anything
widened them, which is why nothing does.

## Engineering habits worth stealing

These are about how the code is built, not what the model is.

- **Three implementations agree bit for bit.** Scalar, AVX2 and bf16 matmuls all use four
  accumulators partitioned by `i % 4`, reduced as `(a0+a1)+(a2+a3)`, in double. The
  partition is written out by hand rather than left to the compiler, because it fixes a
  summation *order*. The AVX2 path uses multiply-then-add rather than FMA, because FMA
  rounds once and the scalar path rounds twice.
- **Invariants are named, and each has a fixture that would catch breaking it.** Three
  are called out as places where a wrong implementation still produces fluent text.
- **Silent corruption is made loud.** If a routed expert fails to load, a counter
  increments and the caller is required to fail the run, because a token computed with
  part of its expert contribution missing still looks like a normal token.
- **The header says what is *not* true of the code.** An earlier version listed two
  invariants from the chunked parallel form of the delta rule; the engine runs the naive
  sequential recurrence, so those matrices are never formed and no test could have caught
  getting them wrong. They were moved out rather than left to look load-bearing.

---

# Part 2 — The weights

**Not started.** Nothing here rests on having read a single weight. Open questions that
need the weights rather than the code:

1. How many of the 896 experts per layer are ever selected, over varied input?
2. Are some experts effectively dead?
3. Does the depth-dependence of expert reuse we measured (12–17% early, 63–72% late)
   show up in the weights, or only in the routing?
4. What do the routing biases look like across layers?
5. How much do the two shared experts differ from the routed ones?

---

# Part 3 — Everything else

**Not started.** Tokenizer, chat template, the streaming cache, the trunk packer, and
how the checkpoint is laid out on disk.
