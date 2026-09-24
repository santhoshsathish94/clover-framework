# Understanding Kimi K3

What this model is and how it works, written from reading the source rather than papers
or summaries. Built in order: **the C code first, then the weights, then everything
else.** Each part says plainly what has been read and what has not.

The point of this document is that hours of reading should become a page someone can
understand. If something here is vague, it means I did not understand it well enough yet.

---

# Part 1 — The C code

**Read so far:** `include/k3/k3.h` in full (572 lines); in `src/core/k3_ops.c` the
function map, the foundational ops, `k3_matmul`, `k3_mla_cached`, `k3_kda_step`,
`k3_moe`, `k3_matmul_bf16`, `k3_router`, `k3_decoder_layer_inc`; in `src/cli/k3_run.c`
the phase order, `forward()` and the decode loop.
**Observed, not just read:** the whole forward pass, instrumented and run on both the
test fixture and the real checkpoint — see the next section.
**Not yet read:** the MXFP4 matmul body (~400 lines), `k3_kda_layer`, `k3_moe_prefill`,
`k3_attn_res`, `k3_bind.c`, `k3_trunk.c`, `k3_cache.c`, the tokenizer, the chat layer.

## The flow, end to end

This is the spine. Everything after it is detail hanging off one of these steps.

### Once, at startup

1. Parse the command line. **Refuse to run without a Direction** (our addition; the
   engine's own first refusal is the memory check at step 6).
2. Read `config.json` from the model directory into the config struct.
3. If asked, compute the automatic memory budget from what the machine has free.
4. Turn the prompt text into token ids.
5. Work out how much KV cache this request needs — 2.37 MB per position.
6. Index the checkpoint: 96 safetensors shards become one table of tensors.
7. Bind the weights. The trunk is either held in RAM or opened for streaming; the
   embedding table, the final norm and lm_head are always resident.
8. Print the memory plan and **refuse to start** if it exceeds what a plan may use.
9. Allocate the working memory: one recurrent state slot per layer, a KV cache per MLA
   layer, and the expert cache slots.
10. If resuming, restore the saved state.

### Once per generated token

The decode loop does the same thing every time: call `forward()` and take the argmax.

- **Step 0 is the prefill.** It feeds the entire prompt through `forward()` in one call,
  so all the KV and recurrent state exist before any token is generated. This runs even
  with `--gen 0`, which is how a reusable warmed prefix is made.
- **Every later step feeds one token** — or a small batch when speculating, where drafts
  are proposed and then verified in a single sweep.
- `forward()` returns the logits for the last position; argmax picks the next token; the
  loop prints one row and repeats until the limit or a stop token.

### Inside one `forward()` pass

1. **Embed.** Copy each token's row out of the embedding table into the working buffer.
2. **Clear the snapshot stack.** Clear the KDA recurrent state too — *but only on the
   full-recompute path*. Incremental decode must carry it across steps, so it is
   deliberately left alone.
3. **Walk the 93 layers in order.** For each layer:
   - if streaming, bind this layer from disk and **hint the next one**, so its read
     overlaps this layer's arithmetic. The order is fixed 0…92 every single token, which
     is exactly why the hint is never wrong.
   - point this layer's experts at the cache
   - run the layer
   - **if any routed expert failed to load, abort the whole run.** A token computed with
     part of its expert contribution missing still looks like a normal token.
4. **One model-level aggregation.** Beyond the two inside every layer, there is exactly
   one more pair of weights that mixes the snapshot stack with the final hidden state.
5. **Normalize the last position, project through lm_head, take the argmax.**

The trunk is walked in a fixed layer order, 0…92, on every pass, which is why the
next-layer prefetch hint is never wrong and why the tuning advice is to feed the trunk
before the expert cache.

**Corrected later from measurement:** an earlier version of this said the trunk is re-read
in full on every token. It is not, under the configuration every run used. With
`--trunk-gb` large enough to pin all 93 layers it is read **once** and held — the engine
reported `binds 744, hits 651 (87.5%), reads 93` over eight passes, and trunk time was
5.4–5.7 s whether the run produced one token or eight. Re-reading happens only when the
budget cannot hold the trunk, a regime none of these runs entered.

### Inside one layer

```
running = h
if stack not empty:      h = attend over [stack…, running]
if layer % 12 == 0:      push running onto stack; running = nothing
h = norm(h);             h = attention(h)          # KDA or MLA
running = running is nothing ? h : running + h
h = attend over [stack…, running]                  # always
h = norm(h);             h = experts(h) or dense(h)
running = running is nothing ? h : running + h
```

### Inside KDA (69 of the layers)

project q, k, v → short convolution with SiLU fused → L2-normalize q and k only →
per-head `β` → per-channel decay → **the recurrence** → head-wise norm → gate → project out.

### Inside MLA (24 of the layers)

project q through a low-rank pair → **one** projection gives the compressed key-value
latent *and* the shared position slot → normalize the latent only → cache the expanded
per-head keys and values → score, softmax, mix → gate → project out.

### Inside the experts (92 of the layers)

route on the full width → hand all 16 chosen experts to the cache **at once** so their
reads overlap → project down to the latent → run the experts there → weighted sum →
normalize the sum → project back up → add the shared experts, unweighted.

## What a real token actually did

Reading the code gives you the flow it *should* follow. To see the flow it *does*
follow, the engine was instrumented: `K3_TRACE=<path>` makes every stage of the forward
pass append one JSON record naming the tensor that passed through it — element count,
L2 norm, min, max, mean, count of non-finite values, and an FNV1a-64 of the raw bytes.
The statistics show how a value behaves. The hash establishes identity, so "the tensor
leaving stage A is bit-for-bit the one entering stage B" becomes checkable instead of
assumed.

Two runs, both with the instrumentation in place:

- the 13-layer test fixture against its torch oracle — **all four gates pass, "ENGINE
  MATCHES THE REFERENCE EXACTLY"**, which is what establishes that the taps changed no
  arithmetic;
- the real checkpoint, int8 trunk, prompt "The capital of France is", one token, 93
  layers, 39.1 s, 54.47 GB of trunk and 99.72 GB of experts read.

**The real run emitted token 17374 = `" Paris"`.**

### Verified by hash, not by reading

Over the real pass and 42 fixture passes:

| statement | result |
|---|---|
| `layer.out[L]` is bit-identical to `layer.in[L+1]` | 92/92 and 504/504 |
| empty stack ⇒ the pre-attention aggregation is a no-op | holds, hash unchanged |
| non-empty stack ⇒ the aggregation changes `h` | 92/92 and 504/504 |
| a snapshot layer ⇒ residual is **replaced** by the attention output | 8/8 and 210/210 |
| an ordinary layer ⇒ residual is **added to** | 85/85 and 336/336 |
| no non-finite value at any stage | 837 and 4,914 tensors |
| routed experts within a decision are distinct | 460/460 |

### The residual stream is a sawtooth, not a climb

This is the thing that reading the code did not show. Snapshots fire at layers
**0, 12, 24, 36, 48, 60, 72, 84** — spacing exactly 12, eight of them. At each one the
residual **collapses**, because the layer replaces it with the attention output rather
than adding to it:

```
L11 274.96  ->  L12  12.34
L23 356.22  ->  L24   1.38
L83 138.83  ->  L84  17.10
```

Mean residual L2 at snapshot layers is **7.14** against **82.56** over all layers; the
run's maximum is 482.16. So the network is eight blocks of twelve layers, each starting
from near nothing and growing, with the accumulated history parked on a stack that the
aggregation re-injects into the norms. The every-12th-layer boundary is a magnitude
regulator, not bookkeeping.

### What the trained weights do that random weights do not

The fixture is randomly initialized, and three things that looked like structure in it
turned out to be artifacts of that. All three were caught by running the real model:

1. **"RMSNorm pins the L2 to √n."** True in the fixture to four decimals, because its
   gains are near unit. False in the real model: observed norm outputs run 15.2–126.6
   and 1.7–46.5 where √n is 189.3. The learned gain does real work and varies roughly
   eightfold across layers. At layer 0 the post-attention gain nearly zeroes its output
   (L2 1.72).
2. **"MLA contributes far less than KDA."** In the fixture MLA was ~12× smaller. In the
   real model MLA is *larger* on average — mean L2 19.49 across 24 layers against 15.55
   across 69 KDA layers.
3. The only fixture number that survived scrutiny was a micro-one: layer 0's norm output
   is 63.224 rather than 64. That is `rms_eps` becoming visible because the embedding is
   small — predicted 63.2280 from the epsilon alone, observed 63.2240.

### Routing, observed

460 routing decisions in the pass — 92 MoE layers × 5 positions, one per token per layer.

- Weights **sum to exactly 1.00000**, so this checkpoint renormalizes the top-16.
- A single expert's share ranges from 0.003 to **0.645**.
- The weights are **never** in descending order — in 460 of 460 decisions. This is not a
  defect: the router selects by the *biased* score but stores the *unbiased* one, so
  non-monotonic weights are direct evidence the routing bias is doing something. It
  demotes the top-scoring expert out of first place in **20%** of decisions.
- Distinct experts per layer, out of 80 slots: 79.0% early, 75.2% middle, 77.4% late —
  roughly flat. This does **not** contradict the earlier depth-dependent reuse
  measurement (12–17% early, 63–72% late); that was cache hit rate over a long run, this
  is distinctness over five positions. They are different quantities and a long prompt is
  needed to compare them.

### What the instrument caught in me

Worth recording, because it is the argument for instrumenting at all:

- The first routing tap sat at the call site in the layer, reading the `idx`/`wt` arrays
  passed to the batched prefill path — **which never writes them**. Every record came
  back as sixteen zeros. Had I trusted it, I would have reported "K3's routing weights do
  not sum to 1" as a property of the model. The tap belongs inside `k3_router`, where the
  decision is actually made.
- Two of my three "invariants" were assertions about the fixture, not the model.
- 14 `test_ops` failures were absent fixture files and a wrong working directory, not
  anything I had changed. I nearly attributed them to the instrumentation.

### Caveats on these numbers

One prompt, five positions, one token, int8 trunk, one machine. Magnitudes are specific
to this checkpoint and quantization; the structural findings (sawtooth, snapshot spacing,
bias reordering, the hash chain) are not.

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

"Attend over" here means: normalize each source, score each one against a single learned
vector, softmax the scores, then mix the **unnormalized** sources by those weights. The
scores come from normalized keys; the output is built from the raw values.

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
in; L2 normalization on `q` and `k` but never on `v`; a head-wise RMSNorm on the output;
and finally a sigmoid gate. The order matters — **normalize first, then gate**.

## Gated MLA — the 24 layers

Ordinary attention, but compressed. Keys and values are projected down to a 512-wide
latent and back up, so the cache stores less.

Two oddities:

- **No position rotation at all.** The 64 "rope" dimensions still exist, are still glued
  onto every query and key, and are still cached — they are simply never rotated. The
  softmax scale is over the full 192 width, not the 128 that carry content. Deleting the
  unused 64 changes the model.
- The output gate multiplies **before** the output projection with no normalization —
  the opposite order to KDA. The engine says plainly that sharing one code path between
  the two is wrong.

## The experts — 92 layers, 896 experts each

The routing is not a softmax. Each expert gets an **independent sigmoid score**, so the
scores do not compete and do not sum to one. A frozen bias is added only to **choose**
the top 16; the weights used for mixing come from the scores **without** that bias. Then
they are renormalized.

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

## What going slow turned up

Three findings, all verified, none reported to anyone.

### 1. The MXFP4 trunk will not bind on upstream `a2ad8e5`

```
k3_bind_mem: language_model.model.layers.0.self_attention_res_proj.weight
             has 1904 elements, engine expects 7168
trunk bind failed at layer 0
```

1904 × 2 bytes = 3808 = exactly 7168 × 0.53125, the MXFP4 rate. The packer quantized a
vector that has to stay full precision, and the binder reads the packed bytes as
elements. Our MXFP4 measurements were taken on the archived `mxfp4-trunk` branch, which
carries the matching binder; the format does not work on a checkout of upstream alone.
The int8 and bf16 trunks are unaffected. Worth confirming which side is wrong before
saying anything upstream.

### 2. Our own Direction boundary breaks a CLI contract test

`make test` fails at the chat option contract: the test asserts `--no-think` is refused
*because of the thinking flags*, but the Direction refusal now fires first and the test
sees the wrong reason. This is a consequence of our change, not upstream's, and it is
exactly the kind of thing that only shows up when the whole suite is run. The boundary
has to move after argument validation, or the test has to pass a Direction.

### 3. Two block comments describe code that no longer exists

Both sit above the matmul reduction, which is what the repository's bit-identity
guarantee rests on.

| Where | The comment says | The code does |
|---|---|---|
| line 254 | "FOUR ACCUMULATORS, PARTITIONED BY i%4, REDUCED AS (a0+a1)+(a2+a3)" | sixteen accumulators, partitioned by i%16, reduced as a 16→4→1 tree |
| line 1101 | "MUL THEN ADD, never `_mm256_fmadd_pd`" | calls `_mm256_fmadd_pd` four lines later, at 1132–1141 |

The comment at 254 also claims the bf16 and AVX2 paths "reproduce this exact partition
and this exact tree". They do reproduce each other exactly — at sixteen, not four.

**This is not a correctness bug.** Every implementation agrees at sixteen accumulators
with explicit fused products, so the three paths still match bit for bit, and the
fixtures still pass. The inline comments right next to the code (lines 272, 1119, 1156)
are accurate and current. It is the two summarizing block comments that were left behind
when the reduction widened from four to sixteen and moved from multiply-then-add to
explicit fused products.

Why it is still worth something: this repository's whole review discipline rests on the
comments being the authority on *why*. A comment that states the opposite of the code is
an invitation for the next person to "restore" the reduction to what the comment says —
which would silently break bit-identity with the fixtures, and the fixtures are the only
thing that would catch it.

**None raised.** Small, verifiable, uncontroversial — but nothing goes upstream without
being asked for.

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
