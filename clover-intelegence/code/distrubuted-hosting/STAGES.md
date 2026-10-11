# Stages, end to end

Every step of a request has a number. The number says *where* in the process an object
is, never *when*, so stage 5521 is always "layer 46, first step" whatever the prompt and
whichever position is being computed. Both the service that owns a stage and the service
it goes to next are arithmetic on that number, so no stage needs a routing table and no
frame needs a destination field.

Source of the numbering: [common/stage.h](common/stage.h).

```
CLOVER_STAGE_STRIDE = 120        one stride per service
CLOVER_STAGE_LAYERS = 92

    1 ..   120   server, on the way in: tokenise, embed, layer 0
  121 .. 11160   layers 1..92, 120 apiece
11161 .. 11280   server tail: final aggregation and normalisation, head, next token
```

`clover_stage_owner`, `clover_stage_local` and `clover_stage_first` are integer
arithmetic on the stride. `clover_stage_hands_over(stage)` is true exactly when the next
stage belongs to a different service, which is where a frame has to be sent rather than
looped locally.

**The chain repeats per token.** `clover_stage_next(CLOVER_STAGE_LAST, more_tokens)`
returns `clover_stage_first(0)` — the last stage of the tail wraps back to the server,
and the whole 11,280-stage walk runs again for the next token. It returns 0 only when
there are no more tokens.

## The shape

```
caller ──► server ──► layer 1 ──► layer 2 ──► … ──► layer 92 ──► server tail ──► caller
 (token     (tokenise,                                          (final normalisation,
  ids)       embed, layer 0)                                     vocabulary head, decode)
```

The server owns both edges. Text to token ids, token id to embedding row, the final
vocabulary projection and the choice of next token all live in
[server/ends.h](server/ends.h). Layer 0 lives with the server rather than as a pod
because it is unlike every other layer: no MLA, no routed experts, no router. Each of
the other 92 layers is a pod holding its own trunk and its own 896 experts, and computes
one layer and nothing else.

| stage | holds | size |
|---|---|---|
| server | tokenizer, vocabulary, `trunk-0`, `trunk-0-qkv`, `leaves.json` | 4.432 GB |
| layer pod | `root-N` (896 experts), `trunk-N`, `operators/qkv-all/layer-N` | 15.512 GB |

Sizes as recorded in [system-architecture.md](system-architecture.md).

## Stages 1..120 — server, on the way in

The server owns this stride and performs the work, but note a discrepancy worth knowing
before relying on the numbers: `server/server.c` contains no reference to
`CLOVER_STAGE_*` and no `stage ==` machine. The 120 numbers are **reserved** for the
server in the numbering scheme; the server does not currently step through them the way
a layer pod does. What it does implement, as ordinary functions:

| step | function in `server.c` | work |
|---|---|---|
| text to ids | tokenizer in `ends.h` | prompt text becomes token ids |
| **vector mapping** | embedding row lookup in `ends.h` | each token id becomes one 7,168-float row |
| layer 0 | `server_qkv` | Q, K, V projections |
| layer 0 | `server_decay` | KDA decay terms |
| layer 0 | `server_update_attention` | recurrent state update |
| layer 0 | `server_attention_output` | attention output projection |
| layer 0 | `server_aggregate` | fold over snapshots and residual |
| layer 0 | `server_dense` + `server_dense_activation` | **dense** FFN, not MoE |

Layer 0 is KDA with a dense feed-forward. It has no router and no experts, which is why
it is not a pod.

## Stages 121..11160 — layers 1 to 92

Layer *N* owns stages `120 + (N-1)*120 + 1` through `120 + N*120`. Layer 1 is 121..240;
layer 46 is 5521..5640; layer 92 is 11041..11160.

Of the 92 pods, **24 are MLA** and 68 are KDA. Verified by counting
`#define TRANSFORMER_MLA 1` across the pod sources: layers 3, 7, 11, … 91 (every fourth)
plus layer 92. The two kinds differ only in stages 6..9; everything from stage 20 on is
identical.

### The 120 stages of a layer

Local numbering 1..120 within the pod, from `transformer_step`.

| local | work |
|---|---|
| 1 | copy input into `incoming` |
| 2 | copy input into `residual` |
| 3 | aggregate, fold 37 — blend residual with accumulated snapshots |
| 4 | **if `LAYER % 12 == 0`**, append this layer's input to the snapshot set |
| 5 | normalise the aggregate into `normalized` |
| **MLA layers** | |
| 6 | the entire attention step, then jump to 20 |
| **KDA layers** | |
| 6 | `transformer_qkv` |
| 7 | `transformer_decay` |
| 8 | `transformer_update_attention` |
| 9 | `transformer_attention_output`, then jump to 20 |
| **both** | |
| 20 | **if `LAYER % 12 != 0`**, `residual = incoming + residual` |
| 21 | aggregate, fold 38 |
| 22 | normalise the aggregate into `postnorm` |
| 23 | finite check on `postnorm` |
| 24 | **route** — choose 16 experts of 896 and their weights |
| 25 | project 31: `postnorm` 7,168 → `latent` 3,584 |
| 26 | zero the mixture accumulator |
| 27 | `root_prefetch` all 16 chosen experts, then jump to 30 |
| 30..109 | **16 experts x 5 steps** (below) |
| 110 | normalise the mixture into `latent_norm` |
| 111 | project 32: `latent_norm` 3,584 → `routed` 7,168 |
| 112 | project 34: shared expert gate |
| 113 | project 35: shared expert up |
| 114 | shared expert activation |
| 115 | project 36: shared expert down |
| 116 | `residual += routed + shared_output` |
| 117 | finite check on the residual |
| 118 | copy residual to output |
| 119 | copy snapshots to output snapshots |
| 120 | `positions++`, retire the position |

The expert band, stages 30..109, is rank `(stage-30)/5` and step `(stage-30)%5`:

| step | work |
|---|---|
| 0 | `root_project` matrix 0 — expert gate, 3,584 → 3,072 |
| 1 | `root_project` matrix 1 — expert up, 3,584 → 3,072 |
| 2 | activation |
| 3 | `root_project` matrix 2 — expert down, 3,072 → 3,584 |
| 4 | accumulate into the mixture, weighted by the router |

**The asymmetry is the point.** Attention is one stage of 120. The experts are 80 of
120, and 48 of those 80 are expert matrix projections.

### Snapshots, and what repeats

Stage 4 pushes a snapshot only when `LAYER % 12 == 0`, and stage 20 then skips the
residual add, because the fold at stage 3 already carried it. So the snapshot set grows
by one every twelfth layer, and `TRANSFORMER_INPUT_SNAPSHOTS` rises with depth — layer 3
takes 1 snapshot in and emits 1. Stages 3 and 21 are the two points where a layer reads
the whole accumulated snapshot set rather than just its own input.

## Stages 11161..11280 — server tail

Implemented in `transformer-93/transformer-93.c` as a real stage machine.

| local | work |
|---|---|
| 1 | environment and finite checks on residual and snapshots |
| 2 | point the source list at each snapshot |
| 3 | point the last source at the residual |
| 4 | score every source |
| 5 | softmax over the scores |
| 6 | aggregate the sources by those weights |
| 7 | finite check |
| 8 | apply the final RMS normalisation |
| 9 | finite check, then jump to 110 |
| 110 | **vocabulary head** — project to 163,840 and pick the token |
| 111 | token id to word bytes |
| 112 | append the word to the response text |
| 113 | `positions++`; if this is the last token go to 118, else 120 |
| 118 | build the JSON response body — last token only |
| 119 | deliver to the caller's callback — last token only |
| 120 | retire |

Stages 118 and 119 run only on the final token, which is exactly where
`clover_stage_next` stops looping.

## Where the time actually goes

Measured on k3 (Ryzen 9 7950X3D, 16 cores / 32 threads, 124 GB, RAID1 of two KIOXIA
CD8), layer 3, the 5-position `france` prompt, through the reference harness. All runs
verified against the independent reference.

| threads | layer 3 wall |
|---|---|
| 1 | 2.49 s |
| 4 | 0.73 s |
| 8 | 0.44 s |
| 16 | **0.30 s** |
| 32 | 0.31 s |

**Sixteen threads is the optimum for a layer running alone**, and 32 is slightly worse —
the second SMT thread per core does not help this work. This does not contradict
[HOSTING.md](HOSTING.md), which records 4 as the knee: that table is a *resident fleet
serving 1 to 32 concurrent requests*, where four threads per request times sixteen
requests already saturates the machine. One layer alone and ninety-two layers sharing a
box are different regimes, and the thread count belongs to the deployment shape.

Expert arrival is not the bottleneck at this size. Reading the 16 chosen experts of one
layer is 281 MB, and it lands in 33 ms:

| request shape | pass 1 | pass 2 |
|---|---|---|
| mmap + `posix_madvise(WILLNEED)`, fault on touch — **current** | 8.58 GB/s | 8.06 GB/s |
| one `pread` per expert, threads over experts | 7.02 | 7.18 |
| chunked `pread`, more requests in flight | 7.02 | 6.86 |

Page cache dropped before each arm, disjoint expert ranges. The path the code already
uses is the fastest of the three, so there is no gain available from restructuring the
expert reads on this hardware.

## Attention at long context

The MLA cache stores K, V and the positional tail interleaved per position, 98,560 bytes
per position per layer. Measured with [tansformers/mla-stride.c](tansformers/mla-stride.c),
which isolates the attention step at real sizes; all four arms produce identical
checksums.

At 1,048,576 positions — 103.35 GB for one layer:

| arm | time | GB/s |
|---|---|---|
| position-major, serial | 62.40 s | 1.66 |
| position-major, parallel | 5.16 s | 20.03 |
| head-major, serial | 12.51 s | 8.26 |
| head-major, parallel | **2.62 s** | 39.45 |

Head-major parallel reaches 39.45 GB/s against a measured RAM ceiling of 42.8 GB/s on
this box, so it is bus-bound and close to the floor for this data shape.

That is a kernel in isolation. Driven through a real pod by
[tansformers/pod-context.c](tansformers/pod-context.c), with the layer's 15.72 GB of
experts pinned so only the attention walk moves, output bit-identical in every pair:

| positions | original | head-major | whole layer |
|---|---|---|---|
| 64 | 31.265 ms/position | 30.864 | 1.01x |
| 1,024 | 40.035 | 32.106 | 1.25x |
| 4,096 | 71.546 | 35.245 | **2.03x** |

Experts cost about 30.9 ms per position whatever the length, so subtracting that floor
puts the attention component at 40.6 ms against 4.3 ms at 4,096 — **9.4x**. Below about
a thousand positions the change is noise, because the cache still fits in L3. At 4,096
the old layout spends 57% of the layer in attention; the new one spends 12%.

Both changes are now in every pod, applied by
[tansformers/port-head-major-mla.mjs](tansformers/port-head-major-mla.mjs). All 24 MLA
layers verify against the reference for france and japan.

The remaining factor is not bandwidth but volume: the latent form is 2,304 bytes per
position per layer against 98,560 expanded, **42.8x fewer bytes**. That is the next
question, and it needs the absorb matrices, which have not been read.

## What this does not cover

- The server's 120 stages are reserved but not implemented as a stage machine; the entry
  path is ordinary function calls.
- **The full campaign passes only with `CLOVER_SOFTMAX=1`.** `transformer_hardmax`
  defaults its owner list to the string `"46"` with both folds enabled, so layer 46 alone
  runs winner-takes-all instead of a softmax blend at stages 3 and 21. That changes its
  aggregate, so `postnorm` changes, so the router picks different experts, so
  `all` aborts at layer 46 on the route assertion. With `CLOVER_SOFTMAX=1`, layer 46
  verifies and all 91 layers pass. Measured cost of the default on layer 46, 256
  positions, four runs each: hardmax 132.21 ms/position mean against softmax 134.22, but
  softmax's best run is 132.145, so the saving is at most about 1.5% and possibly none.
  One layer of 93, so on the order of 0.02% of model time.
- Timings above are layer 3 at 5 positions and at up to 4,096 positions through a real
  pod, plus the isolated attention kernel at up to 1M. No layer has run end to end at
  1M, because the expert cost per position makes that impractical on one box.
- The 1M attention figures are a kernel in isolation, not a layer in the pipeline.
- The pod measurements pin the experts and give the layer the whole machine. A pod under
  real traffic shares it.
- The 3600 MT/s DIMM configuration against a 4800 rating has never been investigated.
