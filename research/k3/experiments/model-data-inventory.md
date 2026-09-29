# Data in the current model

The user clarified that they meant physical source datasets, not this
classification of inference roles and numeric types. See the
[actual server dataset census](source-datasets.md), including vision/projector
weights, trunk layer offsets, and layer-specific expert ownership.

This describes inference data in the inspected Clover K3 implementation, not
the training corpus. Sources: the [model equation](../model/k3-model-equation.md)
and [final integrated source](reverse-integration/step1/candidate.c). No new
representation experiment or model run was performed for this inventory.

A data object's **role**, **numeric format**, **shape**, **lifetime**, and
**origin** are different properties. The following categories organize those
properties; they are not additional equation items or additive memory totals.

## What the data is for

| Kind | Actual examples | Where it comes from / how long it lasts |
|---|---|---|
| Input | Token IDs, position indices, prefix length | Supplied for the current prompt or continuation |
| Model constants and derived lookup tables | Dimensions, layer types, epsilon values, activation constants, E2M1 codebook, DQ/DQd/DQ2 tables | Fixed by the implementation or generated from the decode equation; the lookup tables are not learned parameter tables |
| Learned weight matrices | Embeddings, output head, attention projections, dense MLP, shared experts, routed experts, router projection | Fixed checkpoint parameters; which expert weights are used depends on the input |
| Weight scales and smaller parameter arrays | MXFP4 group scales, int8 row scales, normalization weights, router correction bias, convolution taps, A_log and dt_bias | Checkpoint/packed-weight representation data; quantization scales are associated with weights, not runtime routing probabilities |
| Runtime vectors and intermediate results | Residuals, normalized inputs, q/k/v projections, expert input latent, gate/up/SiTU outputs, attention outputs | Computed from the current input and preceding results, mostly float32 |
| Scores, decisions and mixing coefficients | Router scores, selected expert IDs and weights, attention scores/softmax coefficients, snapshot coefficients, alpha/beta/gate values | Input-dependent results that determine selection or how vectors are combined |
| Carried state and caches | Residual stream, layer snapshots, KDA recurrent matrices, convolution history, MLA keys/values/rope-slot values, prefix state | Carries information across layers or token positions; cache lifetimes differ by object |
| Outputs | Final normalized vector, vocabulary logits, selected next-token ID | Products of the current forward call; a logit is a score, not already a probability |
| Addressing and runtime metadata | Tensor shapes/dtypes, layer/expert/part IDs, file paths, offsets, byte lengths, index records, route-cache headers, buffer pointers and allocation sizes | Describes where data lives and which data to use; knowing an expert ID does not load its weights |
| Experimental representation descriptors | Source references and coefficients, signs/exponents/odd codes, bit masks, histograms, arrangement/joint ranks, scale headers/exceptions, compressed byte streams | Alternative encodings of the values above; not new learned knowledge or independent information savings |
| Verification and measurement evidence | Reference copies, original-versus-candidate values, traces, checksums, elapsed time, CPU/fault/I/O counters, peak RSS | Test/measurement artifacts; not required mathematical model parameters, but retained in the current correctness harness |

The categories can overlap in role: a residual is a computed vector and carried
state, and an expert ID can be a routing result and later cache metadata. This
does not mean there are two independent copies of its information.

## How the numbers are represented

| Format | Meaning in this implementation |
|---|---|
| MXFP4 | Expert weights:4-bit E2M1 codes, two per byte, with one8-bit E8M0 scale code per32 weights. Decode behavior, including special cases, follows `dq_init`. |
| I8R | Signed int8 weight entries plus a float32 scale per row. The C layout starts each row with4 scale bytes, followed by its int8 entries. |
| BF16 |16-bit floating-point learned values, including embedding/head tables and many norm parameters. Widened to float32 by bit placement, not interpreted as ordinary integer magnitudes. BF16 is not IEEE FP16. |
| FP32 | Most activation vectors, carried numeric state, logits, several learned parameter arrays and decoded weights. Values may be positive, negative, zero, fractional or greater than one in magnitude. |
| FP64 | Double-precision accumulation and temporary scalars in selected projections, norms and attention calculations; also widened decode tables. It is not the storage format of every activation. |
| Integer/byte fields | Token/expert IDs, counts, positions, flags, offsets, sizes and serialized headers; the inspected code uses16/32/64-bit fields and byte arrays where appropriate. |
| Exact large integers |128-bit arrangement/joint arithmetic and GMP integers in the experimental codecs. These represent ranks or exact dyadic integers, not a new neural tensor dtype. |
| Compressed byte streams | Zlib payloads and serialized descriptors. A stream's byte count is a storage property, not the vector's mathematical dimension. |

The same value can pass through several formats. An expert code plus scale
produces a float32 weight, which is widened to double for accumulation. A BF16
parameter becomes float32 at use. Those are stages of representation, not
different learned values.

## Examples of shapes

| Object | Shape in the current model |
|---|---|
| Main hidden/residual vector |7168 values per position |
| Routed expert input/output latent |3584 values per position |
| Routed expert intermediate activation |3072 values |
| Shared-expert intermediate activation |6144 values |
| Layer0 dense intermediate activation |33792 values |
| KDA projected q/k/v or gate vector |96 heads x128 =12288 values |
| MLA normalized query latent |1536 values |
| MLA expanded query |96 heads x192 =18432 values |
| MLA normalized KV latent and shared rope slot |512 plus64 values before expansion |
| Current expanded MLA KV cache per position/layer |96x128 keys +96x128 values +64 shared rope-slot values |
| KDA recurrent state per active layer |96 matrices, each128x128 |
| KDA convolution history |3 paths x12288 channels x3 preceding values |
| Router |896 scores;16 selected experts and16 mixing weights per MoE layer/position |
| Output |163840 vocabulary logits and one selected token ID |

These are logical shapes, not a claim that every layer's full state is resident
simultaneously. This runner reuses working buffers between layers; prefix
serialization carries the per-layer state when enabled.

For scale, one7168-value float32 vector is28672bytes (28KiB). The active-layer
KDA state is6291456bytes (6MiB), while its convolution history is442368bytes
(432KiB). One expanded MLA KV position is98560bytes (96.25KiB). These individual
payload calculations exclude allocation, indexing and container overhead and
are not a decomposition of measured peak RSS.

A32-weight expert group is16 code bytes plus one scale byte:17bytes. The same
32 decoded float32 weights would occupy128bytes. That quantized representation
already exists; our alternative descriptors must be compared with17bytes, not
mistakenly credited with the original32-bit-to-4-bit conversion.

## What this means for equation work

- Fixed constants and derived lookup tables can sometimes be regenerated
  directly because their defining rule is already known.
- Learned values require preserving their information, even if a different
  lossless code stores it more efficiently.
- Runtime vectors may be represented by their generating inputs and exact
  operation sequence, but those inputs, rounding steps and lifetimes remain.
- Carried state requires enough history/checkpoints to reproduce what later
  positions consume; a compact-looking real-arithmetic identity is not enough.
- Indexes, route IDs, weight bytes, buffers and diagnostic copies must be
  accounted for separately when measuring memory or storage.

The7168-component vector is therefore one member of a much larger data
inventory. It is not the format, width or value range of all model data.