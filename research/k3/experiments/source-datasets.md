# Actual source datasets on AX102

Inspected on 2026-09-29 from the data, not inferred from documentation:
all 96 safetensors headers in `/root/k3model`, the actual packed-trunk index,
and bounded payload samples from checkpoint and trunk files.
The [census report](source-data-inspection-results.json) preserves every shard's
header hash, category totals, all layer/expert ownership counts, byte ranges,
and sample hashes. The [inspection script](source-data-inspection.py) is read-only
with respect to those datasets.

## Physical source groups

Decimal GB/TB below describe tensor payload bytes, not runtime RAM.

| Source dataset | Actual contents | Tensor payload |
|---|---|---:|
| Input embedding table | One BF16 matrix, 163840 x 7168 | 2348810240 bytes (2.349 GB) |
| Output head table | A separate BF16 matrix, 163840 x 7168 | 2348810240 bytes (2.349 GB) |
| Global output parameters | model.norm, output_attn_res_norm, output_attn_res_proj | 43008 bytes |
| Layer trunk weights, in the checkpoint | All non-routed-expert tensors of layers 0 through 92, including shared experts | 108811877376 bytes (108.812 GB) |
| Routed expert weights | 896 experts for each of layers 1 through 92; six stored tensors per expert | 1446456066048 bytes (1.446 TB) |
| Multimodal projector | Three BF16 tensors under mm_projector | 92289024 bytes (0.092 GB) |
| Vision tower | 165 BF16 tensors under vision_tower | 802428928 bytes (0.802 GB) |

These seven groups partition all 497220 tensor records and their
1560860324864 payload bytes. The 96 shard files total 1560936091448 bytes;
the remaining 75766584 bytes are their length fields and headers. This is the
checkpoint whose directory appears as approximately 1.5T in the server listing.
This inventory does not enumerate tokenizer/config files as learned tensor data.

The input/head tables and global output parameters are in shard 94, the
multimodal projector in shard 95, and the vision tower in shard 96. In this
actual checkpoint, each language layer occupies its corresponding shard among
1 through 93. Shard numbers are storage containers, not extra model layers.

## One trunk file, separate layer weights

The packed `/root/k3trunk_i8/trunk.bin` is **54468222976 bytes (54.468 GB)**.
It contains a packed representation of the 2455 non-routed-expert tensors
already present in the checkpoint, not an additional set of learned parameters.
Its index records 506 F32 tensors, 512 BF16 tensors, and 1437 I8R tensors.
All its tensor names were found in the checkpoint; no routed-expert tensor
name occurs in the trunk, and no non-routed language-layer tensor is missing.
This census does not independently verify the int8 quantization conversion.

The actual trunk index divides that one file into **93 distinct layer records**:

| Layer | Start byte in trunk.bin | Reserved layer bytes |
|---|---:|---:|
| 0 | 0 | 1171529728 |
| 1 | 1171529728 | 634626048 |
| 2 | 1806155776 | 634626048 |
| 92 | 54045634560 | 422588416 |

Each record contains that layer's named tensors, with their own local offset,
shape, dtype and byte length. A tensor's absolute address is:

`layer.file_off + tensor.off`

Thus the file is not one common trunk weight set applied 92 times. Each layer
has its own parameter bytes; the index tells the loader where each one begins.
The distinction is **93 total layers, of which 92 have routed experts**:
layer 0 has a dense MLP, while layers 1 through 92 own routed expert sets.

## Which experts belong to which layer?

The actual tensor names include both layer ID and expert ID, for example the
pattern `language_model.model.layers.L.block_sparse_moe.experts.E.w1.weight_packed`.
Every expert has w1, w3 and w2, each with packed-code and scale tensors.
The census found all six tensors for every one of the **82432 layer/expert
pairs (92 x 896)**. Every expert-bearing layer's IDs run from 0 through 895.

Layer 1/expert 0 and layer 2/expert 0 are separately stored parameters. The
shared expert pool on disk is a collection indexed by `(layer, expert)`, not
896 global experts whose weights are reused across all layers. A layer's router
selects from that layer's section of the pool.

**Shared experts are different from routed experts.** Their gate/up/down tensors
are included in each expert-bearing layer's trunk record. The checkpoint contains
276 such tensors (92 x 3), totaling 24310185984 bytes. That total is a subset of
the 108811877376 trunk-source bytes above, not an additional dataset to add again.

## What was and was not checked

Read all embedded tensor headers (75766584 bytes including length fields),
validated non-overlapping in-bounds ranges and reconciled all file sizes.
Read 12 checkpoint tensor boundary samples and two trunk tensor boundary
samples per layer; unconverted sampled trunk tensors matched their checkpoint
boundary hashes. No full 1.5 TB payload scan or whole-file weight hash was done.
No model execution, source-data mutation, or checkpoint conversion was performed.

This corrects the earlier [data-role inventory](model-data-inventory.md), which
answered a different question. The source datasets above, not activation types
or diagnostics, are the relevant starting point for the user's question.

Subsequently, the complete input embedding payload was scanned and verified
in two passes. Its [full value census](input-table-values/README.md) records
6658 distinct scalar values across all1174405120 entries. That later full scan
covers the input table only; it does not broaden the rest of this header census.

The output head was later fully scanned as well: 6590 distinct values across
1174405120 entries. Its [fruit dataset and integration](fruit/README.md) retain
the exact output weights in a 1662954740-byte lossless file. The seed+fruit
model passed both recorded prompts. These later full scans cover the two
tables, not all trunk/expert/vision payloads in this source census.