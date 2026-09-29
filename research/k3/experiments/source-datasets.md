# Actual source datasets on AX102

Current dataset names (2026-09-30): **seed** is the input embedding dataset,
**fruit** is the output-head dataset, and **root** is the compact routed-expert
dataset. The compiled scalar, map and map-list tables are called **C constants**,
not root. Root currently covers layer1; the name does not imply conversion of
layers2-92. Earlier references below to a root value table describe the previous
format, not the current naming. Verified filenames, format identifiers and
runtime settings are unchanged by this naming decision.

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

## First routed expert: layer 1, expert 0

On 2026-09-29, all six tensors of this expert were read from the actual second
checkpoint shard. The [inspection report](expert0-inspection-results.json)
records tensor offsets, full payload hashes, complete code/scale histograms,
and the first decoded 32-weight group from each matrix. The
[inspection script](expert0-inspection.py) leaves source data unchanged.

| Matrix | Role | Logical weight shape | Packed-code bytes | Scale bytes |
|---|---|---|---:|---:|
| w1 | Gate | 3072 x 3584 | 5505024 | 344064 |
| w3 | Up | 3072 x 3584 | 5505024 | 344064 |
| w2 | Down | 3584 x 3072 | 5505024 | 344064 |

The six stored arrays are U8 containers: the packed arrays hold two 4-bit codes
per byte; the scale arrays hold one byte per 32 weights. Each matrix has
11010048 logical weights and 344064 scale groups. The whole expert contains
33030144 weights, 1032192 groups, 16515072 code bytes and 1032192 scale bytes:
**17547264 bytes (17.547264 MB, or 16.734375 MiB)**. Fully expanded float32
weights alone would occupy 132120576 bytes; expansion was not performed.

All 16 code patterns occur in every matrix. Gate/up scale bytes cover 112-122;
down scale bytes cover 119-122. These are scale codes, not final weights.

The actual first w1 group has packed bytes beginning `[22,185,206,35]`, which
decode low nibble first to codes `[6,1,9,11,14,12,3,2]`. Its scale byte is 121,
giving a multiplier of 0.015625 (1/64). The first eight resulting weights are:

```text
[0.0625, 0.0078125, -0.0078125, -0.0234375,
 -0.0625, -0.03125, 0.0234375, 0.015625]
```

All 17547264 stored bytes were read and counted. Only the first 32 weights per
matrix were decoded as examples; this was not an expert forward run or a full
distinct decoded-value census. Other experts were not read in this inspection.

## First expert as an executable data tree

The user subsequently authorized a tree representation for this same expert.
The [builder and reader](expert-tree.py) now store an expert root, three matrix
branches and 9728 row descriptions. Each row describes how its code bytes and
scales are recovered. The regular group and leaf positions are computed from
the matrix shape instead of storing a separate record for every weight.

```text
Root: trunk layer 1 / expert 0
	w1: 3072 rows -> 112 groups per row -> 32 weights per group
	w3: 3072 rows -> 112 groups per row -> 32 weights per group
	w2: 3584 rows ->  96 groups per row -> 32 weights per group
```

This gives 1032192 implicit group nodes and 33030144 weight leaves. Descending
to a group reconstructs its exact scale and four binary masks; their bits feed
the existing MXFP4 weight equation. Evaluating upward through a row multiplies
the generated weights by the input and accumulates them in the original
sixteen-double-lane order. The nonlinear joins between matrices remain described
by the expert equation, but a full SiTU/expert forward is not implemented in
this storage reader.

### Actual file size

| Component | Bytes |
|---|---:|
| Original expert codes and scales | 17547264 |
| Tree fixed header | 64 |
| Compressed root and row descriptions | 443669 |
| Code payloads | 16334839 |
| Scale descriptors | 121345 |
| Complete expert tree | **16899917** |
| Net bytes saved | **647347 (3.69%)** |

For row code payloads, 8997 rows chose zlib-compressed packed codes and 731 kept
their raw packed codes; compressed four-plane layout won no rows. Scale rows
use the earlier tested base/usual-offset/exception codec. Four masks are generated
when a group is read, not stored as an additional duplicate payload. The root's
compressed JSON contains shapes, offsets, codecs, scale lengths and decoded-row
hashes. All file overhead is included above; companion audit files and software
are not. This is a measured representation of one expert, not a compression
optimum or a result for the remaining experts.

### Read and evaluate

On AX102:

```sh
python3 /opt/clover-k3/expert-tree-20260929-a/expert-tree.py read \
	/opt/clover-k3/expert-tree-20260929-a/data/expert.tree \
	--matrix w1 --row 0 --group 0
```

The Python `ExpertTree` API provides `read_row`, `group`, `weight` and
`project_row`. For example, `weight("w1", 0, 0)` returns **0.0625** from the
actual first weight; `group("w1", 0, 0)` exposes all four masks, the scale,
32 codes and exact float32 bit patterns. A group or leaf lookup decodes its
own row, not the entire expert. The reader loads the root index and needs the
tree file, reader software, NumPy and the provided scale codec; it does not
need the original checkpoint to reconstruct weights.

### Verification and evidence

- [Build result](expert-tree/results.json): all six stored tensors recovered
	byte-for-byte and all 33030144 generated float32 weights checked against an
	independent codebook decode, including signed zero. Tensor hashes match the
	earlier actual-expert inspection.
- [Root description](expert-tree/root.json): inspect the exact matrix branches
	and row-node metadata in readable form.
- [Row audit](expert-tree/rows.jsonl): original bytes, candidate code-layout
	sizes, selected codec and scale descriptor bytes for every row.
- [Navigation and evaluation result](expert-tree/navigation-results.json):
	27 groups, 54 individual leaf reads and nine complete row projections checked
	against directly read checkpoint rows by a
	[separate checker](expert-tree-check.py).

Controls also covered 804 scale roundtrips, all 4096 code/scale combinations,
code layouts, tree corruption and invalid paths. The nine projection inputs
were controlled vectors, not captured inference inputs. No full model or
nonlinear expert forward, latency, peak-RAM benefit or other expert coverage
is claimed. The original source file remains unchanged.

The binary remains on AX102 at
`/opt/clover-k3/expert-tree-20260929-a/data/expert.tree`; code, root and reports
are preserved here. Its SHA256 is
`cd36f613176a6312dc2724befe86b5d3171f5d0c0d810443811a7d621ebcaf58`.

## Layer 1 routed experts only

The user narrowed the next exploration to layer 1's 896 routed experts,
explicitly excluding trunk and everything else. The
[expert-only scanner](layer1-experts.py) therefore reads payloads only from
`language_model.model.layers.1.block_sparse_moe.experts.0..895` in the actual
second checkpoint shard. Parsing the shard's shared header locates those
ranges; no trunk, shared-expert or other-layer tensor payload was read.
No layer or model execution was performed.

The [full branch map](layer1-expert-branches.json) has layer 1's routed-expert
pool as its root and all 896 experts as children. Each expert has the same
three-matrix structure, with distinct stored parameter records:

```text
Layer 1: routed-expert branch
	expert 0
		w1 (gate): 3072 rows x 112 groups x 32 weights
		w3 (up):   3072 rows x 112 groups x 32 weights
		w2 (down): 3584 rows x  96 groups x 32 weights
	expert 1
		same three-matrix shape, its own code/scale payloads
	...
	expert 895
		same three-matrix shape, its own code/scale payloads
```

Each matrix has two stored tensors: packed 4-bit codes and scale bytes.
One scale byte applies to a group of 32 codes. All six tensors were found and
read in full for every expert; all 896 have the expected shape and size.

| Expert-pool quantity | Verified count or bytes |
|---|---:|
| Experts | 896 |
| Matrices | 2688 |
| Stored tensors | 5376 |
| Logical matrix rows | 8716288 |
| Groups of 32 weights | 924844032 |
| Logical weights | 29595009024 |
| Packed code bytes | 14797504512 |
| Scale bytes | 924844032 |
| Total expert payload bytes read | **15722348544 (15.722 GB)** |

GB is decimal; byte totals exclude the shared shard header. Each expert is
17547264 bytes. These figures contain no trunk weights and describe stored
payload, not process RAM or a new encoding size.

### What the data showed

| Matrix | Code patterns observed | Scale-byte range | Distinct scale bytes |
|---|---|---|---:|
| w1 | All 16, in every expert | 110-123 | 14 |
| w3 | All 16, in every expert | 110-122 | 13 |
| w2 | All 16, in every expert | 119-122 | 4 |

No same-kind, same-shape whole-tensor hash repeated. No complete expert payload
hash repeated either. Thus matching branch structure does not mean matching
parameter values. This does not rule out shared smaller patterns or other
compression opportunities: those relationships were not tested.

The [census report](layer1-experts-results.json) contains aggregate code/scale
histograms. The branch map contains each expert's own histograms, tensor names,
shapes, offsets, byte lengths and SHA256 hashes. All per-expert frequencies
reconcile to the aggregate, all ranges are non-overlapping and in bounds, and
expert 0's tensor hashes match the earlier full inspection.

All **15722348544 expert payload bytes** were scanned. Separate code and scale
histograms do not give an exact count of decoded weight values or establish
spatial/cross-expert correlation. This map references the existing tensors; it
is not a self-contained compressed layer container and does not replace the
verified single-expert tree. The source file remains unchanged.

Artifacts are preserved on AX102 under
`/opt/clover-k3/layer1-experts-only-20260929-a/data` and linked locally above.
The ordered expert-payload SHA256 is
`cfeb895c80fa5b34fb219c3e3a364d5ae9cbbeee414cb35e0651d38d9ec42516`;
its order is expert ID, matrix w1/w3/w2, then packed codes/scales.

## Binary tree with inherited common information

The user next requested a root with left/right children down to leaves, with
shared information lifted toward the root. This structure is now constructed
for **layer 1's routed experts only**. It is a source-backed hierarchy, not a
replacement for the existing 15.722 GB of expert weight payloads.

The [binary tree](layer1-binary-tree.json) starts as follows:

```text
Root: layer 1 routed experts
	Shared facts: source file, U8 storage, 4-bit codes, 8-bit scales,
								32 weights/group, common matrix payload sizes
	Shared shape templates: gate/up and down
	Most frequent scale: 121 (with exceptions, not a universal value)
	Most frequent code: 1 (with exceptions, not a universal value)
	|
	+-- left: experts 0..447
	|     +-- left:  experts 0..223
	|     +-- right: experts 224..447
	|
	+-- right: experts 448..895
				+-- left:  experts 448..671
				+-- right: experts 672..895
```

Each expert range splits until one expert is identified. Its three matrices
also use a binary split, not a three-way branch:

```text
One expert
	+-- left: gate/up pair
	|     +-- left:  w1
	|     +-- right: w3
	+-- right: w2

One matrix
	row range -> left/right halves -> one row
	group range -> left/right halves -> one 32-weight group
	coordinate range -> left/right halves -> one weight leaf
```

The stored index has **5375 nodes: 2687 binary branches and 2688 matrix nodes**.
Below each matrix, row/group/coordinate nodes are generated from the root's
shape templates when traversed. The tree can therefore address all
29595009024 logical weight leaves without writing billions of node objects.

### What moves upward

At every stored branch, the builder compares its children's shared fields.
Fields identical on both sides move to their parent and are removed from the
children. Repeating this bottom-up places an identical fact at its highest
valid ancestor. Fields that differ remain in descendant nodes. An independent
check confirmed that no equal sibling field was left unhoisted.

The common matrix schemas are defined once at the root: w1/w3 reference the
3072x3584 gate/up shape, while w2 references the 3584x3072 down shape. The
different tensor offsets, hashes and frequency counts remain associated with
their actual matrix records. Nothing merges different learned parameter values.

Most-frequent values are separately labeled summaries:

| Root summary | Observed occurrences | Total |
|---|---:|---:|
| Scale byte 121 | 737890075 | 924844032 scale bytes |
| Code 1 | 3255595128 | 29595009024 weight codes |

Scale 121 is also the mode of every matrix in this layer's expert pool, so that
mode fact is inherited from the root. It is not the actual scale at every leaf.
For example, expert 0/w1/row 0/coordinate 224 uses scale 120 and code 12,
producing -0.015625. This exception and the negative zero at coordinate 9
were checked directly against the actual expert bytes.

**Defaults do not replace exceptions.** This structural version still reads
each requested code and scale from their verified source locations. It does
not yet store a separate exception-only weight payload. No data-storage saving
is inferred from the smaller number of explicit nodes.

### Traverse the tree

The [builder and reader](layer1-binary-tree.py) provides `matrix`, `locate` and
`read_weight`. `locate` returns the complete left/right path and inherited
fields without reading model payloads. `read_weight` follows that path, reads
one packed-code byte and one scale byte, and evaluates the existing MXFP4
decoding equation. The original expert dataset is required for that operation.

On AX102:

```sh
python3 /opt/clover-k3/layer1-binary-tree-20260929-a/layer1-binary-tree.py read \
	/opt/clover-k3/layer1-binary-tree-20260929-a/data/binary-tree.json \
	--expert 0 --matrix w1 --row 0 --coordinate 224
```

### Verified scope

- [Build result](layer1-binary-tree-results.json): all 896 experts, all 2688
	matrix records and all 5376 stored-tensor descriptors reconstruct exactly
	from inherited fields; 8064 paths checked across every expert.
- [Independent checks](layer1-binary-tree-checks.json): every stored internal
	node has exactly two children; partitions are exact; all inherited shapes,
	domains, modes, histograms and source offsets match the prior full census.
	Six actual weight leaves match independent source reads, including scale
	exceptions and signed zero.
- [Example paths](layer1-binary-tree-paths.json): complete left/right routes
	to actual weights in experts 0, 447, 448 and 895.
- [Independent checker](layer1-binary-tree-check.py): verification logic kept
	separate from the builder's own roundtrip checks.

The readable JSON index occupies 6783080 bytes, including histograms and
provenance; it does not contain the expert weight payload. This cycle did not
scan the complete payload again, compress it, or execute a model. The prior
expert-only full scan supplies the tensor hashes. No trunk, shared-expert or
other-layer payloads were accessed. Original source data was not changed.

## Interactive tree explorer

The [web viewer](tree-viewer/index.html) displays the verified binary hierarchy
as a zoomable graph. It is a research tool separate from the public site and
does not load model weights or contact AX102. D3 7.9.0 and Lucide 0.468.0 are
pinned local copies with their license files beside the viewer.

Start it from the repository root with Node.js:

```sh
node research/k3/experiments/tree-viewer/server.mjs
```

The server binds only to 127.0.0.1, uses port8770 or the next available port
through8780, and prints the URL. The viewer needs HTTP to fetch its JSON data;
opening the HTML directly as a file is not sufficient. No npm install is needed.

- **Full tree** renders all5375 stored nodes, covering all896 experts and
	2688matrix endpoints. Zoom and pan to inspect the radial overview.
- **Branch** displays a readable left/right neighborhood. Select a graph node
	to inspect it; double-click or use Focus to descend. The inspector's left/right
	buttons and the breadcrumbs provide another navigation path.
- The Expert/Matrix/Row/Weight inputs locate a specific leaf. Binary row/group/
	coordinate descendants are generated as needed; billions of node objects are
	never materialized. The depth control changes the displayed neighborhood.
- The inspector separates fields stored at the selected node from inherited
	fields and names their origin. The root's frequent values are labeled
	non-universal; they are not substituted for actual weights.
- Six verified samples show actual values, including scale120 instead of121
	and negative zero. Other leaves display their exact packed-code/scale offsets
	and "Value not loaded". Local metadata alone cannot supply those weight bytes.
- A selected node can be exported as JSON. Its generated payload was tested,
	but completed file download was not confirmed by the integrated-browser tool.

At small zoom, collision-aware labeling keeps nodes visible while showing only
labels that fit. The inspector remains available for every selected node. The
mobile layout places the inspector below the graph. Keyboard focus on the canvas
supports parent/child/sibling arrows and Enter to focus a branch.

The [navigation tests](tree-viewer/tree-model.test.mjs) compare all measured
Python paths and offsets, every matrix's final weight address, root coverage,
unknown-value states and invalid inputs. Run them with:

```sh
node --test research/k3/experiments/tree-viewer/tree-model.test.mjs
```

[Browser checks](tree-viewer/browser-checks.json) cover desktop/mobile nonblank
canvas rendering,5375-node overview, clicks, zoom, lookup, verified values,
deep links and failed-load recovery. Screenshots and actual rendered-label
rectangles were checked; tooltip overflow and mobile label collisions found
during testing were repaired. This visualization does not change or further
verify the underlying checkpoint beyond the cited tree evidence.

### Actual stored-record inspector

The default view now shows the verified C-constant format described below.
The earlier root-table snapshot remains available under the Format selector;
the following partial-snapshot description applies only to that older option.

The [stored-record view](tree-viewer/records.html) is separate from the
source-backed tree. It displays an [actual byte snapshot](tree-viewer/records-data.json)
exported from the new layer-1 root/expert store, not illustrative records.
Use the same local server and open
`http://127.0.0.1:8770/tree-viewer/records.html`.

This snapshot contains the root row (ID1, 22 BF16 values, 46 bytes), root-up
and root-down (FK1 only, two bytes each), and 1407 complete matrix records
for experts0-468. These child tables have no additional universally common
values. All value cells show their exact decimal representation and two
little-endian bytes. Selecting a cell reveals its BF16 bits and byte offset.

Expert records show local values, palette offsets and sizes, scale-selector
lists, and the full block index. Row0/group0 exposes32 actual weights and
their packed codes, group-palette reference IDs, root/local cell destinations,
and stored block hex. For example, expert0/w1/weight0 follows code6 to
reference8 to root.value_9 =0.0625 (bytes `80 3d`). Weight9 preserves negative
zero (`00 80`). Expert24/w1 also demonstrates an actual local-value reference.

**This is a partial-build snapshot, not a completed-container claim.** The
payload bytes were read from the in-progress file; record boundaries were
reconstructed from the completed build audit and census. The final compressed
metadata had not yet been written. Only the first64-row block of each shown
matrix was decoded for this export; the display exposes its first32 weights.
Unavailable expert IDs are explicitly rejected. The snapshot does not update
automatically as the separate full-build/verification campaign progresses.

The [exporter](root-table-view.py) checks stored local values and palette maps,
and independently decodes each displayed group. The
[record tests](tree-viewer/records-model.test.mjs) verify every exported group
against actual nibble bytes and root/local cells, root file bytes, signed zero,
block continuity and invalid selections. Run with:

```sh
node --test research/k3/experiments/tree-viewer/records-model.test.mjs
```

Desktop1440px and mobile390/320px views were checked in the browser, including
root/local links, matrix changes, block records and unavailable-expert states.
Wide tables scroll within their own regions. The original tree remains linked
and unchanged apart from the record-view navigation link.

## Compiled constant palette: complete layer 1

The user requested moving the repeated scalar values into C constants instead
of storing them in root/local value cells. This is implemented as a separate
dataset and native reader. The earlier root-table store and source checkpoint
are preserved; no model or other layer was modified.

The [C palette](../expert-constant-palette.h) contains all66 exact BF16 values
measured across layer1's896 routed experts, as `static const uint16_t
k3_l1_value_bits[66]`. These are bit patterns, not rounded decimal literals.
Positive and negative zero are distinct. Compiled palette bytes were compared
against the entire verified value census, not just a sample.

The [native reader](expert-constant-store.c) reconstructs BF16 blocks or
float32 rows using that array. Its only runtime dataset is `experts.bin`:
no root, root-up, root-down, local scalar dictionary, original checkpoint,
or Python helper is needed by the C executable. The
[converter and verifier](expert-constant-store.py) require the prior dataset
and its completed verification report during construction and checking.

### What is stored

All integer fields are little-endian. This version is specific to layer1 and
its three matrix shapes; it does not imply that other layers have the same
value domain. A palette CRC in the header rejects a mismatched C palette.

| Region | Layout | Bytes |
|---|---|---:|
| Header | Magic `K3CONST1`; version/layer/counts/strides; palette/index CRC32; index/payload lengths; reserved | 64 |
| Matrix index | 2688 records, each uint64 map offset + uint32 map count/first block/block count/reserved | 64512 |
| Block index | 136192 records, each uint64 offset + uint32 size/codec/decoded CRC32/reserved | 3268608 |
| Group mappings | Each group palette has16 uint8 references into the compiled66-value array;255 means absent pair | 312400 |
| Compressed blocks | Existing grouped zlib blocks, unchanged byte-for-byte | 14194656836 |
| Scalar values in dataset | None | **0** |
| Dataset total | All header/index/payload bytes | **14198302420** |
| C palette | Counted once with the software | **132** |
| Dataset plus C palette | Excludes executable code and audit files | **14198302552** |

Matrix identity is derived from record index: `expert = index / 3`, with
remainder0/1/2 selecting w1/w3/w2. Each block holds64 rows. Stored offsets are
relative to the payload start. The raw grouped block is packed4-bit codes
followed by one selector byte per32 weights. The selector chooses a stored
16-entry group palette; the4-bit code selects its global constant reference.
Neither the original scale values nor per-matrix scalar values are stored.

For example, expert0/w1/row0/weight0 now follows code6, group palette9,
reference28, then `k3_l1_value_bits[28] = 0x3d80`, giving0.0625. Weight9
uses reference33, `0x8000`, preserving negative zero. The compressed code and
selector bytes did not change; the group references now address the C array.

### Measured difference

The [prior root-table result](root-table-results.json) completed full
checkpoint verification: all29595009024 float32 weights and all15722348544
original code/scale bytes matched. Its total was14201668545 bytes.

The [constant-store result](expert-constant-results.json) is3365993 bytes
smaller after counting the132-byte C palette. Attribute that difference to
two separate changes:

- Scalar deduplication:91432 local bytes +50 shared-table bytes removed,
	minus132 compiled bytes = **91350 bytes saved**.
- Binary index:6607763 compressed JSON bytes replaced by3333120 fixed binary
	bytes = **3274643 bytes saved**. The new index uses CRC32 per decoded block;
	the prior index carried SHA256 per block, so this is not identical metadata.

The new dataset plus palette is1524045992 bytes smaller than the original
quantized payload, **9.6935%**, almost all from the previously measured zlib
block compression. Moving constants into C did not remove expert arrangement
data or create a multi-gigabyte saving by itself. This is not a global optimum,
model performance measurement, or peak-RAM claim.

### Verification and use

- The converter remapped every group palette and compared every reconstructed
	block against the prior stored SHA256. Compressed block bytes were copied.
- The compiled C decoder streamed all29595009024 BF16 patterns, matching the
	full verified weight SHA256;60 native float32 row reads also matched exactly.
- [Native controls](expert-constant-controls.json) covered27 exact row reads
	including raw/zlib blocks and signed zero, plus15 rejected invalid addresses,
	headers, references, selectors, checksums and truncated files.
- CRC32 checks detect accidental corruption, not hostile modification. The
	external report pins dataset, palette, implementation and executable SHA256.
	This storage experiment is not integrated into model inference.

The binary remains at
`/opt/clover-k3/layer1-constant-store-20260929-a/data/experts.bin` on AX102.
Its SHA256 is
`ce9c018400905a2fe4fbb93b170b942a273cfb257a1c951c49c69841556ff0b0`.
The native executable is beside `data/`, named `reader`:

```sh
/opt/clover-k3/layer1-constant-store-20260929-a/reader row \
	/opt/clover-k3/layer1-constant-store-20260929-a/data/experts.bin \
	0 w1 0 > row.f32
```

`row` writes a little-endian float32 row; `palette` writes132 exact palette
bytes; `verify FILE` streams every weight as little-endian BF16. Building the
C source requires a C11 compiler and zlib headers/library; the experiment
used existing private zlib dependencies without installing system packages.

The default [records view](tree-viewer/records.html) now shows all2688 actual
binary matrix records and compiled constants, with no stored local cells.
Its [exporter](expert-constant-view.py) reads the completed binary index and
payload, and obtains palette bytes from the compiled executable.
[Snapshot tests](tree-viewer/constant-records.test.mjs) check all displayed
references, binary record fields, palette identity and agreement with every
weight from the older visual snapshot. The Format selector preserves access
to that earlier partial view; it is not evidence of the current build state.

### Actual remaining bytes inspected

After moving the66 scalar values to C, the user requested inspecting the
remaining dataset. The [read-only inspector](expert-remaining-data.py) checked
the actual full-file SHA256, all2688 matrix records, all136192 block records,
and every group-reference map. It separately decoded the first64-row block of
each matrix in expert0 and compared the first float32 row with the native
reader. This did not change the dataset or build another representation.

The [inspection report](expert-remaining-data-results.json) accounts for every
one of the14198302420 stored bytes:

| Remaining component | Actual bytes |
|---|---:|
| File header | 64 |
| Matrix index | 64512 |
| Block index | 3268608 |
| Group-reference maps | 312400 |
| Compressed code and selector blocks | 14194656836 |

**99.9743% is compressed code/selector data.** The compiled constants supply
the possible scalar values; the codes and selectors still specify which
value occurs at each weight position. Their separate compressed contributions
cannot be assigned from the joint zlib stream. Decoded code and selector sizes
are not additional bytes on disk.

Actual expert0/w2, block0 (rows0-63), starts at file byte13886395. Its94165
stored bytes expand to98304 packed-code bytes plus6144 group selectors. The
first group is:

```text
Packed code bytes: d1 53 dc 25 92 00 3d c3 91 aa 92 18 09 13 c1 8c
First eight codes:       [1, 13,  3,  5, 12, 13,  5,  2]
Group selector:         2
Selected constant IDs:  [22, 60, 25, 27, 59, 60, 27, 24]
```

That block's6144 selectors contain1006 copies of1,5109 copies of2, and29
copies of3, at their own recorded positions. Its first32 selectors and all32
weights in the first group are included in the report. The map selected by2
is exactly the same16-reference map selected by9 in expert0's first w1/w3
groups; selector IDs are local to each matrix, not global scalar IDs.

Repetition remains in the maps:19525 stored16-reference rows contain only465
distinct byte patterns. Two patterns occur2688 times each. Unique map bytes
alone would occupy7440 bytes, but using a shared map dictionary also needs
references and format metadata;7440 is not a measured replacement-store size.
The complete matrix map lists have515 distinct patterns across2688 records.

The index also carries fixed or derivable fields. All136192 codec fields are1
(zlib); all reserved fields are zero; matrix shapes/counts and first-block
indices derive from matrix position, and offsets follow contiguous lengths.
Reserved fields occupy555528 bytes including the header; codec fields occupy
544768 bytes. These observations identify possible next candidates, not a
claim that the remaining file is irreducible or that a new compact format has
been measured. Dataset/reader hashes and source files remained unchanged.

## Layer 1 direct compact model integration

The user requested a real model run consuming compact expert data without
reconstructing weight arrays. The integration is now verified for **layer1's
routed experts in two five-token prefills**, using a separate copy of the
combined seed/fruit and13-stage model. Other layers, shared experts, source
datasets and frozen earlier implementations remain unchanged.

The [compact consumer](layer1-compact.c) uses the verified
[map-constant dataset](expert-map-results.json), whose scalar values, group maps
and repeated map lists are compiled constants. It reads a compressed block,
inflates only packed codes and group selectors, and looks up one weight at a
time. That weight is widened exactly and multiplied immediately into the
original16 double lanes, with separate multiplication/addition and the same
final reduction tree. It writes projection outputs, not a BF16 or float32
weight block. Block CRC32 is checked by accumulating weight bytes into row
CRCs and combining them, without allocating decoded-weight storage.

The [model wrapper](layer1-compact-model.h) applies this path only to routed
gate/up/down projections in layer1. The original `Xm` computes comparison
outputs into small activation buffers, then the compact path writes the live
outputs. Every output bit must match. SiTU is also compared using the original
gate/up results, including float32 rounding and the uncapped-gate sigmoid.
The existing sampled representation consumers remain active and rewrite their
three checked-equal rows per matrix. They are not weakened or skipped.

### Actual model results

The [independent integration report](layer1-compact-integrated/results.json)
records the following for each full model run:

| Check | France | Japan |
|---|---:|---:|
| Layer1 experts selected | 74 | 74 |
| Complete gate/up/down projection calls | 222 | 222 |
| Projection values compared exactly | 778240 | 778240 |
| SiTU values compared exactly | 245760 | 245760 |
| Compact blocks consumed | 11248 | 11248 |
| Compressed expert bytes read by compact path | 1172494817 | 1172387988 |
| Weight lookups, reused across positions | 2444230656 | 2444230656 |
| Final norm plus logits compared exactly | 171008 floats | 171008 floats |
| All reference routing selections | Exact | Exact |

Together these prompts exercised **82 distinct layer1 experts**, all three
complete matrices for each, rather than all896 experts. The reference routes
independently determine the required expert set and each expert's position
count; the stored block index independently determines expected read bytes.

Final output MD5 remained `23d162dcefb18211a7540ef12948f1eb` for France and
`4b2a7b96fb6323e5639feced66f54a8e` for Japan. The full byte comparisons, not
the emitted token alone, define the result.

The unchanged13-stage validator passed both runs with its existing1926-row,
214-pair and5778-group coverage. Seed input and both fruit output-head
consumers passed their original identity/consumption checks and original-table
guard probes. A compact-disabled France run and the preserved prior combined
model matched the fresh reference. Missing and wrong-format compact datasets
were rejected with exit1 and no final output written.

### What stays compact

The compact consumer's explicit buffers were:

| Allocation | Bytes |
|---|---:|
| Packed codes plus selectors, reused per block | 121856 |
| Compressed input buffer, largest stored block | 110833 |
| Stored index | 1094912 |
| Derived block offsets | 1089544 |
| Row CRC array | 256 |
| Double accumulators per worker, capacity5 positions | 640 |
| Expanded expert-weight arrays in compact path | **0** |

These are explicit consumer allocations, not measured peak RSS or a total
memory bound. Zlib/OpenMP internals, model activations and comparison buffers
are additional. Existing gate/up activation arrays remain; paired-row SiTU
fusion was not part of this first integration.

**This is a correctness integration, not a speed or memory benchmark.** The
original packed expert staging and original projection oracle remain for
comparison, so the run also reads original expert data. No reduction in total
model disk traffic, total RAM or latency is claimed. The compact path itself
consumes the new file directly and does not call the verification decoder that
materializes a BF16 block. A later deployment path must remove duplicate oracle
work while retaining independent validation before performance is measured.

### Reproduce and inspect

The tested implementation and raw run evidence are preserved in
[the integration snapshot](layer1-compact-integrated/model/candidate.c).
The final report pins their hashes, the compiled model, dependencies, original
source/gate/index, and the unchanged map dataset. The large dataset and compiled
execution environment remain on AX102 under
`/opt/clover-k3/layer1-compact-model-20260929-a` and
`/opt/clover-k3/layer1-map-store-20260929-a`.

The [runner](layer1-compact-run.py) includes all required prior flags and refuses
existing output paths. To repeat France on AX102, use a fresh folder name:

```sh
python3 /opt/clover-k3/layer1-compact-model-20260929-a/layer1-compact-run.py \
	/opt/clover-k3/layer1-compact-model-20260929-a repeat-france france \
	/opt/clover-k3/layer1-compact-model-20260929-a/candidate \
	--compact --reference /opt/clover-k3/layer1-compact-model-20260929-a
```

The [independent checker](layer1-compact-check.py) validates route coverage,
byte accounting, prior-stage reports, source identity, seed/fruit checks and
rejection controls. Its final result SHA256 is
`9e398c8b93652dd2fe3e63e078fb39475b0e7cff41897e72fe4f32ee30a7d99f`.
No new generation, longer-context, other-prompt or other-layer compact-consumer
coverage is claimed.

## Named equation functions: seed, root, fruit

The combined C model now has three actual entry points named **seed**, **root**
and **fruit**, following the user's dataset names. They were implemented in a
new isolated copy; the previous verified model remains unchanged. This is a
consumer refactor, not a new weight format or an HTML change.

### Function contracts

```c
void seed(SeedInput *reader, unsigned token, uint16_t output[7168]);

void root(int layer, int expert, int positions,
		  const float *const *input, float *const *output,
		  RootObserver observer, void *context);

void fruit(float *output, const float *input, int expression);
```

| Function | Input | Equation result | What is not reconstructed |
|---|---|---|---|
| `seed` | Open seed reader and one token ID | One7168-value BF16 embedding row | Complete vocabulary embedding table |
| `root` | Layer1, expert ID, batch of3584-value input vectors | One3584-value float32 expert output per position | Any expert weight matrix or BF16/float32 weight block |
| `fruit` | Normalized7168-value input, or the existing normalized-expression mode | 163840 float32 logits | Complete output-head matrix |

The [seed function](seed-function.h) calls the existing bounded row decoder.
Its caller owns reader open/close and dataset-identity validation; bounds and
dimensions are checked by the lookup itself. The current model calls it once
per prompt token, retains only those requested rows, and reuses them for both
input consumers. A row is14336 bytes; the existing compressed format requires
decoding one16-row block, not the entire2.349GB table.

The [root function](root-function.c), declared in [its header](root-function.h),
owns the whole expert equation:

```text
gate = compact_gate(input)
up = compact_up(input)
hidden = SiTU(gate, up)
output = compact_down(hidden)
```

Each projection uses the verified compact block consumer and compiled scalar/
map constants. The original16 double lanes, separate multiply/add, float32
projection rounding, and uncapped-gate sigmoid are preserved. No original
expert-weight pointer is passed to `root`. It rejects any layer other than1
before reading data; the layer argument does not imply support for unconverted
layers. This build supports1-5 positions per call and reuses weights across
the positions selecting the same expert.

`root` retains two3072-value activation vectors per position and reuses the
gate vector for the hidden activation. Its fixed five-position scratch is
122880 bytes, plus pointer arrays. These are equation activations, not weights.
The compact reader still uses the previously verified121856-byte packed block
buffer and110833-byte compressed buffer, shared index and derived offsets.
`RootObserver` is an optional validation callback for gate/up/SiTU/down outputs;
normal computation can pass `NULL` for the observer and its context.

The [fruit function](equation-functions-integrated/model/fruit-head.h) is the
existing streaming projection implementation, now named `fruit` at both call
sites. It decodes bounded16-row BF16 blocks per worker, consumes each weight
directly in the original output-head reduction, and releases the reader after
the pass. `expression=0` uses the supplied normalized vector; `expression=1`
uses the model's existing normalized-source expression and accepts a null input
pointer. Reader/report lifecycle and that expression context remain with the
model adapter. The complete head is never materialized; the returned logits
are the projection result needed by the current model and its correctness gate.

### Verified integration

The [final report](equation-functions-integrated/results.json) verifies both
France and Japan five-token prefills. Each run executed5 seed lookups,74 root
calls covering80 routed position uses, and2 fruit passes (ordinary projection
and the existing independently checked tail expression). Together the runs
exercised82 distinct layer1 experts.

For each prompt, `root` matched245760 gate,245760 up,245760 SiTU, and286720
down-output values exactly against the original computation. Its final output
replaces the complete oracle expert output after the old sampled checks have
run. All171008 final norm/logit float32 values and all routing selections
matched the preserved reference; seed/fruit reports and compact block traces
were unchanged from the prior passing integration. All13 prior representation
gates passed with1926 rows,214 expert/layer pairs and5778 sampled groups.

[API controls](equation-functions-integrated/controls.json) cover seed token
boundaries0 and163839, root stage order with and without an observer, and seven
invalid calls including unsupported layer2. These contract tests use stub
readers; the actual weights and arithmetic are covered by the real model runs.

The [independent checker](equation-functions-check.py) verifies the exact four
modified model files, all unchanged source files, compiled `seed`/`root`/`fruit`
symbols, route-derived call counts, output bytes, all prior gates and dataset/
dependency identities. The [preparation script](equation-functions-prepare.py)
and [run script](equation-functions-run.py) preserve the isolated build and
reproducible launch settings. Tested sources and raw reports are in the
[model snapshot](equation-functions-integrated/model/candidate.c). The executed
environment remains at `/opt/clover-k3/equation-functions-20260930-a` on AX102.

Final report SHA256:
`a14d90ad70a558d0cb16c56646ff6e99b565417f5dad9c0a9f37e2311c1b1710`.

Original packed-weight I/O and oracle computation are still retained for
validation. Buffer sizes are explicit allocations, not total process RSS;
zlib/OpenMP internals and other model activations are additional. No speed,
total-RAM, longer-context, generation or layers2-92 compact-consumer result is
claimed by this refactor.