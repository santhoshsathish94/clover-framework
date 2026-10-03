# Dataset Inventory

## Current Physical Layout

The original dataset paths below are real files/directories. There are no mapping
links anywhere inside dataset. This applies to AX102 and the local small-data and
metadata copy; large numerical payloads remain only on AX102.

```text
dataset/
	inputs/
	outputs/
	tiktoken.model
	vocabulary.bin
	leaves.json
	trunk-0/ ... trunk-92/
	root-1/ ... root-92/
	operators/
		trunk-0-qkv/
		qkv-all/layer-1/ ... layer-92/
	eqidx.bin                       AX102 only, unchanged
	model/                          AX102 only, unchanged
	existing package documentation and historical inventories
```

Mapping links exist only under code/distrubuted-hosting/*/bin: client, normalization
and all92transformers have bin/dataset pointing to this root. Server has a real
bin/dataset directory containing only two links, trunk-0 and trunk-0-qkv, pointing
to the canonical trunk and operators/trunk-0-qkv directories. Linux uses relative
symlinks and Windows uses absolute junctions. Configuration and binaries remain
in bin, unchanged. There are96mapping links across95packages.

The283AX102moves replaced279exact aliases with their own real targets and moved
4client entries to the root. Local284moves include canonical placement of three
server manifest files. Original hashes/timestamps passed for all378moved local
files; all moved remote file/directory identities passed, with SHA256 also checked
for files at most1MiB. Existing external model aliases still resolve to the same
objects. No duplicateQKV0or retired external client alias was recreated.

Both final dataset-link counts are0. AX102 real dataset file count remains3500,
including metadata/documentation. Its client and all94stage loaders passed through
bin mappings; local client/normalization loaders passed. No inference or new data
generation ran. Older package directories retain original documentation companions,
not payloads or links. See [layout](../README.md), [context](../CONTEXT.md) and the
separate DATASET-ORIGINAL-PATHS inventories. Documentation updates follow frozen
payload verification; historical plans and assertions are unchanged.

## Historical Layout Notes

The following notes describe earlier moves into bin and package-grouped central
folders, and are retained as history.
Their physical-location claims are superseded by the central layout above. Paths
through code/.../bin/dataset remain usable because they are now links.

leaves.json now lives in
[normalization/bin/dataset](../code/distrubuted-hosting/normalization/bin/dataset/README.md); its path
here is an AX102 compatibility alias. The normalized-vector stage after layer92
uses only those three global tensors. eqidx.bin remains the original whole-model
index, and model/model-00094-of-000096.safetensors remains the shared embedding,
LM-head and original tail-parameter shard. Neither is private to stage93, and
neither was moved for normalization. Logical dataset totals are unchanged.

The redundant operators/qkv-all/layer-0 folder and its old clover-data alias were
explicitly removed. Its265008216-byte operator payload matched the server's
canonical QKV file byte-for-byte; the distinct manifest was moved unchanged into
server/bin/dataset/trunk-0-qkv/qkv-all-layer-0-manifest.json. The server still loads
its original qkv.bin. See the [cleanup journal](../code/distrubuted-hosting/server/QKV-DEDUP-JOURNAL.tsv).
All other model aliases are retained. Historical inventories below are snapshots
from before this duplicate removal, not instructions to recreate the duplicate.

Layers 2-92 now follow the transformer-1 package arrangement. Their trunk-N,
root-N and operators/qkv-all/layer-N directories reside in each
transformer-N/bin/dataset; the paths here are compatibility aliases. All 273
directory moves preserved 3,325 original files and their full hashes/identities.
See the [complete layer index](../code/distrubuted-hosting/tansformers/remaining/LAYERS.md) and
[campaign evidence](../code/distrubuted-hosting/tansformers/remaining/README.md). The local main
dataset folder still contains historical inventories, not the numerical payloads.

The trunk-0 QKV operator and complete trunk-0 directory have moved into
[server dataset notes](server/README.md). Their old operators/trunk-0-qkv
and trunk-0 paths here are AX102 compatibility aliases. The real common/KDA/dense
files now supply the server's remaining layer-0 operations from its own dataset.
Client-only removed aliases stay removed; no unrelated dataset was moved.

The subsequent layer-1 package now owns trunk-1, root-1 and
operators/qkv-all/layer-1 under
[transformer-1/bin/dataset](../code/distrubuted-hosting/tansformers/transformer-1/bin/dataset/README.md).
All forty files moved unchanged, including existing metadata and observations.
Old paths here remain aliases. The transformer computes live using stored
parameter values; historical expert observations are not runtime inputs.

The main model datasets are on AX102 at:

```text
/opt/clover-k3/clover-intelegence/dataset
```

Client inputs, outputs and vocabulary now live under
`/opt/clover-k3/clover-intelegence/code/distrubuted-hosting/client/bin/dataset`. Their previous paths
in the main dataset folder were removed with explicit approval. There are no
client data aliases or duplicate client runtime files in this directory.

This local folder holds inventory metadata, not the1.388TBnumeric payload or the
small vocabulary files, which moved into the client package. Numerical data was
moved unchanged; nothing was regenerated. See [context](../CONTEXT.md).

- [MOVE-PLAN.json](MOVE-PLAN.json):373source/destination units,2328file identities.
- [MOVE-JOURNAL.jsonl](MOVE-JOURNAL.jsonl):one durable entry after each verified move.
- [RELOCATION.json](RELOCATION.json):final complete relocation checks.

The subsequent complete-root extension is recorded separately:

- [ROOTS-MOVE-PLAN.json](ROOTS-MOVE-PLAN.json):92roots,inputs,outputs,leaves.json.
- [ROOTS-MOVE-JOURNAL.jsonl](ROOTS-MOVE-JOURNAL.jsonl):started/completed entries in move order.
- [ROOTS-RELOCATION.json](ROOTS-RELOCATION.json):3398combined files/1,388,069,070,299bytes;
	all previous observations and payload identities preserved.

These files contain locations, sizes and file identity metadata, not model outputs.
The first journal sequence is index,prepared trunk0-92,derived operators,existing
expert observations and the global model tensor shard. The extension sequence is
complete root1-92,inputs,outputs,leaves.json. Original root observation paths now
resolve through the whole-root aliases instead of individual child links. Old
model paths remain usable except the explicitly retired qkv-all/layer-0 paths;
client input/output/vocabulary/config aliases were also removed. See the
[layout and launch instructions](../README.md).

The subsequent text-mapping move added
[tiktoken.model](../code/distrubuted-hosting/client/bin/dataset/tiktoken.model) and
[vocabulary.bin](../code/distrubuted-hosting/client/bin/dataset/vocabulary.bin), totaling4,560,217bytes.
The logical dataset inventory is3399files/1,387,808,622,300bytes across the main,
client, server and transformer datasets, excluding relocation records/docs and counting
aliases only once. [tokenizer_config.json](../code/distrubuted-hosting/client/bin/configs/tokenizer_config.json)
is now stored under code/distrubuted-hosting/client/bin/configs. These small assets have identical local
copies for C testing. Their first move is in
[the text move plan](../TOKENIZER-MOVE-PLAN.tsv); the current runtime locations and
unchanged payload hashes are in [the bin move plan](../CLIENT-BIN-MOVE-PLAN.tsv) and
[completed journal](../CLIENT-BIN-MOVE-JOURNAL.tsv). The
[removed aliases](../CLIENT-BIN-ALIASES.tsv) are historical records, not live paths. See the
[client package](../code/distrubuted-hosting/client/README.md) for current commands and local data limits.